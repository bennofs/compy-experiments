#!/usr/bin/env python3
import argparse
import base64
import contextlib
import fcntl
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from hashlib import sha256
from typing import Iterable
from tempfile import TemporaryDirectory

import joblib
import pandas as pd
import pygit2
from tqdm import tqdm

from lib.debootstrap import debootstrap
from lib.runner import Runtime
from lib import config, sources
from lib.strace_parser import StraceParser

MAX_CHANGED_FUNCTIONS = 10


def add_src_to_sources_list(rootfs: Path):
    """Modify the apt sources list to include deb-src sources"""
    with (rootfs / "etc/apt/sources.list").open('r') as sources_list:
        deb_lines = [l for l in list(sources_list) if l.startswith('deb ')]
    with (rootfs / "etc/apt/sources.list").open('a') as sources_list:
        for l in deb_lines:
            sources_list.write("\ndeb-src " + l[4:])
        sources_list.write("\n")


def install_build_deps(runtime: Runtime, rootfs: Path):
    runtime.spawn(runtime.config_container(rootfs, [
        "apt-get", "update"
    ]))
    runtime.spawn(runtime.config_container(rootfs, [
        'env', 'DEBIAN_FRONTEND=noninteractive',
        'apt-get', '--force-yes', '-y', 'build-dep', 'ffmpeg'
    ]))


def run_build(tmpdir: Path, runtime: Runtime, rootfs: Path, jobs=1):
    runtime.spawn(runtime.config_container(rootfs, [
        "bash", "configure"
    ], workdir="/src"))

    strace_path = config.get_tracing_helpers_path() / 'strace-static'
    shutil.copy(strace_path, rootfs / "strace-static")
    os.chmod(rootfs / "strace-static", 0o755)

    strace_cmd = ["/strace-static", "-xx", "--seccomp-bpf", "-f", "-e",
                  "execve,chdir,fchdir,execveat,fork,vfork,clone,%process", '-y', '-s', '999999999', "-o", "/strace"]
    c = runtime.config_container(rootfs, strace_cmd + ["make", f"-j{jobs}"], workdir="/src/")
    proc = runtime.spawn(c, check=False)

    build_success = proc.returncode == 0
    print("build success" if build_success else "build fail", file=sys.stderr)
    return build_success


def parse_strace_commands(rootfs: Path):
    strace_parser = StraceParser('/src')
    buffer = b""
    commands = []
    with open(rootfs / 'strace', 'rb') as f:
        while True:
            data = f.read(1024 * 1024)
            if not data:
                break
            buffer += data
            new_commands, buffer = strace_parser.process_buffer(buffer)
            commands.extend(c.to_dict() for c in new_commands)
    new_commands, _ = strace_parser.process_buffer(buffer, final=True)
    commands.extend(c.to_dict() for c in new_commands)
    return commands


def parse_cflags(argv):
    includes = []
    defines = []
    fflags = []
    filenames = []
    i = 1
    while i < len(argv):
        flag = argv[i]
        i += 1

        if flag == '-isystem' and i + 1 < len(argv):
            includes.append(argv[i])
            i += 1
            continue

        if len(flag) == 2 and i + 1 < len(argv) and flag in {'-I', '-D'}:
            arg = argv[i]
            i += 1
        else:
            arg = flag[2:]

        if flag.startswith('-I'):
            includes.append(arg)
        if flag.startswith('-D'):
            defines.append(arg)
        if flag.startswith('-f'):
            fflags.append(arg)

        is_arg = not (flag.startswith("-") or (i > 0 and argv[i-1].startswith('-') and (len(argv[i-1]) == 2 or argv[i-1][:2] not in {'-I', '-D', '-W', '-f'})))
        if is_arg and i != 0:
            filenames.append(flag)

    return {
        'includes': includes,
        'defines': defines,
        'fflags': fflags,
        'filenames': filenames,
    }


def find_commands_for_file(fname, commands):
    for command in commands:
        if not any(arg.split('/')[-1] == fname for arg in command['argv'] if arg):
            continue
        yield command


def fix_mpglib(datadir: Path, srcdir: Path):
    f = srcdir / "libavcodec/mpegaudiodec.c"
    if not f.exists() or "mpglib" not in f.read_text(): return
    if (srcdir / "libavcodec/mpglib").exists(): return

    shutil.copytree(datadir / "mpglib", srcdir / "libavcodec/mpglib")


def fix_libac3(datadir: Path, srcdir: Path):
    f = srcdir / "libavcodec/ac3dec.c"
    if not f.exists() or "libac3" not in f.read_text(): return
    if (srcdir / "libavcodec/libac3").exists(): return

    shutil.copytree(datadir / "libac3", srcdir / "libavcodec/libac3")


def apply_patches(datadir: Path, srcdir: Path):
    fix_mpglib(datadir, srcdir)
    fix_libac3(datadir, srcdir)


def process_file(runtime: Runtime, rootfs: Path, fpath: str, commands):
    fname = os.path.basename(fpath)
    container_path = os.path.normpath('/src/' + fpath)
    for command in find_commands_for_file(fname, commands):
        parsed = parse_cflags(command['argv'])
        cmd = ['gcc']
        cmd += ['-f' + f for f in parsed['fflags']]
        cmd += ['-I' + i for i in parsed['includes']]
        cmd += ['-D' + d for d in parsed['defines']]
        cmd += ['-E', container_path]

        result = runtime.spawn(runtime.config_container(rootfs, cmd, workdir=command['workdir']),
                               check=False, stdout=subprocess.PIPE)
        if result.returncode == 0: break
    else:
        return None

    return {
        'preprocessed': base64.b64encode(result.stdout).decode(),
        'command': command,
        'flags': parsed,
    }


@contextlib.contextmanager
def open_lock(lock_path: Path, mode='r', flags=fcntl.LOCK_SH):
    with open(lock_path, mode) as f:
        fcntl.lockf(f, flags)
        try:
            yield f
        finally:
            fcntl.lockf(f, fcntl.LOCK_UN)


def restore_from_cache(runtime: Runtime, cache_root: Path, cache_key: str, dest: Path):
    cache_path = cache_root / cache_key
    lock_path = cache_root / (cache_key + ".lock")

    if not cache_path.exists():
        logging.info("not using cache, %s not found cache", cache_key)
        return

    try:
        with open_lock(lock_path):
            dest.mkdir(parents=True, exist_ok=True)
            runtime.spawn(runtime.config_host([
                'rsync', '-ra', str(cache_path) + '/', str(dest) + '/'
            ]))
            logging.info("successfully restored %s from cache", cache_key)
        return True
    except (IOError, subprocess.CalledProcessError) as e:
        logging.info("failed to restore %s from cache: %s", cache_key, str(e))


def save_to_cache(runtime: Runtime, cache_root: Path, cache_key: str, src: Path):
    cache_path = cache_root / cache_key
    lock_path = cache_root / (cache_key + ".lock")
    cache_root.mkdir(exist_ok=True, parents=True)

    with open_lock(lock_path, mode='w', flags=fcntl.LOCK_EX):
        try:
            if cache_path.exists():
                logging.warning("%s already exists in cache, not overwriting", cache_key)
                return
            runtime.spawn(runtime.config_host([
                'rsync', '-ra', str(src) + '/', str(cache_path) + '/'
            ]))
            logging.info("successfully saved %s to cache", cache_key)
            return True
        except (subprocess.CalledProcessError, IOError) as e:
            logging.info("failed to save %s to cache: %s", cache_key, str(e))
            shutil.rmtree(cache_path)
            lock_path.unlink()



def prepare_rootfs(runtime, args, rootfs):
    # check if there is a cached rootfs
    cache_key = f'{args.dist}-{args.arch}-{sha256(args.mirror.encode()).hexdigest()[:16]}'
    cache_root = Path('/nonexistentpath' if args.rootfs_cache is None else args.rootfs_cache)

    if not rootfs.exists() and restore_from_cache(runtime, cache_root, cache_key, rootfs):
        return

    debootstrap(runtime, args.dist, rootfs, args.mirror, arch=args.arch)

    # install build dependencies
    add_src_to_sources_list(rootfs)
    install_build_deps(runtime, rootfs)

    # these versions don't have the yasm package
    if args.dist not in {'hamm', 'slink', 'potato', 'sarge'}:
        runtime.spawn(runtime.config_container(rootfs, [
            'env', 'DEBIAN_FRONTEND=noninteractive',
            'apt-get', '--force-yes', '-y', 'install', 'yasm', 'nasm',
        ]))

    if args.rootfs_cache is not None:
        save_to_cache(runtime, cache_root, cache_key, rootfs)


def main(args, cleanup):
    runtime = Runtime()
    tmpdir = Path(str(cleanup.enter_context(TemporaryDirectory(prefix="build-ffmpeg-"))))
    logging.info("building in dir %s", tmpdir)

    rootfs = tmpdir / "rootfs"
    prepare_rootfs(runtime, args, rootfs)

    # create checkout of src repo at the base revision
    with open(args.spec) as f:
        spec = json.load(f)
    sources.create_detached_checkout(args.repo, rootfs / "src", spec['commit_base'])
    apply_patches(Path(args.data), rootfs / 'src')

    # run the build
    build_success_before = run_build(tmpdir, runtime, rootfs)

    # load build commands
    commands_before = parse_strace_commands(rootfs)

    # process each changed file
    processed_before = {}
    failed_before = set()
    for path in spec['commit_paths_before']:
        result = process_file(runtime, rootfs, path, commands_before)
        if result is not None:
            processed_before[path] = result
        else:
            failed_before.add(path)

    # now checkout the fixed revision, and re-run the build
    # first try to run an incremental build
    # if that fails, clean and run a full build
    dest_repo = pygit2.Repository(rootfs / "src")
    dest_repo.reset(spec['sha_id'], pygit2.GIT_RESET_HARD)
    apply_patches(Path(args.data), rootfs / 'src')
    build_success_after = run_build(tmpdir, runtime, rootfs)
    if not build_success_after:
        shutil.rmtree(rootfs / "src")
        sources.create_detached_checkout(args.repo, rootfs / "src", spec['sha_id'])
        apply_patches(Path(args.data), rootfs / 'src')
        build_success_after = run_build(tmpdir, runtime, rootfs)

    commands_after = parse_strace_commands(rootfs)
    processed_after = {}
    failed_after = set()
    for path in spec['commit_paths_after']:
        result = process_file(runtime, rootfs, path, commands_after)
        if result is None:
            result = process_file(runtime, rootfs, path, commands_before)

        if result is not None:
            processed_after[path] = result
        else:
            failed_after.add(path)

    result = {
        'before': {
            'build_success': build_success_before,
            'files': processed_before,
            'failed': list(failed_before),
        },
        'after': {
            'build_success': build_success_after,
            'files': processed_after,
            'failed': list(failed_after),
        },
    }
    with open(args.out, 'w') as f:
        json.dump(result, f)


class NoopExitStack():
    def enter_context(self, v):
        return v.__enter__()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Compile a ffmpeg build spec and extract samples")
    parser.add_argument("--repo", required=True, help="Path to ffmpeg git repo")
    parser.add_argument("--no-cleanup", default=False, action='store_true', help="Don't remove temporary files on exit (useful for debugging)")
    parser.add_argument("--dist", default="stable", help="Debian distro to use for building")
    parser.add_argument("--arch", default="amd64", help="Debian architecture to use for building")
    parser.add_argument("--mirror", default="https://deb.debian.org/debian", help="Debian mirror for building")
    parser.add_argument("--data", required=True, help="Path to the directory containing extra data needed for building")
    parser.add_argument("--rootfs-cache", help="If specified, cache debootstrap rootfs in this directory")
    parser.add_argument("spec", help="Spec to build")
    parser.add_argument("out", help="Result filename")

    ARGS = parser.parse_args()
    del parser

    with contextlib.ExitStack() as _stack:
        R = main(ARGS, NoopExitStack() if ARGS.no_cleanup else _stack)
        pass