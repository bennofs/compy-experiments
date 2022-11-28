#!/usr/bin/env python3
import argparse
import sys
import os
import subprocess
import json
import tarfile
import tempfile
import threading
import contextlib
import shutil
import glob

from pathlib import Path

from runner import Runtime
from debootstrap import debootstrap, snapshot_mirror

def build(runtime: Runtime, work_dir, metadata, args):
    rootfs = work_dir + "/" + "rootfs"
    build_dir = rootfs + "/build"
    dist = metadata['dist'].split("-")[0]
    mirror = snapshot_mirror(args.mirror, metadata['dsc_info']['first_seen'])
    package_id = f"{metadata['name']}-{metadata['version']}"
    out_dir = f"{args.out}/{package_id}"
    os.makedirs(out_dir, exist_ok=True)

    if not args.build_only:
        # prepare build environment
        debootstrap(runtime, dist, rootfs, mirror)
        os.makedirs(build_dir, exist_ok=True)
        with tarfile.open(args.spec) as f:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(f, build_dir)

        # extract source and install dependencies
        dsc_name = next(name for name in os.listdir(build_dir) if name.endswith(".dsc"))

        runtime.spawn(runtime.config_container(rootfs, [
            "apt-get", "-y", "--force-yes", "install", "devscripts"
        ]))
        runtime.spawn(runtime.config_container(rootfs, ["dpkg-source", "-x", dsc_name, "src"], workdir="/build"))
        runtime.spawn(runtime.config_container(rootfs, ["mk-build-deps", "./debian/control"], workdir="/build/src"))
        build_deps_deb = glob.glob(build_dir + "/src/*.deb")[0]
        shutil.copy(build_deps_deb, f"{out_dir}/build-deps.deb")

        apt_cmd = f'dpkg -i "{os.path.basename(build_deps_deb)}" || env DEBIAN_FRONTEND=noninteractive apt-get -o Debug::pkgProblemResolver=yes -y --force-yes --no-install-recommends -f install'
        if args.fetch_only:
            apt_cmd = apt_cmd + " --download-only"

        runtime.spawn(runtime.config_container(rootfs, ['bash', '-c', apt_cmd], workdir="/build/src"))

        if args.fetch_only:
            return

    # perform the build under strace
    print("starting build")

    strace_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'strace-static'))
    shutil.copy(strace_path, rootfs + "/strace-static")
    os.chmod(rootfs + "/strace-static", 0o755)
    strace_cmd =  ["/strace-static", "-xx", "--seccomp-bpf", "-f", "-e", "execve,chdir,fchdir,execveat,fork,vfork,clone,%process", '-y', '-s', '999999999', "-o", "/strace"]
    c = runtime.config_container(rootfs, strace_cmd + ["debian/rules", "build"], workdir="/build/src/")
    with contextlib.ExitStack() as stack:
        stdout = stack.enter_context(open(work_dir + '/stdout', 'w'))
        stderr = stack.enter_context(open(work_dir + '/stderr', 'w'))

        proc = stack.enter_context(runtime.spawn_background(c, stdout=stdout, stderr=stderr))

        def save_output():
            print("build done, storing output", file=sys.stderr)
            subprocess.run(["zstd", '-f', work_dir + "/stdout", "-o", out_dir + "/stdout.zst"])
            subprocess.run(["zstd", '-f', work_dir + "/stderr", "-o", out_dir + "/stderr.zst"])
            subprocess.run(["zstd", '-f', rootfs + "/strace", "-o", out_dir + "/strace.zst"])
            subprocess.run(["tar", "--zstd", "-C", build_dir, "-f", out_dir + "/" + "build.tar.zst", '-c', '.'])
            subprocess.run(["tar", "--zstd", "-C", rootfs + "/var/lib/dpkg"  , "-f", out_dir + "/" + "dpkg-fixed.tar.zst", '-c', '.'])
        stack.callback(save_output)

        proc.wait()

        if proc.returncode == 0:
            Path(out_dir + "/success").touch()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build a package spec and extract compile commands and files required for building the samples")
    parser.add_argument("spec", metavar="SPEC", help="A package spec tar file")
    parser.add_argument("--tmp", metavar="DIR", help="Location of a temporary directory used for temporary artifacts", default="/tmp/builder")
    parser.add_argument("--mirror", metavar="URL", help="URL of a snapshot.debian.org mirror", default="http://snapshot.debian.org")
    parser.add_argument("--out", metavar="DIR", help="Directory where to stoe the result", default="build")
    parser.add_argument("--keep-tmp", action='store_true', help="Don't cleanup temporary files")
    parser.add_argument("--fetch-only", action='store_true', help="Exit after downloading all the build dependencies")
    parser.add_argument("--build-only", metavar="WORKDIR", help="Only perform the build in the given already-prepared workdir")

    args = parser.parse_args()
    os.makedirs(args.tmp, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    args.tmp = os.path.abspath(args.tmp)
    args.out = os.path.abspath(args.out)
    args.spec = os.path.abspath(args.spec)

    with tarfile.open(args.spec, 'r') as tar:
        metadata = json.load(tar.extractfile('package.json'))

    runtime = Runtime()
    pkgid = f"{metadata['name']}-{metadata['version']}"
    if args.build_only:
        work_dir = os.path.abspath(args.build_only)
    else:
        work_dir = tempfile.mkdtemp(dir=args.tmp, prefix=pkgid + "-")
    try:
        build(runtime, work_dir, metadata, args)
    finally:
        if not args.keep_tmp and not args.build_only:
            runtime.spawn(runtime.config_host(["rm", "-rf", "--one-file-system", work_dir]))
