#!/usr/bin/env python3
import json
import os
import tarfile
import argparse
import os.path
import shutil
import io
import clang.cindex
import re
import sys
import contextlib
import tempfile
import subprocess
import pygit2

import pandas as pd

from zstandard import ZstdDecompressor
from collections import defaultdict
from tempfile import TemporaryDirectory
from hashlib import sha256
from strace_parser import StraceParser
from pathlib import Path

def gcc_cpp_include_paths(gcc_version):
    return [
        f'/usr/include/c++/{gcc_version}',
        f'/usr/include/x86_64-linux-gnu/c++/{gcc_version}',
        f'/usr/include/c++/{gcc_version}/backward',
    ]

def get_clang_resource_root():
    return subprocess.run(["clang", "-print-resource-dir"], stdout=subprocess.PIPE).stdout.decode().strip()

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

def extract_flags(fname, commands):
    for command in commands:
        if not any(arg.split('/')[-1] == fname for arg in command['argv'] if arg):
            continue
        #print(" ".join(command['argv']))
        parsed = parse_cflags(command['argv'])
        yield command, parsed


RE_GCC_VERSION = re.compile(r'^/usr/lib/gcc/x86_64-linux-gnu/(.*)/cc1$')
def extract_gcc_version(commands):
    for command in commands:
        match = RE_GCC_VERSION.match(command['exe'])
        if match:
            return match.group(1)

RE_PACKAGE = re.compile(r'^Package: (?P<name>[^\n]+)$')
RE_VERSION = re.compile(r'^Version: (?P<version>[^\n]+)$')
RE_ARCH = re.compile(r'^Architecture: (?P<arch>[^\n]+)$')
def parse_dpkg_status(status_file):
    pkg = None
    version = None
    arch = None
    for i, line in enumerate(status_file.split("\n")):
        if not line.strip():
            if len(set(x is None for x in (pkg, version, arch))) != 1:
                print("pkg, version or arch missing", pkg, version, arch, i, file=sys.stderr)
                version = None
                pkg = None
                arch = None
                continue

            if pkg is not None:
                yield pkg, version, arch

            pkg = None
            version = None
            arch = None

        match = RE_PACKAGE.match(line)
        if match is not None:
            pkg = match.group('name')

        match = RE_VERSION.match(line)
        if match is not None:
            version = match.group('version')

        match = RE_ARCH.match(line)
        if match is not None:
            arch = match.group('arch')



def parse_dpkg_deps(out_dir):
    path_old = out_dir + "/dpkg.tar.zst"
    path_new = out_dir + "/dpkg-fixed.tar.zst"

    path = None
    if os.path.exists(path_old):
        path = path_old
        fname = "./var/lib/dpkg/status"

    if os.path.exists(path_new):
        path = path_new
        fname = "./status"

    if path is None:
        print("error: dpkg tar does not exists", out_dir, file=sys.stderr)
        return []

    data = subprocess.run(["tar", "-Oxf", path, fname], stdout=subprocess.PIPE).stdout.decode()
    return list(parse_dpkg_status(data))

def matcher_tree(all_include_dirs):
    include_dirs_matcher = {
        'usr': {
            'include': True,
            'local': {
                'include': True,
            }
        }
    }

    for d in sorted(all_include_dirs):
        if not d.startswith('/'):
            continue
        d = d[1:]
        components = d.split('/')
        where = include_dirs_matcher
        while where != True:
            c = components.pop(0)
            if not components:
                where[c] = True
                break

            where.setdefault(c, {})
            where = where[c]

    return include_dirs_matcher

def resolve(root, cwd, path):
    if path.startswith('/'):
        return os.path.normpath(root + path)
    return os.path.normpath(root + '/' + cwd + '/' + path)


RE_VERSION_QUAL = re.compile(r"^[0-9]+:")
def setup_work_dir(root_dir, out_dir, pool_dir="debian-pool"):
    print("parsing dpkg debs")
    deps = parse_dpkg_deps(out_dir)

    print("preparing build dir")
    build_dir = root_dir + "/build"
    os.makedirs(build_dir, exist_ok=True)
    subprocess.run(["tar", "--zstd", "-C", build_dir, "-xf", out_dir + "/build.tar.zst"])

    for pkg, version, arch in deps:
        if pkg.endswith("-build-deps"):
            continue
        version = RE_VERSION_QUAL.sub("", version)
        deb_path = pool_dir + f"/{pkg}_{version}_{arch}.deb"
        print("extracting", deb_path)
        subprocess.run(['dpkg', '-x', deb_path, root_dir])
    for dirpath, dirnames, fnames in os.walk(root_dir):
        for f in dirnames + fnames:
            entry = dirpath + "/" + f
            if os.path.islink(entry):
                target = os.readlink(entry)
                if target.startswith("/"):
                    os.unlink(entry)
                    os.symlink(root_dir + target, entry)

def load_commands(out_dir):
    print("loading commands")
    commands = []
    with ZstdDecompressor().stream_reader(open(out_dir + '/strace.zst', 'rb')) as f:
        strace_parser = StraceParser('/build')
        buffer = b""
        while True:
            data = f.read(1024 * 1024)
            if not data:
                break
            buffer += data
            new_commands, buffer = strace_parser.process_buffer(buffer)
            commands.extend(c.to_dict() for c in new_commands)
        new_commands, _ = strace_parser.process_buffer(buffer, final=True)
        commands.extend(c.to_dict() for c in new_commands)
        print("done loading commands")
        return commands


def make_flags(root_dir, cmd, parsed, gcc_version, clang_builtin):
    includes = [resolve(root_dir, cmd['workdir'], f) for f in parsed['includes'] + gcc_cpp_include_paths(gcc_version) ]
    includes += [clang_builtin]
    includes += [root_dir + i for i in STDINC]
    flags = ['-nostdinc'] + ['-I' + d for d in includes] + ['-D' +d for d in parsed['defines']] + ['-f' + f for f in parsed['fflags']]
    return flags


def diag_to_dict(diag: clang.cindex.Diagnostic):
    return {
        "message": diag.spelling,
        "severity": diag.severity,
        "category": diag.category_name,
        "file": diag.location.file.name if diag.location.file is not None else None,
        "line": diag.location.line,
        "column": diag.location.column,
    }


def read_file(path):
    try:
        with open(path) as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, 'rb') as f:
            return list(f.read())


def patch_from_repo(args, root_dir, sample_details):
    src_dir = root_dir + "/build/src"
    repo = pygit2.Repository(args.repos_dir + "/" + sample_details['repo_name'])
    tree = repo[sample_details['commit']].tree

    repo.checkout_tree(tree, directory=src_dir, strategy=pygit2.GIT_CHECKOUT_FORCE|pygit2.GIT_CHECKOUT_RECREATE_MISSING)


STDINC = ["/usr/local/include", "/usr/include/x86_64-linux-gnu", "/usr/include"]
RE_LINE_NUMBER = re.compile(r"^[ \t]*[0-9]+[ \t]*", re.MULTILINE)
def main(args, metadata, cve_files_path):
    cve_files_path = cve_files_path.set_index(['hash', 'label'])

    os.makedirs(args.tmp, exist_ok=True)

    pkg_id = f"{metadata['name']}-{metadata['version']}"
    if 'commit' in metadata:
        pkg_id = pkg_id + "-" + metadata['commit']
    out_dir = args.build_dir + "/" + pkg_id
    commands = load_commands(out_dir)
    clang_builtin = get_clang_resource_root() + "/include"
    results_dir = args.results
    os.makedirs(results_dir, exist_ok=True)

    with contextlib.ExitStack() as stack:
        root_dir = args.root_dir
        if not root_dir and not args.keep_tmp:
            root_dir = stack.enter_context(TemporaryDirectory(prefix='root-', dir=args.tmp))
        if not root_dir:
            root_dir = tempfile.mkdtemp(prefix="root-", dir=args.tmp)

        if not args.root_dir or not os.path.exists(args.root_dir):
            setup_work_dir(root_dir, out_dir, args.pool_dir)

        gcc_version = extract_gcc_version(commands)
        print("GCC version", gcc_version)
        for sample in metadata['samples']:
            sample_details = cve_files_path.loc[sample['hash'], sample['label']]

            if args.patch_from_repo and not 'commit' in metadata:
                patch_from_repo(args, root_dir, sample_details)

            h = sha256(sample['code']['file'].encode()).hexdigest()
            sample_results = results_dir + f"/{sample['cve']}-{sample['filename']}-{h[:8]}"
            os.makedirs(sample_results, exist_ok=True)

            if os.path.lexists(sample_results + "/success"):
                continue

            with open(sample_results + '/sample.json', 'w') as f:
                json.dump(sample, f)

            print('##', sample['filename'])
            for cmd, parsed in extract_flags(sample['filename'], commands):
                cwd = cmd['workdir']
                if cwd is None:
                    print("no cwd", cmd)
                    continue

                flags = make_flags(root_dir, cmd, parsed, gcc_version, clang_builtin)

                fnames = [resolve(root_dir, cwd, f) for f in parsed['filenames']]
                fnames = [f for f in fnames if f and f.split('/')[-1] == sample['filename']]
                for i, fname in enumerate(fnames):
                    cx = clang.cindex.Index.create()
                    if not os.path.exists(fname):
                        continue

                    if args.patch_from_repo:
                        tu = cx.parse(fname, flags)
                    else:
                        tu = cx.parse(fname, flags, [(fname, RE_LINE_NUMBER.sub("", sample['code']['file']))])

                    config_hash = sha256(str(i).encode() + b":" + json.dumps(parsed, sort_keys=True).encode()).hexdigest()
                    config_name = f"{os.path.basename(cmd['exe'])}-{config_hash[:8]}"

                    with open(sample_results + f"/diag-{config_name}.json", 'w') as f:
                        json.dump({
                            'pkg_name': metadata['name'],
                            'pkg_version': metadata['version'],
                            'pkg_id': pkg_id,
                            'diagnostics': list(diag_to_dict(diag) for diag in tu.diagnostics),
                            'cmd': cmd,
                            'parsed': parsed,
                            'flags': flags,
                            'fname': fname,
                        }, f)

                    paths = set(i.include.name for i in tu.get_includes())
                    files = [(os.path.normpath('/fakeroot/' + path[len(root_dir):]), read_file(path)) for path in paths if path.startswith(root_dir)]

                    with open(fname, 'rb') as f:
                        content = f.read()
                    with open(sample_results + f"/results-{config_name}.json", 'w') as f:
                        json.dump({
                            'files': files,
                            'flags': make_flags('/fakeroot/', cmd, parsed, gcc_version, clang_builtin),
                            'parsed': parsed,
                            'cmd': cmd,
                            'file_bytes': list(content),
                            'fname': os.path.normpath('/fakeroot/' + fname[len(root_dir):])
                        }, f)

                    if all(diag.severity < clang.cindex.Diagnostic.Error for diag in tu.diagnostics):
                        if not fname.endswith(sample_details['path']):
                            continue
                        try:
                            Path(sample_results + "/success").symlink_to(f'results-{config_name}.json')
                        except FileExistsError:
                            pass
                    else:
                        print("failed", sample_results, config_name)
                        with open(sample_results + f"/file-{config_name}.c", 'w') as f:
                            f.write(RE_LINE_NUMBER.sub("", sample['code']['file']))
                        for diag in [diag for diag in tu.diagnostics if diag.severity >= clang.cindex.Diagnostic.Error][:3]:
                            print(diag)



def is_ipython():
    try:
        __IPYTHON__
    except NameError:
        return False
    else:
        return True

def load_metadata(spec):
    with tarfile.open(spec) as f:
        return json.load(f.extractfile('package.json'))



if __name__ == '__main__' and not is_ipython():
    parser = argparse.ArgumentParser(description="Try to compile the samples in a package spec using the given build context")
    parser.add_argument("spec", metavar="SPEC_TAR", help="The spec to build")
    parser.add_argument("--build-dir", metavar="DIR", help="Directory containing build outputs for specs. There must be a subdirectory for the package to process", default="build")
    parser.add_argument("--pool-dir", metavar="DIR", help="Directory containing debian binary packages for dependencies", default="debian-pool")
    parser.add_argument("--results", metavar="DIR", help="Directory containing the results", default="compile-sample-files")
    parser.add_argument("--tmp", metavar="DIR", help="Location of a temporary directory used for temporary artifacts", default="/tmp/compile")
    parser.add_argument("--root-dir", metavar="DIR", help="Continue previous build")
    parser.add_argument("--keep-tmp",action='store_true', help="Don't delete temporary directory")
    parser.add_argument("--patch-from-repo", action='store_true', help="Copy the files at the state of the patch from the repo")
    parser.add_argument("--file-details", help="Data file with details about the sample files")
    parser.add_argument("--repos-dir", help="Directory containing source repos")


    args = parser.parse_args()
    cve_files_path = pd.read_json(args.file_details)
    main(args, load_metadata(args.spec), cve_files_path)
