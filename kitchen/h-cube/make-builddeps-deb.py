#!/usr/bin/env python3
import argparse
import os
import os.path
import json
import sys
import subprocess
import glob
import shutil

from hashlib import sha1
from tempfile import TemporaryDirectory

import helpers

def main(args):
    deb_cache = None

    if args.cache_dir is not None:
        deb_cache = os.path.abspath(args.cache_dir)
        os.makedirs(deb_cache, exist_ok=True)

    args.out_dir = os.path.abspath(args.out_dir)
    args.build_dir = os.path.abspath(args.build_dir)
    args.package = os.path.abspath(args.package)
    args.files = os.path.abspath(args.files)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.build_dir, exist_ok=True)

    with open(args.package, 'rb') as f:
        package_dsc = f.read()
    package_sha1 = sha1(package_dsc).hexdigest()

    with open(args.specs, 'r') as f:
        parsed = (json.loads(l) for l in f)
        package_spec = next(spec for spec in parsed if spec['sha1'] == package_sha1)
    package_id = f"{package_spec['name']}-{package_spec['version']}"
    dist = package_spec['dists']['testing']

    result_path = f"{args.out_dir}/build-deps-{dist}-{package_id}.deb"
    if os.path.exists(result_path):
        return

    with TemporaryDirectory(dir=args.build_dir, prefix="tmp-" + package_id) as temp_dir:
        os.chdir(temp_dir)
        src_path = os.path.dirname(args.package)
        cache_mount = f"type=bind,source={deb_cache}" if deb_cache is not None else "type=tmpfs"
        podman_common = [
            'podman', 'run', '--rm', "--entrypoint=",
            "--mount", f"type=bind,source={src_path},destination=/srcpkg",
            "--mount", f"type=bind,source={temp_dir},destination=/build",
            "--mount", f"{cache_mount},destination=/cache",
            "--mount", f"type=bind,source={args.files},destination={args.files}",
        ]

        # extract source
        dsc_name = os.path.basename(args.package)
        subprocess.run(podman_common + [
            "--workdir", "/srcpkg",
            args.dev_image + dist,
            "dpkg-source", "-x", "--no-check", "--no-copy", f"{dsc_name}", "/build/src"
        ], check=True)

        # create builddeps package
        subprocess.run(podman_common + [
            "--workdir", "/build",
            args.dev_image + dist,
            "mk-build-deps", "src/debian/control"
        ], check=True)

        # copy builddeps to output dir
        builddeps = glob.glob("*.deb")
        shutil.move(builddeps[0], result_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Make a builddeps debian package for a given source pakage")
    parser.add_argument('--cache-dir', help='Directory for cache files. If not specified, no cache files are writting to disk')
    parser.add_argument('--build-dir', help='Directory for temporary build files', default="/tmp/build")
    parser.add_argument('--snapshot-host', help='Address of the debian snapshot service. Set to a caching proxy for increased peformance.', default='snapshot.debian.org')
    parser.add_argument('--out-dir', help="Directory for the output file", default="build-deps")
    parser.add_argument('--dev-image', help="Docker label prefix for dev images", default="dedev/")
    parser.add_argument('--specs', help='JSON file containing additional metadata for packages', default="specs.json")
    parser.add_argument('--files', help="Path to the directory containing all the snapshot files named by hash", default="snapshot-files")
    parser.add_argument('package', metavar="PACKAGE", help='Path to .dsc of source package')

    main(parser.parse_args())