#!/usr/bin/env python3
import argparse
import requests
import json
import os
import helpers
import subprocess
import tempfile
import glob

from hashlib import sha1

def main(args):
    with open(args.package, 'rb') as f:
        fhash = sha1(f.read()).hexdigest()

    with open(args.specs, 'r') as f:
        specs = [json.loads(l) for l in f]

    spec = next(s for s in specs if s['sha1'] == fhash and s['dist'])
    args.cache_dir = None if args.cache_dir is None else os.path.abspath(args.cache_dir)

    snapshot_dir = os.path.abspath('snapshot-files')
    args.package = os.path.abspath(args.package)
    args.build_dir = os.path.abspath(args.build_dir)
    pkgid = f'{spec["name"]}-{spec["version"]}'

    with tempfile.TemporaryDirectory(dir=args.build_dir) as build_root:
        args.build_dir = build_root
        os.chdir(args.build_dir)

        dev_docker = [
            'docker', 'run', '--rm',
            "--volume", args.build_dir + ":/build",
            "--workdir=/build",
            "--mount", "type=bind,source=" + snapshot_dir + ",destination=" + snapshot_dir,
            "--mount", "type=bind,source=" + os.path.dirname(args.package) + ",destination=/pkg",
        ]

        if args.cache_dir is not None:
            dev_docker += ['--volume', args.cache_dir + ":/cache"]
        else:
            dev_docker += ['--mount', 'type=tmpfs,destination=/cache']

        dev_docker += [ args.dev_image ]

        subprocess.run(dev_docker + [
            "debootstrap",
            '--cache-dir', '/cache',
            spec['dist'].split('-')[0],
            '/build/rootfs',
            'http://' + args.snapshot_host + '/archive/debian/' + spec['first_seen'] + '/'
        ], check=True)

        subprocess.run(dev_docker + [
            'dpkg-source',
            '-x', f'/pkg/{os.path.basename(args.package)}',
            '/build/src',
        ], check=True)

        subprocess.run(dev_docker + [
            'mk-build-deps',
            '/build/src/debian/control',
        ], check=True)

        build_deps_deb = glob.glob("*.deb")[0]
        os.rename(build_deps_deb, f"../deps-{pkgid}.deb")

        input("done?")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build a binary package from a debian source package and extract compile_commands.json")
    parser.add_argument('--cache-dir', help="Directory used as a cache for deb dependencies and file infos")
    parser.add_argument('--snapshot-host', help='Address of the debian snapshot service. Useful for caching.', default='snapshot.debian.org')
    parser.add_argument('--build-dir', help="Directory in which to perform the build", default="build")
    parser.add_argument('--dev-image', help="Docker image containing debian dev tools (deboostrap, devscripts)", default="local/debian-dev")
    parser.add_argument('--specs', help='JSON file containing additional metadata for packages', default="specs.json")
    parser.add_argument('package', metavar="PACKAGE", help='Path to .dsc of source package')

    main(parser.parse_args())