#!/usr/bin/env python3
import sys
import tarfile
import argparse
import json
import os
import os.path
import tempfile

from runner import Runtime

def snapshot_mirror(base_url, stamp):
    if not base_url.endswith("/"):
        base_url += '/'
    return base_url + "archive/debian/" + stamp + "/"


def debootstrap(runtime: Runtime, release_name, target_dir, mirror_url, arch="amd64"):
    target_dir = os.path.abspath(target_dir)
    script_arg = [os.path.abspath(os.path.join(os.path.dirname(__file__), 'debian-bootstrap'))] if release_name == 'sid' else []

    config = runtime.config_host([
        "env", "container=lxc", "debootstrap",
        "--variant=buildd", "--no-check-gpg", "--no-merged-usr",
        release_name, target_dir, mirror_url
    ] + script_arg)

    dev_files = [ "null", "zero", "full", "random", "urandom", "tty" ]
    proc_files = [ "cmdline" ]
    os.makedirs(target_dir + "/proc", exist_ok=True)
    os.makedirs(target_dir + "/dev", exist_ok=True)

    config['mounts'].extend([
        { 'destination': target_dir + "/dev/" + fname, 'type': 'none', 'source': '/dev/' + fname, 'options': ['bind'] }
        for fname in dev_files
    ])
    config['mounts'].extend([
        { 'destination': target_dir + "/proc/" + fname, 'type': 'none', 'source': '/proc/' + fname, 'options': ['bind'] }
        for fname in proc_files
    ])
    runtime.spawn(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test building the debootstrap environment for building a package spec")
    parser.add_argument("spec", metavar="SPEC", help="Path to a package spec tarball")
    parser.add_argument("--tmp-dir", metavar="DIR", help="Location of a temporary directory used for temporary artifacts", default="/tmp/debootstrap")
    parser.add_argument("--mirror", metavar="URL", help="URL of a snapshot.debian.org mirror.", default="http://snapshot.debian.org")
    args = parser.parse_args()

    runtime = Runtime()

    with tarfile.open(args.spec, 'r') as tar:
        metadata = json.load(tar.extractfile('package.json'))
        mirror = snapshot_mirror(args.mirror, metadata['dsc_info']['first_seen'])
        os.makedirs(args.tmp_dir, exist_ok=True)
        target_dir = tempfile.mkdtemp(dir=args.tmp_dir)
        print("debootstrapping in", target_dir)
        debootstrap(runtime, metadata['dist'].split("-")[0], target_dir, mirror)
        #runtime.spawn(runtime.config_host(["rm", "-rf", "--one-file-system", target_dir]))
