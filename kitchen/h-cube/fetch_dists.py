#!/usr/bin/env python3
import snippets
import json
import os
import sys
import tarfile
import pickle
import io

import settings

from collections import defaultdict
from hashlib import sha1, md5
from tqdm import tqdm

def find_dsc(sha1_to_name, pkg, ver, hashes):
    for h in hashes:
        for name in sha1_to_name[(pkg, ver, h)]:
            if name.endswith(".dsc"):
                return h


def fix_data(p):
    if p['name'] == 'tightvnc' and p['version'] == '1:1.3.9-9deb10u1':
        p['version'] = '1:1.3.9-9+deb10u1' # + is missing in CVE list
        p['first_seen'] = '20200401T154020Z'
    if p['name'] == 'tinc' and p['version'] == '1.0.24-2+deb8u1':
        p['version'] = '1.0.24-2.1+deb8u1' # missing .1 in version
    if p['name'] == 'wireshark' and p['version'] == '2.0':
        p['version']  = None # 2.0 doesn't exist, all packages >= 2.0 were never affected
    if p['name'] == 'ffmpeg' and p['version'] == '2.8.5-1':
        p['version'] = '7:2.8.5-1'
    if p['name'] == 'glibc' and p['version'] in {'2.11.1-1', '2.11-1'}:
        p['name'] = 'eglibc' # there is no glibc package with this version
    if p['name'] == 'eglibc' and p['version'] == '2.11-1':
        p['version'] = '2.11.1-1' # there is no 2.11-1
    if p['name'] == 'qemu' and p['version'] == '1:2.10.0-1':
        p['version'] = '1:2.10.0+dfsg-1' # missing dfsg

def cves(p):
    return set(s['cve'] for s in p['samples'])


def without_code(p):
    p = dict(p)
    for sample in p['samples']:
        sample.pop('code', None)
    return p


def main(data_file):
    with open(data_file) as f:
        pkgs = json.load(f)

    files_dir = settings.SNAPSHOT_FILES_PATH

    info_dir = "file-info"
    os.makedirs(info_dir, exist_ok=True)

    dists_dir = 'dists'
    os.makedirs(dists_dir, exist_ok=True)

    sources_dir = 'sources-gz'
    os.makedirs(sources_dir, exist_ok=True)

    specs_dir = 'specs'
    os.makedirs(specs_dir, exist_ok=True)

    sha1_to_name = snippets.sha1_to_fname_map(files_dir, "src_hashes.pickle", "file-info")
    with open(f"{files_dir}/.md5sum") as f:
        sha1_to_md5 = { l.split()[1]: l.split()[0] for l in f }

    with open('src_hashes.pickle', 'rb') as f:
        src_hashes = pickle.load(f)
    pkgid_to_hashes = { f"{pkg}-{ver}": hashes for pkg, ver, hashes in src_hashes }

    seen = set()
    for p in tqdm(pkgs):
        try:
            fix_data(p)

            if p['version'] is None:
                print("no version", cves(p), p['name'])
                continue
            pkgid = f"{p['name']}-{p['version']}"
            p['sha1'] = find_dsc(sha1_to_name, p['name'], p['version'], pkgid_to_hashes[pkgid])
            if p['sha1'] is None:
                print("missing dsc", pkgid)
                continue
            if p['sha1'] in seen:
                continue
            seen.add(p['sha1'])
            p['md5'] = sha1_to_md5[p['sha1']]
            infos = snippets.get_file_info(p['sha1'], cache_dir=info_dir)
            p['dsc_info'] = snippets.pick_best_file_info(infos)
            p['dists'] = snippets.get_dists(p['dsc_info']['first_seen'], cache_dir=dists_dir)
            p['dist'] = snippets.find_dist(p['dists'], p, cache_dir=sources_dir)

            dist = p['dists']['testing']
            if os.path.exists(f"strace/{dist}-{pkgid}.strace.zst"):
                p['prebuilt_dist'] = dist

            with tarfile.open(f"{specs_dir}/{pkgid}.tar", "w") as tar:
                for h in pkgid_to_hashes[pkgid]:
                    for name in sha1_to_name[(p['name'], p['version'], h)]:
                        tar.add(f"{files_dir}/{h}", arcname=name)

                with open(f"changes/{p['name']}") as f:
                    p['changes'] = json.load(f)

                package_json = json.dumps(p).encode()
                t = tarfile.TarInfo(name="package.json")
                t.size = len(package_json)
                tar.addfile(t, io.BytesIO(package_json))
        except Exception:
            print("failed", without_code(p), file=sys.stderr)
            raise

if __name__ == '__main__':
    main(sys.argv[1])
