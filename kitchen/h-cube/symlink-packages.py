#!/usr/bin/env python3
import os
import os.path
import requests
import pickle
from tqdm import tqdm
from pprint import pprint
import re
import json
from collections import defaultdict

import snippets
import settings


def main():
    with open(f'{settings.SNAPSHOT_FILES_PATH}/.pgp') as pgp_file_names:
        debian_dsc = { name.strip(): snippets.read_bytes(f'{settings.SNAPSHOT_FILES_PATH}/{name.strip()}') for name in tqdm(list(pgp_file_names)) }

    with open(f"{settings.SNAPSHOT_FILES_PATH}/.md5sum") as f:
        md5_to_sha1 = { l.split()[0]: l.split()[1] for l in f }

    with open("./src_hashes.pickle", 'rb') as f:
        src_hashes = pickle.load(f)

    hash2name = defaultdict(set)
    for dsc_hash, dsc in tqdm(debian_dsc.items()):
        if not dsc.startswith(b'-----BEGIN PGP SIGNED MESSAGE'):
            continue
        pkg, ver, files = snippets.dsc_pkg_ver_files(dsc, md5_to_sha1)
        hash2name[(pkg, ver, dsc_hash)].add(f'{pkg}-{ver}.dsc')
        for f_hash, fname in files:
            hash2name[(pkg, ver, f_hash)].add(fname.decode())

    hashes = list(os.listdir(f"{settings.SNAPSHOT_FILES_PATH}"))
    missing_fname = list(set(hashes) - set(h for _, _, h in hash2name))

    file_infos = { h: snippets.get_file_info(h, cache_dir="file-info") for h in tqdm(missing_fname) if not h.startswith('.') }

    hash2src = { hash: (pkg, ver) for pkg, ver, hashes in src_hashes for hash in hashes }

    os.makedirs("debian-source", exist_ok=True)
    for pkg,ver,hashes in tqdm(src_hashes):
        os.makedirs(f"debian-source/{pkg}-{ver}", exist_ok=True)

    cwd = os.getcwd()
    for pkg,ver,hashes in tqdm(src_hashes):
        for h in hashes:
            for name in hash2name.get((pkg, ver, h), set()) | set(x['name'] for x in file_infos.get(h, ())):
                try:
                    os.symlink(f"{settings.SNAPSHOT_FILES_PATH}/{h}", f"debian-source/{pkg}-{ver}/{name}")
                except FileExistsError:
                    pass
if __name__ == '__main__':
    main()
