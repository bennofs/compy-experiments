#!/usr/bin/env python3
import helpers
import json
import os

from hashlib import sha1
from tqdm import tqdm

def main():
    with open("./pkgs-to-build.json") as f:
        pkgs = json.load(f)

    info_dir = "file-info"
    os.makedirs(info_dir, exist_ok=True)

    dists_dir = 'dists'
    os.makedirs(dists_dir, exist_ok=True)

    for p in tqdm(pkgs):
        pkgid = f"{p['name']}-{p['version']}"
        path = f'debian-source/{pkgid}/{pkgid}.dsc'
        try:
            with open(path, 'rb') as f:
                fhash = sha1(f.read()).hexdigest()
            info = helpers.get_file_info(fhash, cache_dir=info_dir)
            helpers.get_dists(helpers.get_first_seen(info), cache_dir=dists_dir)
        except FileNotFoundError:
            print("missing", path)

if __name__ == '__main__':
    main()