#!/usr/bin/env python3
import time
import json
import requests
import pickle
import itertools
import sys

from tqdm import tqdm

ips = ["193.62.202.27", "185.17.185.185"]
ip_iter = itertools.cycle(ips)

def fetch_src_hashes(s, pkg, version, ip):
    u = f"http://{ip}/mr/package/{pkg}/{version}/srcfiles"
    try:
        resp = s.get(u, headers={'Host': 'snapshot.debian.org'})
        if resp.status_code == 404:
            return
        resp.raise_for_status()
        srcs = resp.json()['result']
        yield from [src['hash'] for src in srcs]
    except json.JSONDecodeError as e:
        print("cannot parse response to " + u)
        print(e.doc)
    except requests.HTTPError as e:
        print("http error for  " + u)
        print(e)

with open("src_hashes.pickle", 'rb') as f:
    src_hashes = pickle.load(f)

with open("pkgs.json" if len(sys.argv) < 2 else sys.argv[1]) as f:
    versioned_pkgs = json.load(f)

def loop(progress):

    s = requests.Session()
    last_conn_error = { ip: 0 for ip in ips }
    ip = next(ip_iter)
    missing = set((pkg, ver) for pkg, ver in versioned_pkgs) - set((pkg, ver) for pkg, ver, _ in src_hashes)

    while missing:
        pkg, version = missing.pop()
        try:
            hashes = set(fetch_src_hashes(s, pkg, version, ip=ip))
            src_hashes.append((pkg, version, hashes))
            time.sleep(0.5)
            progress.update(1)
        except requests.ConnectionError as e:
            missing = set((pkg, ver) for pkg, ver in versioned_pkgs) - set((pkg, ver) for pkg, ver, _ in src_hashes)
            with open("src_hashes.pickle", "wb") as f:
                pickle.dump(src_hashes, f)

            last_conn_error[ip] = time.time()
            ip = next(ip_iter)
            if time.time() - last_conn_error[ip] < 10:
                print("conn error, sleeping")
                time.sleep(60)
            else:
                print('conn error, not sleeping')

if __name__ == '__main__':
    try:
        missing = set((pkg, ver) for pkg, ver in versioned_pkgs) - set((pkg, ver) for pkg, ver, _ in src_hashes)
        with tqdm(total=len(missing)) as progress:
            loop(progress)
    finally:
        with open("src_hashes.pickle", "wb") as f:
            pickle.dump(src_hashes, f)
