import requests
import json
import time
import random
import gzip
import lzma
import sys
import re
import os
import pickle

from lxml import html
from collections import defaultdict
from tqdm import tqdm


def fetch(url):
    # try 5 times, with exponential backoff
    i = 1
    delay = 1.5
    while True:
        try:
            r = requests.get(url)
            break
        except requests.ConnectionError:
            if i > 5:
                raise
            delay += random.randrange(0, 2)
            time.sleep(delay)
            delay *= 2
        i += 1
    return r


def get_file_info(file_hash, cache_dir=None):
    if cache_dir is not None:
        try:
            with open(f"{cache_dir}/{file_hash}.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            pass


    r = fetch(f"https://snapshot.debian.org/mr/file/{file_hash}/info")
    result = r.json()["result"]
    if cache_dir is not None:
        with open(f"{cache_dir}/{file_hash}.json", 'w') as f:
            json.dump(result, f)

    return result


def pick_best_file_info(infos):
    info = next((i for i in infos if i['archive_name'] == 'debian'), None)
    if info is None:
        info = next((i for i in infos if i['archive_name'] == 'debian-security'), None)
    if info is None:
        print(infos)
    return info


def get_dists(stamp, cache_dir=None):
    if cache_dir is not None:
        try:
            with open(f"{cache_dir}/{stamp}.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            pass

    r = fetch(f"https://snapshot.debian.org/archive/debian/{stamp}/dists/")
    result = {}
    for row in html.fromstring(r.content).xpath("//tr"):
        cells = row.xpath("td")
        if len(cells) < 2:
            continue
        url = cells[1].xpath("a/@href")
        if cells[0].text == 'd' and not url[0].startswith('.'):
            result[url[0].rstrip('/')] = url[0].rstrip('/')
        if cells[0].text == 'l' and len(url) > 1:
            if '..' in url[1].rstrip("/"):
                result[url[0].rstrip('/')] = url[0].rstrip('/')
            else:
                result[url[0].rstrip('/')] = url[1].rstrip('/')

    if cache_dir is not None:
        with open(f"{cache_dir}/{stamp}.json", 'w') as f:
            json.dump(result, f)

    return result


def fetch_sources(stamp, dist, cache_dir=None, pool='main', archive='debian'):
    if cache_dir is not None:
        try:
            with gzip.open(f"{cache_dir}/{dist.replace('/', '_')}-{pool.replace('/', '_')}-{stamp}.gz", 'r') as f:
                return f.read()
        except FileNotFoundError:
            pass

    print("fetching", stamp, dist, pool, archive, file=sys.stderr)
    r = fetch(f"https://snapshot.debian.org/archive/{archive}/{stamp}/dists/{dist}/{pool}/source/Sources.gz")
    if r.status_code == 404:
        r = fetch(f"https://snapshot.debian.org/archive/{archive}/{stamp}/dists/{dist}/{pool}/source/Sources.xz")
        if r.status_code == 404:
            return b""
        data = lzma.decompress(r.content)
        compressed_data = gzip.compress(data)
    else:
        compressed_data = r.content
        data = gzip.decompress(r.content)
    if cache_dir is not None:
        with open(f"{cache_dir}/{dist.replace('/', '_')}-{pool.replace('/', '_')}-{stamp}.gz", 'wb') as f:
            f.write(compressed_data)
    return data


# higher priority means checked first
def dist_priority(name, release):
    is_wanted_release = release and release in name
    name_score = 2
    if name.startswith("sid") or name.startswith("unstable") or name.startswith("testing"):
        name_score += 1
    if "-" in name:
        name_score -= 1
    if '..' in name or "experimental" in name:
        name_score -= 10
    return (is_wanted_release, name_score)



def find_dist(dists, info, cache_dir=None):
    queue = list(sorted(set(dists.values()), key=lambda name: dist_priority(name, info.get("release"))))
    pool = "/".join(info['dsc_info']['path'].split("/")[2:-2])
    while queue:
        v = queue.pop()

        if info['md5'].encode() in fetch_sources(info['dsc_info']['first_seen'], v, pool=pool, archive=info['dsc_info']['archive_name'], cache_dir=cache_dir):
            return v


RE_PKG = re.compile(b'^Source: (.*)$', re.MULTILINE)
RE_VER = re.compile(b'^Version: (.*)$', re.MULTILINE)
RE_FILE = re.compile(b'^ ([0-9a-f]{40}) [0-9]+ (.*)$', re.MULTILINE)
RE_FILE_MD5 = re.compile(b'^ ([0-9a-f]{32}) [0-9]+ (.*)$', re.MULTILINE)
def dsc_pkg_ver_files(dsc, md5_to_sha1):
    pkg = RE_PKG.search(dsc).group(1).decode()
    ver = RE_VER.search(dsc).group(1).decode()
    from_md5 = [(md5_to_sha1[m.group(1).decode()], m.group(2)) for m in RE_FILE_MD5.finditer(dsc)]
    from_sha1 = [ (m.group(1).decode(), m.group(2)) for m in RE_FILE.finditer(dsc) ]
    return pkg, ver, list(set(from_md5) | set(from_sha1))


def read_bytes(name):
    with open(name, 'rb') as f:
        return f.read()


def sha1_to_fname_map(files_dir, src_hashes, info_cache=None):
    with open(f'{files_dir}/.pgp') as pgp_file_names:
        debian_dsc = { name.strip(): read_bytes(f'{files_dir}/{name.strip()}') for name in tqdm(list(pgp_file_names)) }

    with open(f"{files_dir}/.md5sum") as f:
        md5_to_sha1 = { l.split()[0]: l.split()[1] for l in f }

    with open(src_hashes, 'rb') as f:
        src_hashes = pickle.load(f)

    hash2name = defaultdict(set)
    for dsc_hash, dsc in debian_dsc.items():
        if not dsc.startswith(b'-----BEGIN PGP SIGNED MESSAGE'):
            continue
        pkg, ver, files = dsc_pkg_ver_files(dsc, md5_to_sha1)
        hash2name[(pkg, ver, dsc_hash)].add(f'{pkg}-{ver}.dsc')
        for f_hash, fname in files:
            hash2name[(pkg, ver, f_hash)].add(fname.decode())

    hashes = list(os.listdir(files_dir))
    missing = set(hashes) - set(h for _, _, h in hash2name)
    for pkg, ver, hashes in tqdm(src_hashes):
        for f_hash in hashes:
            if (pkg, ver, f_hash) in hash2name: continue
            hash2name[(pkg, ver, f_hash)].add(get_file_info(f_hash, cache_dir=info_cache)[0]['name'])

    return hash2name
