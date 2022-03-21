#!/usr/bin/env python3
import pickle
import glob
import os.path

import settings

def main(src_hashes):
    present = set(os.path.basename(x) for x in glob.glob(f"{settings.SNAPSHOT_FILES_PATH}/*"))

    all_hashes = set(x for a,b,h in src_hashes for x in h)
    for h in all_hashes:
        if h not in present:
            print(f'http://snapshot.debian.org/file/{h}')

if __name__ == '__main__':
    with open("src_hashes.pickle", 'rb') as f:
        src_hashes = pickle.load(f)
    main(src_hashes)
