#!/usr/bin/env python3
import argparse
import json
import logging
import sys
from pathlib import Path

from lib import osv


def main(args):
    for src in osv.gather_fix_commits(osv.load_osv_files(Path(args.dataset))):
        #json.dump(src, sys.stdout)
        #sys.stdout.write("\n")
        pass


if __name__ == '__main__':
    logging.basicConfig(level='INFO')

    parser = argparse.ArgumentParser(description='Fetch all the source repos from a OSV dataset')
    parser.add_argument('dataset', metavar='DATASET', help='Path to the OSV dataset (directory with OSV yaml files)')

    main(parser.parse_args())