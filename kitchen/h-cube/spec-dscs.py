#!/usr/bin/env python3
import argparse
import json
import os.path
import sys

def main(args):
    with open(args.specs) as f:
        seen = set()
        for line in f:
            spec = json.loads(line)
            pkgid = f"{spec['name']}-{spec['version']}"
            if spec['sha1'] in seen:
                continue
            seen.add(spec['sha1'])
            path = os.path.abspath(f"{args.sources}/{pkgid}/{pkgid}.dsc")
            if not os.path.exists(path):
                print("missing", path, file=sys.stderr)
                sys.exit(1)
            else:
                print(path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Print path to dsc for all packages in given specs.json")
    parser.add_argument("sources", metavar="SOURCES_DIR", help="Path to debian source package directory")
    parser.add_argument("specs", metavar="SPECS", help="Path of specs.json")

    main(parser.parse_args())
