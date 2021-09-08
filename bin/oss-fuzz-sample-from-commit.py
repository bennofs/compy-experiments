#!/usr/bin/env python3
"""Extract a vulnerability sample from a patch git commit.

Given a commit in a git repository that fixes a vulnerability, this script collects all changed sources files.
It then tries to determine compile flags so that these source files can be successfully compiled.
"""
import argparse

from lib import config


def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(epilog=__doc__)
    parser.add_argument("--project-repo", metavar="DIR", help="Path to git repo containing the commit")
    parser.add_argument()
    parser.add_argument("--project-commit", metavar="SHA1", help="A vulnerability-fixing commit")

    args = parser.parse_args()
    if not config.is_ipython():
        main(args)