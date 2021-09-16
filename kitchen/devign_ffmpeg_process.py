#!/usr/bin/env python3
import argparse
import contextlib
import json
import os
import re
import sys
import typing
from pathlib import Path
from typing import Iterable
from tempfile import TemporaryDirectory

import joblib
import pandas as pd
import pygit2
from tqdm import tqdm

from lib.debootstrap import debootstrap
from lib.runner import Runtime
from lib import config


MEMORY = joblib.Memory(str(config.get_cache_dir('joblib')), verbose=0)
MAX_CHANGED_FILES = 10
MAX_CHANGED_FUNCTIONS = 10


def get_tree_changed_files(orig: typing.Optional[pygit2.Tree], changed: pygit2.Tree):
    """Return tuples of old_file_path, new_file_path for all changed files between two trees."""
    diff = changed.diff_to_tree() if orig is None else changed.diff_to_tree(orig)
    for patch in diff:
        yield patch.delta.old_file.path, patch.delta.new_file.path


def get_tree_file_paths(tree: pygit2.Tree, prefix="") -> Iterable[str]:
    """Return an iterable of all file paths in the given tree."""
    for obj in tree:
        if obj.type == pygit2.GIT_OBJ_TREE:
            yield from get_tree_file_paths(obj, prefix=prefix + obj.name + "/")
            continue

        yield prefix + obj.name


def get_commit_all_changed_paths(commit: pygit2.Commit) -> Iterable[str]:
    """Return an iterable of all the paths changed by the given commit, compared to the commits first parent."""
    if len(commit.parents) == 0:
        return set(get_tree_file_paths(commit.tree))

    return set(path for change in get_tree_changed_files(commit.tree, commit.parents[0].tree) for path in change)


@MEMORY.cache(ignore=['repo'])
def gather_commit_info(repo, devign_commits):
    """For each commit, gather the time, number of parents and the set of changed files"""
    commit_time = []
    commit_paths_before = []
    commit_paths_after = []
    commit_changed_paths = []
    commit_parents = []
    for sha in tqdm(devign_commits.sha_id, desc="gather commit info"):
        commit_obj: pygit2.Commit = repo[sha]
        commit_time += [commit_obj.commit_time]
        commit_parents += [[str(x) for x in commit_obj.parent_ids]]

        parent_tree = None if not commit_obj.parents else commit_obj.parents[0].tree
        changes = list(get_tree_changed_files(parent_tree, commit_obj.tree))
        paths_before, paths_after = zip(*changes) if changes else ([], [])

        commit_changed_paths += [set(paths_before) | set(paths_after)]
        commit_paths_before += [paths_before]
        commit_paths_after += [paths_after]

    df = pd.DataFrame({
        'commit_time': commit_time,
        'commit_changed_paths': commit_changed_paths,
        'commit_parents': commit_parents,
        'commit_paths_before': commit_paths_before,
        'commit_paths_after': commit_paths_after,
        'index': devign_commits.index,
    }).set_index('index')
    return df


# regex patterns of files to ignore
IGNORED_FILE_PATTERNS = {
    r'.*\.asm$', r'.*\.S$', # we don't support assembly
    r'.*\.(ref|sw)$', r'.*\.ref.mmx$', r'tests/ref/.*', # data files for tests
    r'(^|.*/)configure$', # configure scripts
    r'.*\.(sh|awk|pl|mak|v|m)$', r'(^|.*/)(M|m)akefile$', # build system files
    r'.*\.texi$', r'.*\.txt$', r'(^|.*/)(APIchanges|MAINTAINERS|Doxyfile|RELEASE|TODO|Changelog|README\.beos)$', # documentation files
    r'(^|.*/)\.(gitignore|gitattributes)$', r'(^|.*/)\.travis\.yml$', # meta files
    r'.*\.conf$', r'.*\.init$', r'.*\.xsd$', # config and data files
    r'^tools/(patcheck|bisect)$',
}
RE_IGNORE_FILE = re.compile('|'.join(f'({x})' for x in IGNORED_FILE_PATTERNS))


def main(args):
    repo = pygit2.Repository(args.repo)
    devign_commits = pd.read_csv(os.path.join(args.devign, "ffmpeg.csv"))
    devign_num_commits = len(devign_commits)

    # compute commit time and changed paths for each commit in the devign dataset
    commit_info = gather_commit_info(repo, devign_commits)
    relevant_changed_paths = commit_info.commit_changed_paths.map(
        lambda ps: set(x for x in ps if not RE_IGNORE_FILE.match(x))
    )
    df = devign_commits.join(commit_info).assign(relevant_changed_paths=relevant_changed_paths)

    # commits with more than one parent aren't supported, make sure that the dataset has no such commits
    df_multiple_parents = df[df.commit_parents.map(len) > 1]
    if len(df_multiple_parents):
        print("error: found commits with multiple parents", file=sys.stderr)
        print(df_multiple_parents, file=sys.stderr)
        sys.exit(1)
    df = df.assign(commit_base=df.commit_parents.map(lambda x: x[0] if len(x) > 0 else None))

    # filter out all commits that have not changed any paths of interest
    has_relevant_file = df.relevant_changed_paths.map(lambda x: len(x) > 0)
    df = df[has_relevant_file]
    num_irrelevant = devign_num_commits - len(df)
    irrelevant_ratio = num_irrelevant/devign_num_commits
    print(f"removed {num_irrelevant} of {devign_num_commits} commits ({irrelevant_ratio*100:.1f}%) because they change no relevant files")

    # filter out commits that change too many files
    has_allowed_num_changed_files = df.relevant_changed_paths.map(len) <= MAX_CHANGED_FILES
    df = df[has_allowed_num_changed_files]
    num_too_long = has_allowed_num_changed_files.value_counts()[False]
    print(f"removed {num_too_long} of {devign_num_commits} commits ({num_too_long / devign_num_commits * 100:.1f}%) because they change too many files")

    # print dataset statistics
    vuln_stats = df.vulnerability.value_counts()
    ratio_kept = len(df) / devign_num_commits
    print(f"dataset has {vuln_stats[0]} non-vulnerable and {vuln_stats[1]} vulnerable commits, {ratio_kept*100:.1f}% of original devign commits")

    # generate build specs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for record in tqdm(df.to_dict('records'), desc="Saving specs"):
        # convert from set to list for json compatibility
        record['commit_changed_paths'] = list(record['commit_changed_paths'])
        record['relevant_changed_paths'] = list(record['relevant_changed_paths'])

        with open(out_dir / (record['sha_id'] + ".json"), 'w') as f:
            json.dump(record, f)

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process samples from the Devign ffmpeg vulnerability dataset")
    parser.add_argument("--devign", required=True, help="Path to devign dataset (directory containing ffmpeg.csv)")
    parser.add_argument("--repo", required=True, help="Path to ffmpeg git repo")
    parser.add_argument("out_dir", help="Output directory for the generated build specs")

    ARGS = parser.parse_args()
    del parser

    R = main(ARGS)
