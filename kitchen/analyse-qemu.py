#%% Imports
import json
import numpy as np
import pandas as pd
from pathlib import Path
import pygit2

from typing import Iterable
from tqdm import tqdm

#%% Load data

QEMU_CSV = Path("/mnt/ml4code/devign/qemu.csv")
qemu_df = pd.read_csv(QEMU_CSV)
print(qemu_df)

#%% Open repo
QEMU_REPO = Path("/code/QEMU")
qemu_repo = pygit2.Repository(QEMU_REPO)

commits = qemu_df.sha_id.apply(lambda x: qemu_repo[x])
#%% Changed file names
def get_tree_changed_files(orig: pygit2.Tree, changed: pygit2.Tree):
    """Return tuples of old_file_path, new_file_path for all changed files between two trees."""
    for patch in changed.diff_to_tree(orig):
        yield patch.delta.old_file.path, patch.delta.new_file.path


def get_tree_file_paths(tree: pygit2.Tree, prefix="") -> Iterable[str]:
    """Return an iterable of all file paths in the given tree."""
    for obj in tree:
        if obj.type == pygit2.GIT_OBJ_TREE:
            yield from get_tree_file_paths(obj, prefix=prefix + obj.name + "/")
            continue

        yield prefix + obj.name


def get_commit_all_changed_paths(commit: pygit2.Commit) -> Iterable[str]:
    """Return an iterable of all the paths changed by the given commit, compared to the commirs first parent."""
    if len(commit.parents) == 0:
        yield from get_tree_file_paths(commit.tree)
        return

    changed = set(path for change in get_tree_changed_files(commit.tree, commit.parents[0].tree) for path in change)
    yield from changed
