import logging
import os
import re
import typing

import pygit2

# Mappings from old or mirrored repositories to their current canonical location
REPO_MAPPINGS = {
    'https://github.com/horhof/quickjs': 'https://github.com/bellard/quickjs',
}
LOGGER = logging.getLogger(__name__)
LANG_FILE_EXTENSIONS = {
    'c++': ['cpp', 'hpp', 'cxx', 'hxx', 'cc', 'hh', 'c++', 'h++'],
    'c': ['c', 'h'],
}


def resolve_repo_url(url):
    """Return the current location of the repo given by the URL, resolving known moved or mirrored locations.

    For example, 'https://github.com/horhof/quickjs' is now located at 'https://github.com/bellard/quickjs'.
    """
    mapped = REPO_MAPPINGS.get(url)
    if mapped:
        return mapped

    return url


def find_git_repos(base_dir, max_depth=None, allow_nested=True) -> typing.Iterable[str]:
    """Find git repos in the specified directory.

    :param base_dir: The directory which should be searched (recursively).
    :param max_depth: Maximum search depth. A depth of zero would only return base_dir itself, if it is a git repo.
    :param allow_nested: If True, search for git repos nested within other repos as well.
    :return: An iterable of absolute paths to git repos.
    """
    root = os.path.abspath(base_dir)
    for dirpath, dirs_to_recurse, fnames in os.walk(root):
        if '.git' in dirs_to_recurse or '.git' in fnames:
            yield dirpath
            if not allow_nested:
                dirs_to_recurse.clear()

        # check maximum recursion depth and stop if reached
        if max_depth is not None and dirpath[len(root):].count(os.pathsep) >= max_depth:
            dirs_to_recurse.clear()


def _replace_gitdir(src_dir, base_dir, file_path):
    """Replace gitdir with a relative path."""
    _GIT_DIR_MARKER = 'gitdir: '
    with open(file_path) as handle:
        lines = handle.readlines()

    new_lines = []
    for line in lines:
        if line.startswith(_GIT_DIR_MARKER):
            absolute_path = line[len(_GIT_DIR_MARKER):].strip()
            if not os.path.isabs(absolute_path):
                # Already relative.
                return

            current_dir = os.path.dirname(file_path)
            # Rebase to base_dir rather than the host src dir.
            base_dir = current_dir.replace(src_dir, base_dir)
            relative_path = os.path.relpath(absolute_path, base_dir)
            LOGGER.info('Replacing absolute submodule gitdir from %s to %s',
                        absolute_path, relative_path)

            line = _GIT_DIR_MARKER + relative_path

        new_lines.append(line)

    with open(file_path, 'w') as handle:
        handle.write(''.join(new_lines))


def make_gitdirs_relative(src_dir, base_dir='/src'):
    """Replace absolute paths in .git files with relative ones.

    Git repos that contain submodules might have a .git file with an absolute path.
    This function replaces those absolute paths with relative ones, so that the repo can be moved to another location.

    :param str src_dir: Directory containing repos to rewrite
    :param str base_dir: Original location of src_dir (absolute paths assume that src_dir is located at base_dir)
    """
    for root_dir, _, files in os.walk(src_dir):
        for filename in files:
            if filename != '.git':
                continue

            file_path = os.path.join(root_dir, filename)
            _replace_gitdir(src_dir, base_dir, file_path)


def filter_is_checkout_of(paths: typing.Iterable[str], repo_path, ref_pattern: typing.Optional[re.Pattern] = None) -> typing.Iterable[str]:
    """Return paths from an iterable which are a checkout of the given original repo.

    For each git repository, it checks whether the HEAD commit of that repo is part of the original repo.
    If yes, then the repo is a checkout of the original repo and returned.

    The optional argument ``ref_pattern`` can be used to limit the scope of refs that are considered to be part of
    the original repo. If given, a commit is only considered to be contained in the original repo if
    it is a parent of a git ref matching the specified regex.
    """

    # open the base repo
    orig_repo = pygit2.Repository(repo_path)

    # collect all commits in the repo
    walker = orig_repo.walk(None)
    for ref in orig_repo.references:
        if ref_pattern is not None and not ref_pattern.match(ref):
            continue
        walker.push(orig_repo.lookup_reference(ref).resolve().target)
    commits = set(commit.id for commit in walker)

    for path in paths:
        try:
            repo = pygit2.Repository(path)
        except pygit2.GitError as e:
            if 'Repository not found' in str(e):
                LOGGER.debug("skipping %s because it is not a git repo", path)
                continue
            raise

        head = repo.head.resolve().target
        if head in commits:
            yield path


def create_detached_checkout(base_dir, dest_dir, commit):
    """Create a checkout of a git repo in ``base_dir`` at location ``dest_dir`` for revision ``commit``.

    The destination repo will use git alternates so objects are shared with the base repo.
    Care must be taken that objects which are referenced by the destination repo are not removed from the base repo.
    """
    dest_repo = pygit2.init_repository(dest_dir, False)

    objects_dir = os.path.join(dest_dir, '.git', 'objects')
    with open(os.path.join(objects_dir, 'info', 'alternates'), 'w') as f:
        f.write(os.path.relpath(os.path.join(base_dir, '.git', 'objects'), objects_dir) + "\n")

    dest_repo.checkout_tree(dest_repo[commit].tree)