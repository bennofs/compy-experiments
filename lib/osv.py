import contextlib
import dataclasses
import json
import logging
import os.path
import pathlib
import re
import subprocess
import tempfile
import typing
from hashlib import sha256

import yaml
from pygit2 import Repository

from . import sources

LOGGER = logging.getLogger(__name__)


def load_osv_files(path: pathlib.Path) -> typing.Iterator[dict]:
    """Deserialize all files matching ``*.yaml`` (recursively) in the given directory as OSV vulnerabilites.

    :returns: An iterator of OSV vulnerabilities, each represented as a nested python dictionary.
    """
    path = pathlib.Path(path) # convert to path if string

    for file_name in path.glob("**/*.yaml"):
        with open(file_name) as f:
            yield yaml.safe_load(f)


def gather_source_repos(vulns) -> typing.Iterator[dict]:
    """Gather the source repositories for the given set of OSV vulnerabilities.

    :returns: An iterator of source repositories.
    """
    seen = set()
    for vuln in vulns:
        for src in vuln['affects']['ranges']:
            repo = sources.resolve_repo_url(src['repo'])
            if repo in seen: continue
            seen.add(repo)

            yield {
                'ref': 'osv/' + vuln['id'],
                'ecosystem': vuln['package']['ecosystem'],
                'name': vuln['package']['name'],
                'src': repo,
                'type': src['type'].lower(),
            }


def gather_fix_commits(vulns) -> typing.Iterator[dict]:
    """Gather pairs of (repo, commit) where commit fixes some vulnerability.

    :returns: An iterator of dicts describing vulnerability-fixing commits.
    """

    stat_total_ranges = 0
    stat_total_vulns = 0
    stat_success_ranges = 0
    stat_success_vulns = 0
    stat_no_fix = 0
    stat_fix_is_range = 0
    stat_fix_is_introduced = 0

    for vuln in vulns:
        stat_total_vulns += 1
        have_success = False
        for range in vuln['affects']['ranges']:
            repo = sources.resolve_repo_url(range['repo'])
            stat_total_ranges += 1

            if 'fixed' not in range:
                stat_no_fix += 1
                continue

            if ':' in range['fixed']:
                stat_fix_is_range += 1
                continue

            if range['fixed'] in range.get('introduced', ''):
                stat_fix_is_introduced += 1
                continue

            stat_success_ranges += 1
            have_success = True
            yield {
                'ref': 'osv/' + vuln['id'],
                'ecosystem': vuln['package']['ecosystem'],
                'name': vuln['package']['name'],
                'src': repo,
                'sha1': range['fixed']
            }
        if have_success:
            stat_success_vulns += 1

    LOGGER.info(
        f"loaded {stat_success_ranges} fix commits from {stat_success_vulns} (of {stat_total_vulns}) vulns, "
        f"processed {stat_total_ranges} ranges, skipped: "
        f"{stat_no_fix} 'missing fix commit' "
        f"{stat_fix_is_range} 'fix is commit range' "
        f"{stat_fix_is_introduced} 'fixed and introduced commit are the same'"
    )


@dataclasses.dataclass
class OSSFuzzBuildSpec:
    oss_fuzz_commit: str
    project_name: str
    project_commit: str
    base_builder_digest: str


@dataclasses.dataclass
class OSSFuzzBuildEnv:
    """Local paths required by oss fuzz tasks"""
    token: str # unique token to prevent name conflicts in docker image tags, container names, etc.
    oss_fuzz_repo_dir: str
    project_repo_dir: str
    results_dir: str
    temp_dir: str


RE_GIT_CLONE = re.compile(r'(git.*clone)')
RE_DOCKER_FROM_LINE = re.compile(r'(FROM .*)')
RE_DOCKER_WORKDIR = re.compile(r'\s*WORKDIR\s*([^\s]+)')


def workdir_from_lines(lines, default='/src'):
    """Gets the WORKDIR from the given lines."""
    for line in reversed(lines):  # reversed to get last WORKDIR.
        match = re.match(RE_DOCKER_WORKDIR, line)
        if match:
            workdir = match.group(1)
            workdir = workdir.replace('$SRC', '/src')

            if not os.path.isabs(workdir):
                workdir = os.path.join('/src', workdir)

            return os.path.normpath(workdir)

    return default


def get_required_post_checkout_steps(dockerfile_path):
    """Get required post checkout steps (best effort)."""

    checkout_pattern = re.compile(r'\s*RUN\s*(git|svn|hg)')

    # If the build.sh is copied from upstream, we need to copy it again after
    # changing the revision to ensure correct building.
    post_run_pattern = re.compile(r'\s*RUN\s*(.*build\.sh.*(\$SRC|/src).*)')

    with open(dockerfile_path) as handle:
        lines = handle.readlines()

    subsequent_run_cmds = []
    for i, line in\
            enumerate(lines):
        if checkout_pattern.match(line):
            subsequent_run_cmds = []
            continue

        match = post_run_pattern.match(line)
        if match:
            workdir = workdir_from_lines(lines[:i])
            command = match.group(1)
            subsequent_run_cmds.append((workdir, command))

    return subsequent_run_cmds


def oss_fuzz_build_builder_image(oss_fuzz_repo: pathlib.Path, oss_fuzz_commit_sha1: str, base_builder_digest, project: str):
    with contextlib.ExitStack() as exit_stack:
        # create a checkout of the oss-fuzz repo at specified commit
        oss_fuzz_repo = Repository(str(oss_fuzz_repo))

        LOGGER.info("check out oss-fuzz:%s", oss_fuzz_commit_sha1)
        checkout_dir = exit_stack.enter_context(tempfile.TemporaryDirectory(prefix="oss-fuzz-checkout."))
        oss_fuzz_root = pathlib.Path(checkout_dir)
        oss_fuzz_repo.checkout_tree(oss_fuzz_repo[oss_fuzz_commit_sha1].tree, directory=oss_fuzz_root, paths=[f'projects/{project}'])

        # load the dockerfile and config for the project
        LOGGER.info(f"building %s image with base builder digest %s", project, base_builder_digest)
        project_path = f'{oss_fuzz_root}/projects/{project}'
        with open(os.path.join(project_path, 'project.yaml')) as f:
            project_config = yaml.safe_load(f)
        dockerfile_path = f'{project_path}/Dockerfile'
        with open(dockerfile_path) as f:
            dockerfile_contents = f.read()

        header = ''.join([
            # some older images use FROM oss-fuzz/... or similar inaccessible image names,
            # so we hardcode the base-builder image here
            f'FROM gcr.io/oss-fuzz-base/base-builder@{base_builder_digest}\n',
            'ENV FUZZING_ENGINE=libfuzzer\n',
            'ENV ARCHITECTURE=x86_64\n',
            'ENV SANITIZER=address\n',
            'ENV FUZZING_LANGUAGE=' + project_config.get('language', '') + '\n',
        ])
        dockerfile_contents = RE_DOCKER_FROM_LINE.sub(header.strip(), dockerfile_contents)

        # build the image
        cache_key = f'{oss_fuzz_commit_sha1}-{base_builder_digest}-{project}'
        image_tag = f'oss-fuzz-image_{project}_{sha256(cache_key.encode()).hexdigest()[:16]}'
        subprocess.run([
            'buildah', 'bud',
            '-t', image_tag,
            '-f', '-',
            project_path,
        ], input=dockerfile_contents.encode(), check=True)

        # if there are post-checkout commands, make a wrapper for the image
        # generate wrapper script executing post checkout commands in entrypoint
        post_checkout = get_required_post_checkout_steps(dockerfile_path)
        if post_checkout:
            wrapper_file_path = f'{project_path}/post-checkout-wrapper-script.sh'
            wrapper_file = exit_stack.enter_context(open(wrapper_file_path, 'w'))
            wrapper_file.write('_ORIG_WORKDIR="$PWD"\n')
            for workdir, cmd in post_checkout:
                wrapper_file.write(f"cd '{workdir}'; sh -c '{cmd}'\n".encode())
            wrapper_file.write('cd "$_ORIG_WORKDIR"\n')
            wrapper_file.write('exec "$@"\n')
            wrapper_file.close()

            ctr = subprocess.run(['buildah', 'from', image_tag], stdout=subprocess.PIPE, check=True).stdout.decode().strip()
            subprocess.run(
                ['buildah', 'copy', '--chmod', '0755', ctr, wrapper_file_path, '/post-checkout-wrapper-script.sh'],
                check=True,
            )

            image_config = json.loads(
                subprocess.run(['buildah', 'inspect', image_tag], check=True, stdout=subprocess.PIPE).stdout.decode()
            )['Docker']['config']
            newcmd = ['/usr/bin/env', 'sh', '/post-checkout-wrapper-script.sh'] + image_config['Cmd']
            subprocess.run(['buildah', 'config', '--cmd', json.dumps(newcmd), ctr], check=True)
            subprocess.run(['buildah', 'commit', '--rm', ctr, image_tag])

        return image_tag
