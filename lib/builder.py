"""Module for executing, modifying and analyzing software build processes.

A software build process is defined by a *builder* container image which when executed builds some software.
Metadata about the build is stored in a special label of the image
"""
import collections
import dataclasses
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
from typing import List, Tuple

from . import sources
from . import config
from . import strace_parser


BUILDER_IMAGE_LABEL = "io.github.bennofs.builder-info"
@dataclasses.dataclass
class ImageInfo:
    """Metadata for a builder image.

    Attributes:
        src_roots: List of paths inside the container containing source code used as part of the build
    """
    src_roots: List[str]


def buildah_from_image(image: str) -> Tuple[str, pathlib.Path]:
    """Create a buildah working container from the given image and mount it.

    Args:
        image: String identifying builder image (as accepted by ``buildah from``).

    Returns:
        A tuple of the container's id and the path to its rootfs on the host.
    """
    container_id = subprocess.run([
        'buildah', 'from', image
    ], check=True, stdout=subprocess.PIPE).stdout.decode().strip()
    rootfs_path = subprocess.run([
        'buildah', 'mount', container_id
    ], check=True, stdout=subprocess.PIPE).stdout.decode().strip()
    return container_id, pathlib.Path(rootfs_path)


def buildah_mount_git_repo(rootfs_path, git_repo):
    mount_dest = os.path.join(rootfs_path, "mnt", "git-repo")
    os.makedirs(mount_dest, exist_ok=True)
    subprocess.run([
        'mount', '-o', 'bind', git_repo, mount_dest
    ], check=True)
    subprocess.run([
        'mount', '-o', 'bind,remount,ro', mount_dest
    ], check=True)


def mounts_for_build_container(git_repo):
    """Return arguments to mount required paths into the build container."""
    return [
        '-v', git_repo + ":" + '/mnt/git-repo:ro',
    ]


def container_rootfs_path(container_id):
    """Return the path to the mounted rootfs of the container on the host.
    """
    return subprocess.run([
        'buildah', 'mount', container_id
    ], stdout=subprocess.PIPE, check=True).stdout.decode().strip()


def inject_git_source(container_id, commit, max_depth=5, ref_pattern=None, merge_func=None):
    if merge_func is None:
        merge_func = lambda old, new: None

    src_dir = buildah_image_config(container_id)['Labels'].get('build_src_root')
    if src_dir is None:
        raise RuntimeError("builder image must have a LABEL build_src_root pointing to the directory of build sources.")

    rootfs = container_rootfs_path(container_id)
    host_src_dir = os.path.normpath(rootfs + "/" + src_dir)
    host_repo_dir = os.path.normpath(rootfs + "/mnt/git-repo")

    checkouts = list(sources.filter_is_checkout_of(
        sources.find_git_repos(host_src_dir, max_depth=max_depth),
        host_repo_dir,
        ref_pattern=ref_pattern
    ))

    for checkout in checkouts:
        shutil.move(checkout, checkout + ".original")
        sources.create_detached_checkout(host_repo_dir, checkout, commit)
        merge_func(checkout + ".original", checkout)

    return checkouts


def buildah_commit_image(ctr_id):
    rootfs_path = container_rootfs_path(ctr_id)
    subprocess.run(['umount', os.path.join(rootfs_path, "mnt", "git-repo")])

    return subprocess.run(['buildah', 'commit', '--rm', ctr_id], check=True, stdout=subprocess.PIPE).stdout.decode().strip()


def podman_strace_image(image, repo_dir):
    with tempfile.TemporaryDirectory() as tmp_dir:
        strace_log = f"{tmp_dir}/strace-log"
        os.mkfifo(strace_log, 0o644)

        img_config = podman_inspect(image)
        cmd = [
            "/strace", "-xx", "--seccomp-bpf", "-f",
            "-e", "execve,chdir,fchdir,execveat,fork,vfork,clone,%process",
            "-y", "-s", "999999999",
            "-o", "/strace-out",
            '--'
        ] + podman_inspect(image)['Config']['Cmd']
        mount_args = [
            '-v', str(strace_log) + ":/strace-out",
            '-v', str(config.get_tracing_helpers_path() / "strace-static") + ":/strace:ro",
            '-v', str(repo_dir) + ":/mnt/git-repo:ro",
        ]
        priv_args = ['--security-opt' , 'seccomp=unconfined', '--cap-add', 'SYS_PTRACE']

        podman_cmd = ['podman', 'run'] + mount_args + priv_args + [image] + cmd
        buffer = b''
        parser = strace_parser.StraceParser(img_config['Config']['WorkingDir'])
        with subprocess.Popen(podman_cmd) as proc, open(strace_log, 'rb') as f:
            while True:
                read = f.read(4096)
                if not read: break
                buffer += read
                commands, buffer = parser.process_buffer(buffer, False)
                yield from commands
        commands, _ = parser.process_buffer(buffer, True)
        yield from commands


def buildah_image_config(id):
    return json.loads(
        subprocess.run(['buildah', 'inspect', id], check=True, stdout=subprocess.PIPE).stdout.decode()
    )['Docker']['config']


def podman_inspect(image_id):
    return json.loads(
        subprocess.run(['podman', 'inspect', image_id], check=True, stdout=subprocess.PIPE).stdout.decode()
    )[0]