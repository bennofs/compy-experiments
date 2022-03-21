#!/usr/bin/env python3
import re
import sys
import os
import argparse
import json
import pwd
import grp
import tempfile
import subprocess
import secrets

from contextlib import contextmanager, ExitStack
from typing import List, Tuple


RE_NUMBER = re.compile("[0-9]+")
def parse_id_map(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            owner, start, end = line.split(':')
            try:
                owner = int(owner)
            except ValueError:
                owner = pwd.getpwnam(owner).pw_uid
            yield owner, int(start), int(end)


def make_id_mappings(base, ranges, total=65536):
    result = []
    remaining = list(ranges)
    while total > 0:
        start, amount = remaining.pop()
        amount = min(amount, total)
        result.append({
            "containerID": base,
            "hostID": start,
            "size": amount,
        })
        total -= amount
        base += amount
    return result


def prepare_bundle(bundle_dir, config):
    if config['root'].get('path') is None:
            config['root']['path'] = os.path.join(bundle_dir, 'rootfs')
            os.mkdir(config['root']['path'])

    with open(os.path.join(bundle_dir, "config.json"), 'w') as f:
        json.dump(config, f)


CONTAINER_CAPABILITIES = [
    'CAP_AUDIT_WRITE',
    'CAP_KILL',
    'CAP_NET_BIND_SERVICE',
    'CAP_CHOWN',
    'CAP_DAC_OVERRIDE',
    'CAP_FOWNER',
    'CAP_FSETID',
    'CAP_SETFCAP',
    'CAP_SETGID',
    'CAP_SETPCAP',
    'CAP_SETUID',
    'CAP_SYS_CHROOT',
    'CAP_SYS_ADMIN',
]

CONTAINER_MOUNTS = [
    { 'destination': '/proc', 'type': 'none', 'source': '/proc', 'options': ['bind'] },
    { 'destination': '/dev', 'type': 'tmpfs', 'source': 'tmpfs', 'options': ["nosuid", "strictatime", "mode=755", "size=65536k"] },
    { 'destination': '/dev/pts', 'type': 'devpts', 'source': 'devpts', 'options': ["nosuid", "noexec", "newinstance", "ptmxmode=0666", "mode=0620", "gid=5"] },
    { "destination": "/dev/shm", "type": "tmpfs", "source": "shm", "options": ["nosuid", "noexec", "nodev", "mode=1777", "size=65536k"] },
    #{ "destination": "/dev/mqueue", "type": "mqueue", "source": "mqueue", "options": ["nosuid", "noexec", "nodev"] },
    { "destination": "/sys", "type": "none", "source": "/sys", "options":  ["rbind", "nosuid", "noexec", "nodev", "ro" ] },
]

BASE_CONTAINER_CONFIG =  {
    'ociVersion': "1.0.2-dev",
    'process': {
        'terminal': False,
        'user': {
            'uid': 0,
            'gid': 0,
        },
        'args': None,
        'env': ["PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"],
        'cwd': '/',
        'capabilities': {
            'bounding': CONTAINER_CAPABILITIES,
            'effective': CONTAINER_CAPABILITIES,
            'inheritable': CONTAINER_CAPABILITIES,
            'permitted': CONTAINER_CAPABILITIES,
            'ambient': CONTAINER_CAPABILITIES,
        },
        'noNewPrivileges': True,
    },
    'root': {
        'readonly': False,
    },
    'hostname': 'container',
    "mounts": CONTAINER_MOUNTS,
    "linux": {
        "resources": {
            "devices": [{ "allow": False, "access": "rwm" }]
        },
        "namespaces": [{"type": "uts"}, {"type": "mount"}],
        "maskedPaths": [
            "/proc/acpi",
            "/proc/asound",
            "/proc/kcore",
            "/proc/keys",
            "/proc/latency_stats",
            "/proc/timer_list",
            "/proc/timer_stats",
            "/proc/sched_debug",
            "/sys/firmware",
            "/proc/scsi"
        ],
        #"readonlyPaths": [
        #    "/proc/bus",
        #    "/proc/fs",
        #    "/proc/irq",
        #    "/proc/sys",
        #    "/proc/sysrq-trigger"
        #]
    }
}

class Runtime:
    uid: int
    gid: int
    subuid_ranges: List[Tuple[int, int]]
    subgid_ranges: List[Tuple[int, int]]

    def __init__(self):
        self.uid = os.getuid()
        self.subuid_ranges = [
            (start, end)
            for owner, start, end in parse_id_map("/etc/subuid")
            if owner == self.uid
        ]

        self.gid = os.getgid()
        self.subgid_ranges = [
            (start, end)
            for owner, start, end in parse_id_map("/etc/subgid")
            if owner == self.uid
        ]


    def base_config(self):
        config = json.loads(json.dumps(BASE_CONTAINER_CONFIG))

        if self.uid != 0:
            uid_mappings = [{ 'containerID': 0, 'hostID': self.uid, 'size': 1}]
            uid_mappings.extend(make_id_mappings(1, self.subuid_ranges))

            gid_mappings = [{ 'containerID': 0, 'hostID': self.gid, 'size': 1}]
            gid_mappings.extend(make_id_mappings(1, self.subgid_ranges))


            config['linux']['namespaces'].append({'type': 'user'})
            config['linux']['uidMappings'] = uid_mappings
            config['linux']['gidMappings'] = gid_mappings

        return config

    def spawn(self, config, wrapper=None, **kwargs):
        with tempfile.TemporaryDirectory(prefix="container-") as path:
            prepare_bundle(path, config)
            name = os.path.basename(path)
            try:
                return subprocess.run(
                    (wrapper or []) + ["runc", "run", name],
                    cwd=path, check=kwargs.pop('check', True), **kwargs
                )
            finally:
                subprocess.run(['runc', 'delete', '-f', name])

    @contextmanager
    def spawn_background(self, config, wrapper=None, **kwargs):
        check = kwargs.pop('check', True)

        def finish_proc(proc):
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            proc.wait()
            if check and proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, cmd)

        with ExitStack() as stack:
            path = stack.enter_context(tempfile.TemporaryDirectory(prefix="container-"))
            prepare_bundle(path, config)
            name = os.path.basename(path)

            stack.callback(lambda: subprocess.run(["runc", "delete", "-f", name]))
            cmd = (wrapper or []) + ["runc", "run", name]
            proc = subprocess.Popen(cmd, cwd=path, **kwargs)
            stack.callback(finish_proc, proc)

            try:
                yield proc
            except:
                proc.kill()
                raise


    def config_host(self, cmd, workdir=None):
        config = self.base_config()
        if workdir is None:
            workdir = os.getcwd()
        config['mounts'].insert(0,
            { "destination": "/", "source": "/", "type": "none", "options": ["rbind"] }
        )
        config['process']['cwd'] = workdir
        config['process']['env'] = [
            f"{key}={value}" for key, value in os.environ.items()
        ]
        config['process']['args'] = cmd

        return config


    def config_container(self, rootfs, cmd, binds=None, workdir='/'):
        config = self.base_config()
        config['process']['args'] = cmd
        config['process']['cwd'] = workdir
        config['root']['path'] = os.path.abspath(rootfs)
        if binds is not None:
            config['mounts'].extend([
                { 'destination': key, 'source': value, 'type': 'none', 'options': ['bind'] }
                for key, value in binds.items()
            ])

        return config


if __name__ == '__main__':
    runtime = Runtime()
    if sys.argv[1] == '/':
        config = runtime.config_host(sys.argv[2:])
    else:
        config = runtime.config_container(sys.argv[1], sys.argv[2:])
    runtime.spawn(config)
