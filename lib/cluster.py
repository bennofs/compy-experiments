import os
import subprocess
from pathlib import Path
from textwrap import dedent

PROJECTS_DIR = Path(os.getenv("P"))
WORKSPACE_DIR = Path(os.getenv("WS"))
EXPERIMENTS_ROOT = Path(os.getenv("EXPERIMENTS_ROOT"))
WHEEL_SLUG = os.getenv("WHEEL_SLUG")

CLUSTER_ENV_HOOK = EXPERIMENTS_ROOT / 'hooks/cluster-env.sh'
CACHE_DIR = WORKSPACE_DIR / "cache"
LOGS_DIR = WORKSPACE_DIR / "logs"

for path in [CACHE_DIR, LOGS_DIR]:
    os.makedirs(path, exist_ok=True)


def sbatch(script: str, *, logfile, args=None, partition='ml'):
    if args is None:
        args = []
    script = dedent(script).strip().encode()
    result = subprocess.run(
        ['sbatch', '--parseable', '-p', partition, '-o', logfile] + args,
        input=script, check=True, stdout=subprocess.PIPE,
    ).stdout
    return int(result.decode().strip().split("\n")[-1])


def cache_compy(commit):
    wheel_path = CACHE_DIR / f'ComPy-{commit}-{WHEEL_SLUG}.whl'
    if os.path.exists(wheel_path):
        return # already build

    script = f'''
    #!/usr/bin/env bash
    set -euo pipefail
    source "{CLUSTER_ENV_HOOK}"
    exec python3 {EXPERIMENTS_ROOT / 'tasks/build-compy.py'} "{commit}" "{os.path.dirname(wheel_path)}"
    '''

    return sbatch(script, logfile=LOGS_DIR / f"build-compy-{commit}.log")