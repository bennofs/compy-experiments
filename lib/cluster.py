import os
import subprocess
import shlex
from datetime import datetime
from pathlib import Path
from textwrap import dedent

PROJECTS_DIR = Path(os.getenv("P"))
WORKSPACE_DIR = Path(os.getenv("WS"))
EXPERIMENTS_ROOT = Path(os.getenv("EXPERIMENTS_ROOT"))
WHEEL_SLUG = os.getenv("WHEEL_SLUG")

CLUSTER_ENV_HOOK = EXPERIMENTS_ROOT / 'hooks/cluster-env.sh'
CACHE_DIR = WORKSPACE_DIR / "cache"
LOGS_DIR = WORKSPACE_DIR / "logs"
RESULTS_DIR = WORKSPACE_DIR / 'results'

for path in [CACHE_DIR, LOGS_DIR, RESULTS_DIR]:
    os.makedirs(path, exist_ok=True)


def sbatch(script: str, *, logfile, args=None, partition='ml'):
    if args is None:
        args = []
    script = dedent(script).strip().encode()
    result = subprocess.run(
        ['sbatch', '--parsable', '-p', partition, '-o', logfile] + args,
        input=script, check=True, stdout=subprocess.PIPE,
    ).stdout
    return int(result.decode().strip().split("\n")[-1])


def compy_wheel_path(commit):
    return CACHE_DIR / f'ComPy-{commit}-{WHEEL_SLUG}.whl'


def cache_compy(commit):
    wheel_path = compy_wheel_path(commit)
    if os.path.exists(wheel_path):
        return # already build

    script = f'''
    #!/usr/bin/env bash
    set -euo pipefail
    source "{CLUSTER_ENV_HOOK}"
    exec python3 {EXPERIMENTS_ROOT / 'tasks/build-compy.py'} "{commit}" "{os.path.dirname(wheel_path)}"
    '''

    return sbatch(script, logfile=LOGS_DIR / f"build-compy-{commit}.log", args=['-c8', '-n1', '--mem', '32G'])


def run_experiment(commit, experiment_script, args, slurm_args, cpu, mem):
    name, _ = os.path.splitext(os.path.basename(experiment_script))
    experiment_script = os.path.realpath(experiment_script)
    cache_jobid = cache_compy(commit)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = f'{name}-{stamp}'
    results = RESULTS_DIR / slug
    os.makedirs(results)

    with open(results / 'commit', 'w') as f:
        f.write(commit)

    quoted_args = " ".join(shlex.quote(a) for a in args)

    script = f'''
    #!/usr/bin/env bash
    set -euo pipefail
    source "{CLUSTER_ENV_HOOK}"
    pip install "{compy_wheel_path(commit)}"
    cd "{results}"
    python3 {experiment_script} {quoted_args} 2>&1 | while read line; do echo "$(date --iso-8601=seconds) $line"; done
    '''

    slurm_args += [
        '--cpus-per-task', str(cpu),
        '--mem', mem,
        '--gres', 'gpu:1',
        '-n1',
        '--job-name', name
    ]
    if cache_jobid is not None:
        slurm_args += ['--dependency', f'afterok:{cache_jobid}']

    return sbatch(script, logfile=results / "log", args=slurm_args)
