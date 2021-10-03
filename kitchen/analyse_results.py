#%% Imports
import glob
import string
from pathlib import Path
from datetime import datetime
import json

import pandas as pd
import tensorflow as tf
import tensorflow.keras
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

from compy.models.graphs.tf2_sandwich_model import sandwich_model

tf.compat.v1.enable_eager_execution()
tf.compat.v1.enable_v2_behavior()
#%%
RESULTS_DIR = Path("/mnt/ml4code/results")

all_results = []
for result_path in RESULTS_DIR.iterdir():
    parts = result_path.name.split("-")
    date_part = parts[-3]
    time_part = parts[-2]
    if not all(x in string.digits for x in date_part):
        date_part = parts[-2]
        time_part = parts[-1]
    t_iso = f'{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}T{time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}'
    t = datetime.fromisoformat(t_iso)
    all_results.append((t, result_path))

DATASET_NAMES = {
    'cvevulns-tokens.pickle': 'first',
    'cvevulns-tokens-exclusive.pickle': 'exclusive',
    'cvevulns-tokens-paired.pickle': 'paired',
    'cvevulns-tokens-paired-undirected.pickle': 'paired-undirected',
    'cvevulns-tokens-exclusive-undirected.pickle': 'exclusive-undirected',
}


SCRIPT_MODEL = {
    'train-ggnn': 'ggnn',
    'train-sandwich': 'sandwich',
    'train-simple-model': 'simple',
    'train-rnn-only': 'rnn-only',
}

t_start = datetime.fromisoformat('2021-09-29T00:00:00')
parsed_results = []
for t, r in all_results:
    if t < t_start:
        continue

    parts = r.name.split('-')
    script = '-'.join(parts[:-3])

    param_text = (r / "parameters").read_text().replace("'", '"')
    parameters = json.loads(param_text)
    dataset = DATASET_NAMES[Path(parameters[1]).name]
    hidden = int(parameters[2].split('=')[1])

    with open(r / 'config.json') as f:
        config = json.load(f)
    timesteps = config.get('ggnn', {}).get('time_steps')

    result = {
        'time': t,
        'path': r,
        'script': script,
        'model': SCRIPT_MODEL[script],
        'hidden': hidden,
        'dataset': dataset,
        'metrics': {},
        'timesteps': timesteps,
    }

    for metrics_path in r.glob('*-metrics.json'):
        split_idx = int(metrics_path.name.split('-')[-2])
        with open(metrics_path, 'rb') as f:
            d = json.load(f)
        result['metrics'][split_idx] = d

    for metric in ['val_loss', 'loss', 'accuracy', 'val_accuracy']:
        result[metric] = pd.DataFrame(
            {epoch: data[metric] for epoch, data in result['metrics'].items() }
        )

    with open(r / 'log') as f:
        for line in f:
            if 'parameter count' in line:
                assert 'parameter_count' not in result
                result['parameter_count'] = int(line.split()[-1])

    parsed_results.append(result)

results_by_key = {}
for r in parsed_results:
    model = SCRIPT_MODEL[r['script']]
    dataset = r['dataset']
    hidden = r['hidden']
    if model in ['ggnn', 'sandwich']:
        timesteps = ' '.join(str(x) for x in r['timesteps'])
        if r['timesteps'] != [3,1,3,1]:
            hidden = f'{hidden}h-{timesteps}'
    key = f'{model}-{dataset}-{hidden}'
    if key in results_by_key:
        print('duplicate', key)
        if results_by_key[key]['time'] > r['time']: continue
    results_by_key[key] = r
#%%
summary = pd.DataFrame.from_records([
    {
        'dataset': r['dataset'],
        'model': r['model'],
        'hidden': r['hidden'],
        'parameter_count': r['parameter_count'],
        'val_acc': r['val_accuracy'].mean().iloc[-1],
        'timesteps': r['timesteps'],
    }
    for r in parsed_results
])
#%%
acc_paired = pd.DataFrame({
    'ggnn-val': results_by_key['ggnn-paired-32']['val_accuracy'].mean(axis=1),
    'sandwich-val': results_by_key['sandwich-paired-32h-3 1']['val_accuracy'].mean(axis=1),
    'ggnn-train': results_by_key['ggnn-paired-32']['accuracy'].mean(axis=1),
    'sandwich-train': results_by_key['sandwich-paired-32h-3 1']['accuracy'].mean(axis=1),
})
acc_paired.plot(figsize=(8,4.5))
plt.xlabel('epoch')
plt.ylabel('accuracy')
#plt.savefig()
plt.savefig('presentation/media/plot-acc-paired.pdf', bbox_inches='tight')
#%%
acc_exclusive = pd.DataFrame({
    'ggnn-val': results_by_key['ggnn-exclusive-32']['val_accuracy'].mean(axis=1),
    'sandwich-val': results_by_key['sandwich-exclusive-32h-3 1']['val_accuracy'].mean(axis=1),
    'ggnn-train': results_by_key['ggnn-exclusive-32']['accuracy'].mean(axis=1),
    'sandwich-train': results_by_key['sandwich-exclusive-32h-3 1']['accuracy'].mean(axis=1),
    'simple-train': results_by_key['simple-exclusive-8']['accuracy'].mean(axis=1),
    'simple-val': results_by_key['simple-exclusive-8']['val_accuracy'].mean(axis=1),
})
acc_exclusive.plot(figsize=(8,4.5)).legend(loc='upper left')
plt.xlabel('epoch')
plt.ylabel('accuracy')
#plt.show()
plt.savefig('presentation/media/plot-acc-exclusive.pdf', bbox_inches='tight')
#%%
