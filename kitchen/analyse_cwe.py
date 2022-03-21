#%% Imports
import glob
import pickle
import json
import gzip
import collections
from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras
import sklearn.metrics
from sklearn.model_selection import StratifiedKFold

from compy.models.graphs.tf2_sandwich_model import sandwich_model, pack_ragged_batch_to_dense_input

tf.compat.v1.enable_eager_execution()
tf.compat.v1.enable_v2_behavior()
#%% Load data
with open("/code/compy-learn/local/cvevulns-tokens.pickle", 'rb') as f:
    dataset = pickle.load(f)
#%% Process samples
# Filter samples which are too long
samples = [s for s in dataset['samples'] if s['x']['code_rep'].seq_len <= 1500]

# Keep only samples related to the vulnerable
samples_positive = [sample for sample in samples if sample['y']]
vuln_funcs = set((s['info']['cve'], s['info']['name']) for s in samples_positive)
samples_hasvuln = [s for s in samples if (s['info']['cve'], s['info']['name']) in vuln_funcs]

balanced_samples = np.array(samples_hasvuln)
#%% Load CVE info
path = Path("/mnt/ml4code/nvd/")
cves = []
for cve_path in path.glob("nvdcve-*.json.gz"):
    with gzip.open(cve_path, 'rb') as f:
        cves += json.load(f)['CVE_Items']
#%% Process CVE info
cve_by_id = { cve['cve']['CVE_data_meta']['ID']: cve for cve in cves }
dataset_cves = set(sample['info']['cve'] for sample in dataset['samples'])
#%% Find CWEs for dataset cves
cwes_by_cve_id = {}
for cve_id in dataset_cves:
    if cve_id not in cve_by_id:
        raise RuntimeError("cve not in cve list")
    cve = cve_by_id[cve_id]
    problemtype_datas = cve['cve']['problemtype']['problemtype_data']
    assert len(problemtype_datas) == 1
    cwes_by_cve_id[cve_id] = set(d['value'] for d in problemtype_datas[0]['description'])
#%% Functions
def batch_samples(samples):
    def to_ragged(arrs):
        flat = tf.concat(arrs, axis=0)
        return tf.RaggedTensor.from_row_lengths(flat, [len(x) for x in arrs])
    return {
        'seq_nodes': to_ragged([sample['x']['code_rep'].get_sequence_nodes() for sample in samples]),
        'non_seq_nodes': to_ragged([sample['x']['code_rep'].get_non_sequence_nodes() for sample in samples]),
        'edges': to_ragged([sample['x']['code_rep'].edges.T for sample in samples]),
        'y': tf.constant([sample['y'] for sample in samples]),
    }


def sequence_graphs_dataset(samples):
    return tf.data.Dataset.from_tensor_slices(batch_samples(samples))


def ragged_batch_to_train_input(*, seq_nodes: tf.RaggedTensor, non_seq_nodes: tf.RaggedTensor, edges: tf.RaggedTensor, y: tf.Tensor):
     return pack_ragged_batch_to_dense_input(seq_nodes=seq_nodes, non_seq_nodes=non_seq_nodes, edges=edges), y


def make_dataset(samples):
    return sequence_graphs_dataset(samples).batch(32).map(lambda x: ragged_batch_to_train_input(**x))


def make_batched_input(samples):
    batch = batch_samples(samples)
    batch.pop('y', None)
    return pack_ragged_batch_to_dense_input(**batch)
#%% Load model
MODEL = Path("local/results/train-sandwich-paired-20210831-205812-35d1b40315a6/64h-0do-02.h5")
METRICS = MODEL.parent / ('-'.join(MODEL.name.replace('.h5', '').split('-')[:3]) + "-metrics.json")
with open(METRICS, 'rb') as f:
    metrics = json.load(f)
with open(MODEL.parent / "config.json") as f:
    config = json.load(f)
model = sandwich_model(config, rnn_dense=True)
model.load_weights(MODEL)
opt = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
model.compile(opt, 'sparse_categorical_crossentropy', metrics=['accuracy'])

data_split_idx = int(MODEL.name.split('-')[2].split('.')[0])
#%% Compute the data the model was trained/evaluated against
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=204)
splits = list(kf.split(balanced_samples, [sample["y"] for sample in balanced_samples]))
train_data_idx, test_data_idx = splits[data_split_idx]

rng = np.random.default_rng(seed=0)
train_samples = balanced_samples[train_data_idx]
rng.shuffle(train_samples)
train_data = make_dataset(train_samples)

test_samples = balanced_samples[test_data_idx]
rng.shuffle(test_samples)
test_data = make_dataset(test_samples)
#%% Gather predictions and true labels for test data
predictions = tf.concat([model(inp[0], training=False) for inp in test_data], axis=0)
predicted_label = tf.cast(tf.argmax(predictions, axis=-1), dtype=tf.bool)
true_label = tf.constant([bool(s['y']) for s in test_samples])
#%% Gather the CWEs for the balanced samples
def ixs_with_cwe(samples, cwe):
    return [i for i,s in enumerate(samples) if cwe in cwes_by_cve_id[s['info']['cve']]]

test_ixs_for_cwe = {
    cwe: np.array(ixs_with_cwe(test_samples, cwe))
    for cwe in set(cwe for cwes in cwes_by_cve_id.values() for cwe in cwes) if ixs_with_cwe(test_samples, cwe)
}

for cwe, ixs in sorted(test_ixs_for_cwe.items(), key=lambda x: len(x[1])):
    acc = sklearn.metrics.accuracy_score(true_label.numpy()[ixs], predicted_label.numpy()[ixs])
    print(f"{cwe:<8} {len(ixs):>2} {acc:1.02f}")
