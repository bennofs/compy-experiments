#%% Imports
import glob

import pandas as pd
import tensorflow as tf
import tensorflow.keras
from sklearn.model_selection import StratifiedKFold

from compy.models.graphs.tf2_sandwich_model import sandwich_model

tf.compat.v1.enable_eager_execution()
tf.compat.v1.enable_v2_behavior()
#%% Functions
def sequence_graphs_dataset(samples):
    def to_ragged(arrs):
        flat = tf.concat(arrs, axis=0)
        return tf.RaggedTensor.from_row_lengths(flat, [len(x) for x in arrs])
    return tf.data.Dataset.from_tensor_slices({
        'seq_nodes': to_ragged([sample['x']['code_rep'].get_sequence_nodes() for sample in samples]),
        'non_seq_nodes': to_ragged([sample['x']['code_rep'].get_non_sequence_nodes() for sample in samples]),
        'edges': to_ragged([sample['x']['code_rep'].edges.T for sample in samples]),
        'y': tf.constant([sample['y'] for sample in samples]),
    })


def ragged_batch_to_train_input(*, seq_nodes: tf.RaggedTensor, non_seq_nodes: tf.RaggedTensor, edges: tf.RaggedTensor, y: tf.Tensor):
    seq_nodes_padded = seq_nodes.to_tensor()
    seq_mask = tf.reshape(tf.ones_like(seq_nodes, dtype=tf.bool).to_tensor(), (-1,))
    seq_shape = tf.shape(seq_nodes_padded)
    num_seq_nodes = seq_shape[0] * seq_shape[1]
    num_non_seq_nodes = tf.shape(non_seq_nodes.flat_values)[0]

    nodes = tf.concat([tf.reshape(seq_nodes_padded, (-1, )), non_seq_nodes.flat_values], axis=0)

    seq_len = tf.cast(seq_nodes.row_lengths(), edges.dtype),
    seq_positions = tf.reshape(
        tf.repeat(tf.range(seq_shape[1], dtype=tf.int32)[tf.newaxis, :], seq_shape[0], axis=0),
        (-1,)
    ) + 1
    node_positions = tf.pad(seq_positions * tf.cast(seq_mask, seq_positions.dtype), [[0, num_non_seq_nodes]])

    seq_offset = tf.range(seq_shape[0]) * seq_shape[1]
    seq_offset = tf.repeat(seq_offset, edges.row_lengths())

    non_seq_starts = tf.cast(non_seq_nodes.row_starts(), edges.dtype)
    non_seq_offset = non_seq_starts + num_seq_nodes - seq_len
    non_seq_offset = tf.repeat(non_seq_offset, edges.row_lengths())

    seq_len_edges = tf.repeat(seq_len, edges.row_lengths())

    def offset_edges(x):
        is_seq = tf.cast(tf.less(x, seq_len_edges), x.dtype)
        return x + seq_offset * is_seq + non_seq_offset * (1 - is_seq)

    edge_type, edge_src, edge_dst = tf.unstack(edges.flat_values, axis=-1)
    edges = tf.stack([edge_type, offset_edges(edge_src), offset_edges(edge_dst)], axis=-1)

    num_nodes = tf.shape(nodes)[0]
    tf.assert_less(edges[:, 1], num_nodes)
    tf.assert_less(edges[:, 2], num_nodes)

    num_graphs = seq_shape[0]
    seq_graph_ids = tf.reshape(tf.broadcast_to(tf.range(num_graphs)[:, tf.newaxis], seq_shape), (-1, ))
    non_seq_graph_ids = tf.repeat(tf.range(num_graphs), non_seq_nodes.row_lengths())
    graph_ids = tf.concat([seq_graph_ids, non_seq_graph_ids], axis=0)

    return {
        'edges': edges,
        'nodes': nodes,
        'graph_ids': graph_ids,
        'mask': tf.concat([tf.reshape(seq_mask, (num_seq_nodes, )), tf.ones(num_non_seq_nodes, dtype=tf.bool)], axis=0),
        'seq_shape': seq_shape,
        'node_positions': node_positions,
    }, y


def make_dataset(samples):
    return sequence_graphs_dataset(samples).batch(32).map(lambda x: ragged_batch_to_train_input(**x))

#%% Load data
df = pd.read_csv("local/train-sandwich.csv", index_col=0)

import pickle
with open("local/cvevulns-tokens.pickle", 'rb') as f:
    graphs = pickle.load(f)

from compy.representations import sequence_graph
vocab = sequence_graph.Vocabulary.load('local/cvevulns-tokens.vocab.bin.npz')
#%% Data processing
import numpy as np
# Filter samples which are too long
samples = [s for s in graphs['samples'] if s['x']['code_rep'].seq_len <= 1500]

# downsample negative samples for 50/50 split
samples_negative = [sample for sample in samples if not sample['y']]
samples_positive = [sample for sample in samples if sample['y']]
assert len(samples_negative) > len(samples_positive)

rng = np.random.default_rng(seed=0)
samples_negative_downsampled = rng.choice(samples_negative, size=len(samples_positive), replace=False)
balanced_samples = np.append(samples_negative_downsampled, samples_positive)
print("total samples: {}".format(len(balanced_samples)))
#%% Take specific split
import itertools
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=204)
splits = list(kf.split(balanced_samples, [sample["y"] for sample in balanced_samples]))
train_data_idx, test_data_idx = splits[0]
train_samples = balanced_samples[train_data_idx]
train_data = make_dataset(train_samples)
test_samples = balanced_samples[test_data_idx]
test_data = make_dataset(test_samples)
#%%
CONFIG = {
    'layers': ['rnn', 'ggnn', 'rnn', 'ggnn', 'rnn'],
    'batch_size': 32,
    'learning_rate': 0.001,
    'base': {
        "num_edge_types": None,
        'hidden_size_orig': None,
        'hidden_dim': 64,
        'dropout_rate': 0.0,
    },
    'ggnn': {
        'time_steps': [3, 1],
        'residuals': {'1': [0]},
        'add_type_bias': True,
    },
    'rnn': {
        'num_layers': 1,
    },
    'num_epochs': 250,
}
#%% Set model params
CONFIG['base']['num_edge_types'] = max(max(s['x']['code_rep'].edges[0]) for s in samples) + 1
CONFIG['base']['hidden_size_orig'] = graphs['num_types']
CONFIG['base']['hidden_dim'] = 32
CONFIG['base']['dropout_rate'] = 0.0

import json
with open("local/32h-fixed/config.json") as f:
    CONFIG = json.load(f)

from compy.models.graphs import tf2_sandwich_model
custom_objs = { x.__name__ : x for x in [getattr(tf2_sandwich_model, x) for x in dir(tf2_sandwich_model)] if isinstance(x, type) }
tf.config.run_functions_eagerly(False)
#%% Load model
model = sandwich_model(CONFIG, rnn_dense=True)
model.load_weights('local/32h-fixed/32h-0do-00-013.h5')
opt = tf.keras.optimizers.Adam(learning_rate=CONFIG['learning_rate'])
model.compile(opt, 'sparse_categorical_crossentropy', metrics=['accuracy'])
#sample_input, sample_label = next(iter(make_dataset(balanced_samples[:1])))
#MODEL(sample_input)
#print("parameter count", MODEL.count_params())
#%% Extract model layers
l_embed = model.layers[2]
l_rnn0 = model.layers[5]
l_rnn1 = model.layers[8]
l_rnn2 = model.layers[10]
l_ggnn0 = model.layers[7]
l_ggnn1 = model.layers[9]
l_global = model.layers[12]

def run_model_manual(model, inp):
    l_embed = model.layers[2]
    l_rnn0 = model.layers[5]
    l_rnn1 = model.layers[8]
    l_rnn2 = model.layers[10]
    l_ggnn0 = model.layers[7]
    l_ggnn1 = model.layers[9]
    l_global = model.layers[12]
    mask = inp['mask']
    seq_shape = inp['seq_shape']
    edges = inp['edges']
    states = l_embed(inp['nodes'], inp['node_positions'])
    print(hash(states.numpy().tobytes()))
    states = l_rnn0(states, seq_shape=seq_shape, mask=mask, training=False)
    print(hash(states.numpy().tobytes()))
    states = l_ggnn0(states, edges, training=False)
    print(hash(states.numpy().tobytes()))
    states = l_rnn1(states, seq_shape=seq_shape, mask=mask, training=False)
    print(hash(states.numpy().tobytes()))
    states = l_ggnn1(states, edges, training=False)
    print(hash(states.numpy().tobytes()))
    states = l_rnn2(states, seq_shape=seq_shape, mask=mask, training=False)
    print(hash(states.numpy().tobytes()))
    output = l_global(states, inp['graph_ids'])
    return tf.nn.softmax(output)
#%% Explore single sample
def sample_input(sample):
    ds = sequence_graphs_dataset([sample]).batch(1).map(lambda x: ragged_batch_to_train_input(**x))
    return next(iter(ds))[0]
inp = sample_input(test_samples[0])
#%% Re-run evaluation of trained model
import re, os
RE_MODEL_FILENAME = re.compile(r'(?P<hidden_dim>[0-9]+)h-(?P<dropout>[0-9]+\.?[0-9]*)do-(?P<split>[0-9]+).h5')
results = []
for model_path in glob.glob('local/train-sandwich/**/*-0.00do*.h5'):
    filename = os.path.basename(model_path)
    match = RE_MODEL_FILENAME.match(filename)
    assert match, f'must be valid filename: {filename}'
    model = tf.keras.models.load_model(model_path, custom_objs)
    eval_split = int(match['split'])
    print("evaluate", filename)
    eval_data = make_dataset(balanced_samples[splits[eval_split][1]])
    loss, acc = model.evaluate(eval_data)
    results.append({
        'hidden_dim': int(match['hidden_dim']),
        'dropout': float(match['dropout']),
        'split': eval_split,
        'val_loss': loss,
        'val_accuracy': acc,
    })
#%% Process ast
def all_ast_nodes(ast):
    todo = [ast]
    while todo:
        node = todo.pop()
        yield node
        todo.extend(dict(n, parent=node.get('id')) for n in node.get('inner', []))


def find_function_node(ast, func):
    for n in all_ast_nodes(ast):
        if n.get('kind', '') != 'FunctionDecl': continue

        if n['name'] == func:
            return n

#node = find_function_node(ast, 'dprintf_DollarString')
#%% Correlate to source code of functions
import os
import subprocess
from compy.datasets import CVEVulnsDataset

dataset = CVEVulnsDataset()
metadata = pd.read_json(os.path.join(dataset.content_dir, "metadata.json"), orient="index")
functions = pd.read_json(os.path.join(dataset.content_dir, 'functions.json'))

def get_sample_original_source(sample):
    info = sample['info']
    file_meta = metadata.loc[info['file_idx']]
    assert file_meta.cve == info['cve']

    file_path = os.path.join(dataset.content_dir, "sources_pp", file_meta.pp_name)
    with open(file_path, 'rb') as f:
        file_data = f.read()

    lang_args = []
    if file_meta.pp_name.endswith(".h") and (b'template <typename' in file_data or b'template <class' in file_data):
        lang_args = ['-xc++']

    clang_result = subprocess.run(
        [
            'clang', '-w', '-Xclang', '-ast-dump=json', '-fsyntax-only', '-target', 'x86_64-pc-linux-gnu', file_path
        ] + lang_args + ["-D" + define for define in file_meta['parsed']['defines']],
        check=True,
        stdout=subprocess.PIPE,
    )
    ast = json.loads(clang_result.stdout.decode())
    func_node = find_function_node(ast, info['name'])
    return func_node, file_data[func_node['range']['begin']['offset']:func_node['range']['end']['offset']+1].decode()
#%% Gather predictions and true labels for test data
predictions = tf.concat([model(inp[0], training=False) for inp in test_data], axis=0)
predicted_label = tf.cast(tf.argmax(predictions, axis=-1), dtype=tf.bool)
true_label = tf.constant([bool(balanced_samples[idx]['y']) for idx in test_data_idx])
#%% Pair vuln samples with non-vuln sample for this function
def find_nv_for_vuln(sample):
    matches = [s for s in samples if not s['y'] and s['info']['name'] == sample['info']['name'] and s['info']['cve'] == sample['info']['cve']]
    return matches

vuln_funcs = set((s['info']['cve'], s['info']['name']) for s in samples_positive)
samples_hasvuln = [s for s in samples if (s['info']['cve'], s['info']['name']) in vuln_funcs]

train_infos = set((s['info']['cve'], s['info']['file_idx'], s['info']['name'], s['y']) for s in train_samples)
samples_hasvuln_notrain = [s for s in samples_hasvuln if (s['info']['cve'], s['info']['file_idx'], s['info']['name'], s['y']) not in train_infos]
#%% Print test data predictions together with sample source code
for idx, prediction in zip(test_data_idx, predictions):
    ast, src = get_sample_original_source(balanced_samples[idx])
    print(src)
    print(f"prediction (vuln%): {prediction[0]*100:02.01f}%")
    print(f"vulnerable (label): {balanced_samples[idx]['y']}")
    print("--")
