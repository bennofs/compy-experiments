import argparse
import pickle
import typing
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

from compy.models.graphs.tf2_sandwich_model import sandwich_model
from compy.representations.sequence_graph import Vocabulary, SequenceGraph


CONFIG = {
    'layers': ['rnn', 'ggnn', 'rnn', 'ggnn', 'rnn'],
    'batch_size': 32,
    'learning_rate': 0.001,
    'base': {
        "num_edge_types": None,
        'hidden_size_orig': None,
        'hidden_dim': 64,
        'dropout_rate': 0.1,
    },
    'ggnn': {
        'time_steps': [3, 1],
        'residuals': {'1': [0]},
        'add_type_bias': True,
    },
    'rnn': {
        'num_layers': 1,
    },
    'num_epochs': 1000,
}


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


def main(args):
    tf.compat.v1.enable_eager_execution()
    tf.compat.v1.enable_v2_behavior()
    #tf.debugging.experimental.enable_dump_debug_info(args.logdir)

    with open(args.data, 'rb') as f:
        data = pickle.load(f)

    # Filter samples which are too long
    samples = [s for s in data['samples'] if s['x']['code_rep'].seq_len <= 1500]

    # downsample negative samples for 50/50 split
    samples_negative = [sample for sample in samples if not sample['y']]
    samples_positive = [sample for sample in samples if sample['y']]
    assert len(samples_negative) > len(samples_positive)

    CONFIG['base']['num_edge_types'] = max(max(s['x']['code_rep'].edges[0]) for s in samples) + 1
    CONFIG['base']['hidden_size_orig'] = data['num_types']

    rng = np.random.default_rng(seed=0)
    samples_negative_downsampled = rng.choice(samples_negative, size=len(samples_positive), replace=False)
    balanced_samples = np.append(samples_negative_downsampled, samples_positive)
    print("total samples: {}".format(len(balanced_samples)))

    # Train and test
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=204)
    split = kf.split(balanced_samples, [sample["y"] for sample in balanced_samples])
    for train_idx, test_idx in split:
        train, test = balanced_samples[train_idx], balanced_samples[test_idx]

        train_data = make_dataset(balanced_samples[train_idx])
        test_data = make_dataset(balanced_samples[test_idx])

        model = sandwich_model(CONFIG, rnn_dense=True)
        opt = tf.keras.optimizers.Adam(learning_rate=CONFIG['learning_rate'])
        model.compile(opt, 'sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_data, validation_data=test_data, epochs=1000, callbacks=[
            tf.keras.callbacks.TensorBoard('/tmp/tb-logs')
        ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", metavar="DATA", help="Pickled dataset to use for training", default="cvevulns-tokens.pickle")
    parser.add_argument("--vocab", metavar="VOCAB", help="Saved vocabulary for the dataset", default="cvevulns-tokens.vocab.bin.npz")
    parser.add_argument("--logdir", metavar="DIR", help="Tensorboard log dir", default="logs/sandwich")

    main(parser.parse_args())
