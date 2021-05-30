#!/usr/bin/env python3
import argparse
import pickle

import numpy as np
import tensorflow as tf

import compy.models.graphs.tf2_sandwich_model

tf.compat.v1.enable_eager_execution()

gnn_config = {
    'layers': ['ggnn', 'rnn'],
    'batch_size': 32,
    'base': {
        "num_edge_types": None,
        'hidden_size_orig': None,
        'hidden_dim': 32,
        'dropout_rate': 0.1,
    },
    'ggnn': {
        'time_steps': [8],
        'residuals': { '1': [0] },
        'add_type_bias': True,
    },
    'rnn': {
        'num_layers': 1,
    },
    'num_epochs': 5,
}


def sequence_graphs_dataset(samples):
    def to_ragged(arrs):
        flat = tf.concat(arrs, axis=0)
        return tf.RaggedTensor.from_row_lengths(flat, [len(x) for x in arrs])

    return tf.data.Dataset.from_tensor_slices({
        'nodes': to_ragged([sample['x']['code_rep'].nodes for sample in samples]),
        'edges': to_ragged([sample['x']['code_rep'].edges.T for sample in samples]),
        'seq_len': tf.constant([sample['x']['code_rep'].seq_len for sample in samples]),
        'y': tf.constant([sample['y'] for sample in samples]),
    })


def ragged_batch_to_train_input(*, nodes: tf.RaggedTensor, edges: tf.RaggedTensor, seq_len: tf.Tensor, y: tf.Tensor):
    graph_sizes = tf.cast(nodes.row_lengths(), edges.dtype)
    graph_starts = tf.cast(nodes.row_starts(), edges.dtype)
    edge_offset = tf.stack([tf.zeros_like(graph_starts), graph_starts, graph_starts], axis=-1)

    edges = edges + edge_offset[:, tf.newaxis, :]

    node_positions = tf.ragged.range(graph_sizes)
    mask = tf.less(node_positions, seq_len[:, np.newaxis])
    node_positions = (node_positions.flat_values + 1) * tf.cast(mask.flat_values, node_positions.flat_values.dtype)

    return {
        'edges': edges.flat_values,
        'node_positions': node_positions,
        'nodes': nodes.flat_values,
        'graph_sizes': graph_sizes,
    }, y


def main(args):
    with open(args.data, 'rb') as f:
        data = pickle.load(f)

    gnn_config['base']['hidden_size_orig'] = data['num_types']
    gnn_config['base']['num_edge_types'] = data['num_edge_types']

    model = compy.models.graphs.tf2_sandwich_model.sandwich_model(gnn_config)
    model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])

    rng = np.random.default_rng(123123)
    samples = np.array([s for s in data['samples']])
    rng.shuffle(samples)

    ds = sequence_graphs_dataset(samples).batch(32).map(lambda x: ragged_batch_to_train_input(**x))
    model.fit(ds, epochs=5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Dataset (pickled sequence graphs)")
    main(parser.parse_args())