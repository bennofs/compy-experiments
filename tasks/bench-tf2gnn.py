#!/usr/bin/env python3
import argparse
import pickle

import numpy as np
import tensorflow as tf

import compy.models

tf.compat.v1.enable_eager_execution()

gnn_config = {
    'layers': ['ggnn'],
    'batch_size': 32,
    'base': {
        "num_edge_types": None,
        'hidden_size_orig': None,
        'hidden_dim': 32,
    },
    'ggnn': {
        'time_steps': [8],
    },
    'num_epochs': 5,
}


def main(args):
    with open(args.data, 'rb') as f:
        data = pickle.load(f)

    gnn_config['base']['hidden_size_orig'] = data['num_types']
    gnn_config['base']['num_edge_types'] = data['num_edge_types']

    model = compy.models.Tf2SandwichModel(gnn_config)

    rng = np.random.default_rng(123123)
    samples = np.array([s for s in data['samples']])
    rng.shuffle(samples)

    test_len = len(data['samples']) // 10
    test, train = samples[:test_len], samples[test_len:]
    model.train(train, test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Dataset (pickled sequence graphs)")
    main(parser.parse_args())