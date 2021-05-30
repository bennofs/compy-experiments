#!/usr/bin/env python3
import argparse
import pickle
import numpy as np

import compy.models
from compy.representations.sequence_graph import Vocabulary

gnn_config = {
    "num_timesteps": 8,
    "hidden_size_orig": None,
    "gnn_h_size": 32,
    "gnn_m_size": 2,
    "learning_rate": 0.001,
    "batch_size": 32,
    "num_epochs": 5,
}

def main(args):
    with open(args.data, 'rb') as f:
        data = pickle.load(f)

    gnn_config['hidden_size_orig'] = data['num_types']
    gnn_config['num_edge_types'] = data['num_edge_types']
    vocab = Vocabulary.unnamed(data['num_types'], data['num_edge_types'])

    model = compy.models.GnnPytorchGeomModel(gnn_config)

    rng = np.random.default_rng(123123)
    samples = np.array([to_common_graph_sample(s, vocab) for s in data['samples']])
    rng.shuffle(samples)

    test_len = len(data['samples']) // 10
    test, train = samples[:test_len], samples[test_len:]
    model.train(train, test)


def to_common_graph_sample(sample, vocab):
    out = dict(sample)
    out['x'] = dict(sample['x'], code_rep=sample['x']['code_rep'].to_graph(vocab))
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Dataset (pickled sequence graphs)")
    main(parser.parse_args())