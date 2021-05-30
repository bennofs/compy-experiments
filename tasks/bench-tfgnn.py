#!/usr/bin/env python3
import argparse
import pickle
import numpy as np

import compy.models
from compy.representations.sequence_graph import Vocabulary

gnn_config = {
    "graph_rnn_cell": "gru",
    "num_timesteps": 8,
    "hidden_size_orig": None,
    "gnn_h_size": 32,
    "gnn_m_size": 2,
    "num_edge_types": 4,
    "prediction_cell": {
        "mlp_f_m_dims": [32, 32],
        "mlp_f_m_activation": "relu",
        "mlp_g_m_dims": [32, 32],
        "mlp_g_m_activation": "relu",
        "mlp_reduce_dims": [32, 32],
        "mlp_reduce_activation": "relu",
        "mlp_reduce_out_dim": 64,
        "mlp_reduce_after_aux_in_1_dims": [],
        "mlp_reduce_after_aux_in_1_activation": "relu",
        "mlp_reduce_after_aux_in_1_out_dim": 32,
        "mlp_reduce_after_aux_in_2_dims": [],
        "mlp_reduce_after_aux_in_2_activation": "sigmoid",
        "mlp_reduce_after_aux_in_2_out_dim": 2,
        "output_dim": 2,
    },
    "embedding_layer": {"mapping_dims": [32, 32]},
    "learning_rate": 0.001,
    "clamp_gradient_norm": 1.0,
    "L2_loss_factor": 0,
    "batch_size": 64,
    "num_epochs": 5,
    "tie_fwd_bkwd": 0,
    "use_edge_bias": 0,
    "save_best_model_interval": 1,
    "with_aux_in": 1,
    "seed": 0,
}


def main(args):
    with open(args.data, 'rb') as f:
        data = pickle.load(f)

    gnn_config['hidden_size_orig'] = data['num_types']
    gnn_config['num_edge_types'] = data['num_edge_types']
    vocab = Vocabulary.unnamed(data['num_types'], data['num_edge_types'])

    model = compy.models.GnnTfModel(gnn_config)

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