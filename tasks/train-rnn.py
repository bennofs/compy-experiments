#!/usr/bin/env python3
import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow.keras
from sklearn.model_selection import StratifiedKFold

tf.compat.v1.enable_eager_execution()
tf.compat.v1.enable_v2_behavior()


def tokens_dataset(samples):
    seqs = [sample['x']['code_rep'].get_sequence_nodes() + 1 for sample in samples]
    padded = tf.keras.preprocessing.sequence.pad_sequences(seqs, padding='post', maxlen=1500)
    y = tf.constant([sample['y'] for sample in samples])
    return tf.data.Dataset.from_tensor_slices((padded, y))


def make_model(*, hidden, vocab_size):
    emb = tf.keras.layers.Embedding(mask_zero=True, input_dim=vocab_size + 1, output_dim=hidden)
    lang = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(hidden // 2))
    out = tf.keras.layers.Dense(1, activation='sigmoid')
    return tf.keras.models.Sequential([emb, lang, out])


#%%
def main(args):
    print("load data from:", args.data)
    data_name = os.path.basename(args.data)
    with open(args.data, 'rb') as f:
        data = pickle.load(f)

    # Filter samples which are too long
    samples = np.array(data['samples'])
    print("total samples: {}".format(len(samples)))

    vocab_size = int(data['num_types'])
    hidden_dim = int(args.hidden)

    tf.compat.v1.enable_eager_execution()
    tf.compat.v1.enable_v2_behavior()

    global MODEL
    MODEL = make_model(hidden=hidden_dim, vocab_size=vocab_size)

    with open('config.json', 'w') as f:
        json.dump(MODEL.get_config(), f)

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    MODEL.compile(opt, 'binary_crossentropy', metrics=['accuracy'])
    sample_input, sample_label = next(iter(tokens_dataset(samples[:1]).batch(32)))
    MODEL(sample_input)
    print("parameter count", MODEL.count_params())

    # Train and test
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=204)
    split = kf.split(samples, [sample["y"] for sample in samples])
    for i, (train_idx, test_idx) in enumerate(split):
        rng = np.random.default_rng(seed=0)
        train_samples = samples[train_idx]
        rng.shuffle(train_samples)
        train_data = tokens_dataset(train_samples).batch(32)
        test_data = tokens_dataset(samples[test_idx]).batch(32)

        model = make_model(hidden=hidden_dim, vocab_size=vocab_size)
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(opt, 'binary_crossentropy', metrics=['accuracy'])
        history_callback = model.fit(train_data, validation_data=test_data, epochs=250, callbacks=[
            tf.keras.callbacks.TensorBoard(Path.home() / f'tb-train-rnn/{data_name}/{i:02}-{args.hidden}h'),
            tf.keras.callbacks.ModelCheckpoint(f'{args.hidden}h-{i:02}-{{epoch:03}}.h5', save_weights_only=True)
        ])
        with open(f'{args.hidden}h-{i:02}-metrics.json', 'w') as f:
            json.dump(history_callback.history, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", metavar="DATA", help="Pickled dataset to use for training", default="cvevulns-tokens.pickle")
    parser.add_argument("--vocab", metavar="VOCAB", help="Saved vocabulary for the dataset", default="cvevulns-tokens.vocab.bin.npz")
    parser.add_argument('--hidden', metavar="DIM", help="Hidden state dimension")

    main(parser.parse_args())
