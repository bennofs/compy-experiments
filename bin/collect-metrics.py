#!/usr/bin/env python3
"""Collect metrics from training runs into a single dataframe."""
import json
import logging
import pathlib
import re

import pandas as pd

from argparse import ArgumentParser
from lib import config


LOGGER = logging.getLogger(__name__)
RE_METRICS_FILENAME = re.compile(r'(?P<hidden_dim>[0-9]+)h-(?P<dropout>[0-9]+\.?[0-9]*)do-(?P<split>[0-9]+)-metrics.json')


def main(args):
    path = pathlib.Path(args.dir)
    count_error = 0
    count_success = 0

    data = {
        'hidden_dim': [],
        'dropout': [],
        'split': [],
        'epoch': [],
        'accuracy': [],
        'loss': [],
        'val_accuracy': [],
        'val_loss': [],
    }
    for metrics_path in path.glob("**/*-metrics.json"):
        match = RE_METRICS_FILENAME.match(metrics_path.name)
        if not match:
            LOGGER.error("metrics file name does not match expected format: %s", metrics_path.name)
            count_error += 1
            continue
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        loss = metrics['loss']
        accuracy = metrics['accuracy']
        val_loss = metrics['val_loss']
        val_accuracy = metrics['val_accuracy']
        num_epochs = len(loss)
        assert num_epochs == len(accuracy)
        assert num_epochs == len(val_loss)
        assert num_epochs == len(val_accuracy)

        data['hidden_dim'] += [int(match['hidden_dim']) for _ in range(num_epochs)]
        data['dropout'] += [float(match['dropout']) for _ in range(num_epochs)]
        data['split'] += [int(match['split']) for _ in range(num_epochs)]
        data['epoch'] += list(range(num_epochs))
        data['accuracy'] += accuracy
        data['loss'] += loss
        data['val_accuracy'] += val_accuracy
        data['val_loss'] += val_loss

        count_success += 1
    LOGGER.info("parsed %d metrics files, %d errors", count_success, count_error)

    df = pd.DataFrame(data)
    df.to_csv(args.dest)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dir", help="Directory containing training metrics")
    parser.add_argument("dest", help="Name of the output file (dataframe serialized as CSV)")
    cli_args = parser.parse_args()
    logging.basicConfig(level='INFO')
    main(cli_args)