#!/usr/bin/env python3
import argparse
import pathlib
import tensorflow as tf
import tensorflow.keras as keras
import json

from typing import Tuple

from compy.representations.extractors import ClangDriver, clang_binary_path
from compy.representations.ast_graphs import ASTGraphBuilder, ASTDataCFGTokenVisitor
from compy.representations.sequence_graph import SequenceGraphBuilder, SequenceGraph, Vocabulary
from compy.models.graphs.tf2_sandwich_model import sandwich_model


tf.compat.v1.enable_eager_execution()
tf.compat.v1.enable_v2_behavior()


def input_from_graph(graph: SequenceGraph):
    nodes = tf.concat([graph.get_sequence_nodes(), graph.get_non_sequence_nodes()], axis=0)
    edges = graph.edges.T
    return {
        'edges': edges,
        'nodes': nodes,
        'graph_ids': tf.zeros_like(nodes),
        'mask': tf.ones_like(nodes, dtype=tf.bool),
        'seq_shape': tf.constant([1, graph.seq_len]),
        'node_positions': tf.constant([0 if i is None else 1 + i for i in graph.get_node_positions()]),
    }


def graph_from_code(code, func_name) -> Tuple[SequenceGraph, Vocabulary]:
    driver = ClangDriver(ClangDriver.ProgrammingLanguage.C, ClangDriver.OptimizationLevel.O0, [], ["-w"])
    driver.setCompilerBinary(clang_binary_path())
    graph_builder = ASTGraphBuilder(driver)
    builder = SequenceGraphBuilder(graph_builder)
    info = builder.string_to_info(code)

    try:
        func_info = next(i for i in info.functionInfos if i.name == func_name)
    except StopIteration:
        func_names = " ".join(i.func_name for i in info.functionInfos)
        raise RuntimeError(f"function {func_name} not found in source file, available functions: {func_names}")

    graph = builder.info_to_representation(func_info, visitor=ASTDataCFGTokenVisitor)
    return graph, builder.vocabulary()


def main(args):
    # convert code to graph
    code = pathlib.Path(args.sample).read_bytes()
    graph, vocab = graph_from_code(code, args.func)

    # load the model
    config_path = pathlib.Path(args.model).parent / "config.json" if args.config is None else pathlib.Path(args.config)
    with config_path.open('rb') as f:
        config = json.load(f)

    # run prediction
    model = sandwich_model(config, rnn_dense=True)
    model.load_weights(args.model)
    inp = input_from_graph(graph)
    predictions = model(inp, training=False)
    print(f"prediction: {predictions[0,1].numpy() * 100:02.0f}% probability of being vulnerable", )

    return graph, vocab, config, model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sample', help="Path to sample to process")
    parser.add_argument('--model', help="Path to the model to use for predictions", required=True)
    parser.add_argument('--config', help="Path to the config json for the given model, defaults to config.json in the same directory as the model", default=None)
    parser.add_argument('--func', help="Name of the function to analyse", default="target")

    ARGS = parser.parse_args()
    main(ARGS)