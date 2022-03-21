#!/usr/bin/env python3
import argparse
import pathlib
import tensorflow as tf
import tensorflow.keras as keras
import json

from typing import Tuple

from compy.representations import Graph
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
    info = graph_builder.string_to_info(code)

    try:
        func_info = next(i for i in info.functionInfos if i.name == func_name)
    except StopIteration:
        func_names = " ".join(i.func_name for i in info.functionInfos)
        raise RuntimeError(f"function {func_name} not found in source file, available functions: {func_names}")

    graph = graph_builder.info_to_representation(func_info, visitor=ASTDataCFGTokenVisitor)
    return graph
#%%
CODE = '''
#include <stdlib.h>
#include <stdint.h>
int decode_packet ( char * buf , char * out ) {
    uint8_t pkt_len = buf [ 0 ] ;
    for (int i = 0 ; i < pkt_len; ++ i ) {
        out[i] = buf [i + 1];
    }
    return pkt_len;
}
'''
graph = graph_from_code(CODE, 'decode_packet')
#%%
def flatten_graph(graph: Graph):
    flat = graph.map_to_leaves().without_self_edges()
    return flat
