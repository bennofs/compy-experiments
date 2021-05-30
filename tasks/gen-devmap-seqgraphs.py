#!/usr/bin/env python3
import argparse
import os
import pathlib
import pickle

from compy.representations.sequence_graph import SequenceGraphBuilder
from compy.representations.ast_graphs import ASTGraphBuilder, ASTDataCFGTokenVisitor
from compy.representations.extractors.extractors import ClangDriver
from compy.datasets import OpenCLDevmapDataset


def main(args):
    dataset = OpenCLDevmapDataset()
    driver = ClangDriver(
        ClangDriver.ProgrammingLanguage.OpenCL,
        ClangDriver.OptimizationLevel.O0,
        [(x, ClangDriver.IncludeDirType.User) for x in dataset.additional_include_dirs],
        ["-xcl", "-target", "x86_64-pc-linux-gnu"],
    )
    graph_builder = ASTGraphBuilder(driver)
    builder = SequenceGraphBuilder(graph_builder)
    data = dataset.preprocess(builder, visitor=ASTDataCFGTokenVisitor)
    data['num_edge_types'] = len(builder.vocabulary().edge_kinds)

    os.makedirs(args.outdir, exist_ok=True)
    outdir = pathlib.Path(args.outdir)
    with open(outdir / 'compy-devmap-seqgraphs.pickle', 'wb') as f:
        pickle.dump(data, f)

    builder.vocabulary().save(outdir / 'compy-devmap-vocab.npz')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("outdir", help="Destination directory for generated files")
    main(parser.parse_args())