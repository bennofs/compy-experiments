import pickle

from compy.datasets.cvevulns import CVEVulnsDataset
from compy.representations.ast_graphs import ASTDataCFGTokenVisitor, ASTGraphBuilder
from compy.representations.sequence_graph import SequenceGraphBuilder, Vocabulary
from compy.representations.extractors.extractors import ClangDriver


def main():
    driver = ClangDriver(ClangDriver.ProgrammingLanguage.C, ClangDriver.OptimizationLevel.O0, [], ["-w"])
    graph_builder = ASTGraphBuilder(driver)
    builder = SequenceGraphBuilder(graph_builder)

    dataset = CVEVulnsDataset()
    data = dataset.preprocess(builder, ASTDataCFGTokenVisitor)

    with open("cvevulns-tokens.pickle", 'wb') as f:
        pickle.dump(data, f)

    builder.vocabulary().save("cvevulns-tokens.vocab.bin")


if __name__ == '__main__':
    main()