"""Semantic edge comparison; never infer a label map from graph isomorphism."""


def canonical_edges(graph):
    return {tuple(sorted((u, v), key=repr)) for u, v in graph.edges()}
