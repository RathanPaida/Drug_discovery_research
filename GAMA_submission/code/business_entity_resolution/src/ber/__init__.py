"""Business entity resolution pipeline - ML Challenge 2026.

bge-m3 bi-encoder retrieval (+ lexical TF-IDF) -> GBDT pruning ->
bge-reranker-v2-m3 cross-encoder -> GBDT stacker -> expected-F0.5 decisions.
"""
__version__ = "0.1.0"
