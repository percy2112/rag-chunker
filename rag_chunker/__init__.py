"""rag-chunker: dependency-free text chunking strategies for RAG pipelines."""

from .chunkers import (
    Chunk,
    fixed_char_chunks,
    word_chunks,
    split_sentences,
    split_paragraphs,
    sentence_chunks,
    recursive_chunks,
)

__all__ = [
    "Chunk",
    "fixed_char_chunks",
    "word_chunks",
    "split_sentences",
    "split_paragraphs",
    "sentence_chunks",
    "recursive_chunks",
]

__version__ = "0.1.0"
