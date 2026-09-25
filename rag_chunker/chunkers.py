"""Text chunking strategies for Retrieval-Augmented Generation.

All functions are pure, dependency-free, and operate on plain ``str`` input.
Chunk boundaries are reported as character offsets into the *original* text
where meaningful, so callers can map a chunk back to its source location.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence

__all__ = [
    "Chunk",
    "fixed_char_chunks",
    "word_chunks",
    "split_sentences",
    "split_paragraphs",
    "sentence_chunks",
    "recursive_chunks",
]

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_BOUNDARY = re.compile(r"\n\s*\n")
_WORD = re.compile(r"\S+")


@dataclass(frozen=True)
class Chunk:
    """A single chunk of text.

    ``start`` / ``end`` are character offsets into the original text. They are
    ``-1`` when a strategy cannot cheaply preserve exact offsets.
    """

    text: str
    index: int
    start: int = -1
    end: int = -1


def fixed_char_chunks(text: str, size: int, overlap: int = 0) -> List[Chunk]:
    """Split ``text`` into fixed-size character windows.

    ``overlap`` characters are repeated between consecutive chunks, which helps
    preserve context across boundaries in RAG retrieval.
    """
    if size <= 0:
        raise ValueError("size must be > 0")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be in the range [0, size)")

    chunks: List[Chunk] = []
    step = size - overlap
    index = 0
    for start in range(0, len(text), step):
        piece = text[start : start + size]
        if not piece:
            break
        chunks.append(Chunk(piece, index, start, start + len(piece)))
        index += 1
        if start + size >= len(text):
            break
    return chunks


def word_chunks(text: str, max_words: int, overlap: int = 0) -> List[Chunk]:
    """Split ``text`` into chunks of at most ``max_words`` words."""
    if max_words <= 0:
        raise ValueError("max_words must be > 0")
    if overlap < 0 or overlap >= max_words:
        raise ValueError("overlap must be in the range [0, max_words)")

    matches = list(_WORD.finditer(text))
    if not matches:
        return []

    chunks: List[Chunk] = []
    step = max_words - overlap
    index = 0
    for i in range(0, len(matches), step):
        window = matches[i : i + max_words]
        if not window:
            break
        start = window[0].start()
        end = window[-1].end()
        chunks.append(Chunk(text[start:end], index, start, end))
        index += 1
        if i + max_words >= len(matches):
            break
    return chunks


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using simple terminal-punctuation heuristics."""
    parts = _SENTENCE_BOUNDARY.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def split_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs on blank lines."""
    parts = _PARAGRAPH_BOUNDARY.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def sentence_chunks(text: str, max_chars: int, overlap_sentences: int = 0) -> List[Chunk]:
    """Group whole sentences into chunks no longer than ``max_chars``.

    Sentences are never split mid-way. ``overlap_sentences`` trailing sentences
    are carried into the next chunk to preserve context.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if overlap_sentences < 0:
        raise ValueError("overlap_sentences must be >= 0")

    sentences = split_sentences(text)
    chunks: List[Chunk] = []
    current: List[str] = []
    index = 0

    def flush() -> None:
        nonlocal index, current
        if current:
            chunks.append(Chunk(" ".join(current), index))
            index += 1

    for sentence in sentences:
        candidate = (" ".join(current + [sentence])).strip()
        if current and len(candidate) > max_chars:
            flush()
            current = current[-overlap_sentences:] if overlap_sentences else []
        current.append(sentence)
    flush()
    return chunks


def recursive_chunks(
    text: str,
    max_chars: int,
    separators: Sequence[str] = ("\n\n", "\n", ". ", " ", ""),
    overlap: int = 0,
) -> List[Chunk]:
    """Recursively split on a hierarchy of separators, then greedily merge.

    Mirrors the popular "recursive character" strategy: it prefers to break on
    the largest semantic separator that keeps pieces under ``max_chars`` before
    falling back to finer-grained ones.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be in the range [0, max_chars)")

    pieces = _split_recursive(text, list(separators), max_chars)
    merged = _merge(pieces, max_chars, overlap)
    return [Chunk(piece, i) for i, piece in enumerate(merged)]


def _split_recursive(text: str, separators: List[str], max_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text] if text else []

    separator = separators[0] if separators else ""
    remaining = separators[1:] if len(separators) > 1 else [""]

    if separator == "":
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

    parts = text.split(separator)
    out: List[str] = []
    for i, part in enumerate(parts):
        piece = part + (separator if i < len(parts) - 1 else "")
        if not piece:
            continue
        if len(piece) <= max_chars:
            out.append(piece)
        else:
            out.extend(_split_recursive(piece, remaining, max_chars))
    return out


def _merge(pieces: List[str], max_chars: int, overlap: int) -> List[str]:
    chunks: List[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
        elif len(current) + len(piece) <= max_chars:
            current += piece
        else:
            chunks.append(current)
            current = (current[-overlap:] + piece) if overlap else piece
    if current:
        chunks.append(current)
    return chunks
