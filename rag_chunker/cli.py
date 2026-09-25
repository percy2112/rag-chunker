"""Command-line interface for rag-chunker.

Usage:
    python -m rag_chunker.cli --strategy recursive --max-chars 200 file.txt
    cat file.txt | python -m rag_chunker.cli --strategy sentence --max-chars 300
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List

from .chunkers import (
    Chunk,
    fixed_char_chunks,
    recursive_chunks,
    sentence_chunks,
    word_chunks,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chunk text for RAG pipelines.")
    parser.add_argument("file", nargs="?", help="Input file (defaults to stdin).")
    parser.add_argument(
        "--strategy",
        choices=("fixed", "word", "sentence", "recursive"),
        default="recursive",
    )
    parser.add_argument("--max-chars", type=int, default=512)
    parser.add_argument("--max-words", type=int, default=100)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    return parser


def _read_input(path: str | None) -> str:
    if path:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    return sys.stdin.read()


def _run(args: argparse.Namespace, text: str) -> List[Chunk]:
    if args.strategy == "fixed":
        return fixed_char_chunks(text, args.max_chars, args.overlap)
    if args.strategy == "word":
        return word_chunks(text, args.max_words, args.overlap)
    if args.strategy == "sentence":
        return sentence_chunks(text, args.max_chars, args.overlap)
    return recursive_chunks(text, args.max_chars, overlap=args.overlap)


def main(argv: List[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    text = _read_input(args.file)
    chunks = _run(args, text)

    if args.json:
        payload = [
            {"index": c.index, "start": c.start, "end": c.end, "text": c.text}
            for c in chunks
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for chunk in chunks:
            print(f"--- chunk {chunk.index} ({len(chunk.text)} chars) ---")
            print(chunk.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
