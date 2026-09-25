# rag-chunker

Dependency-free text chunking strategies for Retrieval-Augmented Generation (RAG) pipelines.

Splitting documents into good chunks is one of the highest-leverage steps in a RAG
system: chunks that are too large dilute retrieval relevance, and chunks that are too
small lose context. `rag-chunker` gives you several well-tested strategies with a
consistent API and **zero runtime dependencies** (pure Python standard library).

## Why

Most chunking utilities are bundled inside large frameworks and drag in heavy
dependencies. `rag-chunker` is a single small module you can vendor into any project,
run offline, and reason about easily.

## Install

```bash
pip install rag-chunker        # once published
# or simply copy the rag_chunker/ package into your project
```

## Strategies

| Function | Splits by | Preserves offsets | Overlap |
|----------|-----------|-------------------|---------|
| `fixed_char_chunks`  | fixed character windows | yes | chars |
| `word_chunks`        | word count              | yes | words |
| `sentence_chunks`    | whole sentences         | no  | sentences |
| `recursive_chunks`   | separator hierarchy     | no  | chars |

## Usage

```python
from rag_chunker import recursive_chunks, sentence_chunks, fixed_char_chunks

text = open("doc.txt", encoding="utf-8").read()

# Recommended default: recursive splitting on semantic boundaries.
for chunk in recursive_chunks(text, max_chars=512, overlap=64):
    print(chunk.index, len(chunk.text))

# Keep whole sentences together, never exceeding max_chars.
chunks = sentence_chunks(text, max_chars=400, overlap_sentences=1)

# Deterministic fixed windows with character offsets back into the source.
for chunk in fixed_char_chunks(text, size=1000, overlap=100):
    assert text[chunk.start:chunk.end] == chunk.text
```

Each function returns a list of `Chunk(text, index, start, end)` dataclass instances.

## Command line

```bash
# From a file
python -m rag_chunker.cli --strategy recursive --max-chars 300 doc.txt

# From stdin, as JSON
cat doc.txt | python -m rag_chunker.cli --strategy sentence --max-chars 300 --json
```

## Development

```bash
python -m unittest discover -s tests -v
```

## License

MIT
