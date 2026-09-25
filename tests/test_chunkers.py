import unittest

from rag_chunker import (
    fixed_char_chunks,
    word_chunks,
    split_sentences,
    split_paragraphs,
    sentence_chunks,
    recursive_chunks,
)


class TestFixedCharChunks(unittest.TestCase):
    def test_no_overlap_covers_all_text(self):
        text = "abcdefghij"
        chunks = fixed_char_chunks(text, size=4)
        self.assertEqual([c.text for c in chunks], ["abcd", "efgh", "ij"])
        self.assertEqual("".join(c.text for c in chunks), text)

    def test_overlap_repeats_context(self):
        # The final window already reaches the end, so no redundant tail chunk.
        chunks = fixed_char_chunks("abcdef", size=4, overlap=2)
        self.assertEqual([c.text for c in chunks], ["abcd", "cdef"])

    def test_offsets_are_correct(self):
        text = "hello world"
        chunks = fixed_char_chunks(text, size=5)
        for c in chunks:
            self.assertEqual(text[c.start : c.end], c.text)

    def test_invalid_args(self):
        with self.assertRaises(ValueError):
            fixed_char_chunks("x", size=0)
        with self.assertRaises(ValueError):
            fixed_char_chunks("x", size=3, overlap=3)


class TestWordChunks(unittest.TestCase):
    def test_basic_split(self):
        chunks = word_chunks("one two three four five", max_words=2)
        self.assertEqual(
            [c.text for c in chunks], ["one two", "three four", "five"]
        )

    def test_overlap(self):
        chunks = word_chunks("a b c d", max_words=2, overlap=1)
        self.assertEqual([c.text for c in chunks], ["a b", "b c", "c d"])

    def test_empty(self):
        self.assertEqual(word_chunks("   ", max_words=3), [])


class TestSentenceHelpers(unittest.TestCase):
    def test_split_sentences(self):
        text = "Hello world. How are you? I am fine!"
        self.assertEqual(
            split_sentences(text),
            ["Hello world.", "How are you?", "I am fine!"],
        )

    def test_split_paragraphs(self):
        text = "Para one.\n\nPara two.\n\n\nPara three."
        self.assertEqual(
            split_paragraphs(text), ["Para one.", "Para two.", "Para three."]
        )


class TestSentenceChunks(unittest.TestCase):
    def test_respects_max_chars(self):
        text = "Aaaa. Bbbb. Cccc. Dddd."
        chunks = sentence_chunks(text, max_chars=12)
        for c in chunks:
            self.assertLessEqual(len(c.text), 12)
        self.assertTrue(len(chunks) >= 2)

    def test_single_long_sentence_kept_whole(self):
        text = "This one sentence is definitely longer than the limit."
        chunks = sentence_chunks(text, max_chars=10)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, text)


class TestRecursiveChunks(unittest.TestCase):
    def test_never_exceeds_when_separators_available(self):
        text = "word " * 100
        chunks = recursive_chunks(text, max_chars=50)
        for c in chunks:
            self.assertLessEqual(len(c.text), 50)

    def test_reconstructs_content(self):
        text = "alpha beta gamma delta epsilon"
        chunks = recursive_chunks(text, max_chars=12)
        joined = "".join(c.text for c in chunks)
        self.assertEqual(joined.replace(" ", ""), text.replace(" ", ""))

    def test_overlap_carries_context(self):
        text = "abcdefghijklmnop"
        chunks = recursive_chunks(text, max_chars=6, separators=("",), overlap=2)
        self.assertTrue(len(chunks) >= 2)


if __name__ == "__main__":
    unittest.main()
