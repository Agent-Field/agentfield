"""Regression tests for agentic_rag skills."""

from skills import deduplicate_chunks


def test_deduplicate_chunks_exact_duplicates():
    chunks = [
        {"text": "The quick brown fox jumps over the lazy dog."},
        {"text": "The quick brown fox jumps over the lazy dog."},
        {"text": "A completely different sentence for testing."},
    ]
    result = deduplicate_chunks(chunks, similarity_threshold=0.9)
    assert len(result) == 2
    assert result[0]["text"] == chunks[0]["text"]
    assert result[1]["text"] == chunks[2]["text"]


def test_deduplicate_chunks_near_duplicates():
    base = " ".join(str(i) for i in range(1, 20))
    near = " ".join(str(i) for i in range(1, 19)) + " twenty"
    chunks = [
        {"text": base},
        {"text": near},
        {"text": "something else entirely that does not match"},
    ]
    result = deduplicate_chunks(chunks, similarity_threshold=0.9)
    assert len(result) == 2
    assert result[0]["text"] == base
    assert result[1]["text"] == chunks[2]["text"]


def test_deduplicate_chunks_respects_threshold():
    base = "alpha beta gamma delta epsilon"
    similar = "alpha beta gamma delta zeta"
    chunks = [
        {"text": base},
        {"text": similar},
    ]
    high = deduplicate_chunks(chunks, similarity_threshold=0.9)
    assert len(high) == 2

    low = deduplicate_chunks(chunks, similarity_threshold=0.5)
    assert len(low) == 1


def test_deduplicate_chunks_empty_input():
    assert deduplicate_chunks([]) == []
