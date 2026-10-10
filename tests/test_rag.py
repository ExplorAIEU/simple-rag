from pathlib import Path

import numpy as np

from rag import Chunk, LocalRAG, split_text


def test_split_text_keeps_short_text_intact():
    assert split_text("A short paragraph.") == ["A short paragraph."]


def test_split_text_splits_long_text_with_overlap():
    text = ("First sentence. " * 100) + "\n\n" + ("第二句话。" * 100)
    chunks = split_text(text, chunk_size=300, overlap=50)

    assert len(chunks) > 2
    assert all(chunk.strip() for chunk in chunks)
    assert all(len(chunk) <= 300 for chunk in chunks)


def test_search_returns_best_matching_chunk(tmp_path: Path):
    rag = LocalRAG(tmp_path / "documents", tmp_path / "index")
    rag.chunks = [
        Chunk("annual leave policy", "policy.md", "chunk 1"),
        Chunk("office address", "contact.md", "chunk 1"),
    ]
    rag.vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    rag.client.embeddings = lambda texts, model: np.asarray(
        [[0.9, 0.1]], dtype=np.float32
    )

    results = rag.search("How much leave do I get?", top_k=1)

    assert results[0].chunk.source == "policy.md"
