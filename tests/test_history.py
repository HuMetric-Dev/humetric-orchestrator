from __future__ import annotations

from pathlib import Path

import numpy as np
from humetric_core import ParsedQuery

from humetric_orchestrator import append_history, centroid_last, read_history


def test_append_and_read_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "history.jsonl"
    q = ParsedQuery(free_text="rust", must_skills=("rust",))
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    append_history(p, q, emb).unwrap()
    append_history(
        p, ParsedQuery(free_text="payments"), np.array([0.0, 1.0, 0.0], dtype=np.float32)
    ).unwrap()

    entries = read_history(p).unwrap()
    assert len(entries) == 2
    assert entries[0].parsed.free_text == "rust"
    assert entries[1].parsed.free_text == "payments"
    np.testing.assert_array_equal(entries[0].embedding, emb)


def test_centroid_of_last_n(tmp_path: Path) -> None:
    p = tmp_path / "history.jsonl"
    for i in range(5):
        append_history(
            p, ParsedQuery(free_text=str(i)), np.array([float(i), 0.0], dtype=np.float32)
        ).unwrap()
    c = centroid_last(p, 3).unwrap()
    assert c is not None
    np.testing.assert_allclose(c, np.array([3.0, 0.0], dtype=np.float32))


def test_centroid_empty_history(tmp_path: Path) -> None:
    p = tmp_path / "absent.jsonl"
    c = centroid_last(p, 5).unwrap()
    assert c is None


def test_centroid_dim_mismatch_returns_none(tmp_path: Path) -> None:
    p = tmp_path / "history.jsonl"
    append_history(p, ParsedQuery(free_text="x"), np.array([1.0, 2.0], dtype=np.float32)).unwrap()
    c = centroid_last(p, 5, dim=8).unwrap()
    assert c is None
