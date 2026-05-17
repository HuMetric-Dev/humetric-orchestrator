from __future__ import annotations

import numpy as np
import psycopg
from humetric_core import ParsedQuery

from humetric_orchestrator import append_history, centroid_last, read_history


def test_append_and_read_roundtrip(pg_conn: psycopg.Connection, user_id: str) -> None:
    q = ParsedQuery(free_text="rust", must_skills=("rust",))
    emb = np.array([1.0] + [0.0] * 1023, dtype=np.float32)
    append_history(pg_conn, user_id, q, emb).unwrap()
    append_history(
        pg_conn,
        user_id,
        ParsedQuery(free_text="payments"),
        np.array([0.0, 1.0] + [0.0] * 1022, dtype=np.float32),
    ).unwrap()

    entries = read_history(pg_conn, user_id).unwrap()
    assert len(entries) == 2
    # newest-first
    assert entries[0].parsed.free_text == "payments"
    assert entries[1].parsed.free_text == "rust"
    np.testing.assert_allclose(entries[1].embedding, emb)


def test_centroid_of_last_n(pg_conn: psycopg.Connection, user_id: str) -> None:
    for i in range(5):
        vec = np.zeros(1024, dtype=np.float32)
        vec[0] = float(i)
        append_history(pg_conn, user_id, ParsedQuery(free_text=str(i)), vec).unwrap()
    c = centroid_last(pg_conn, user_id, 3).unwrap()
    assert c is not None
    # last 3 entries were i=2,3,4; centroid of 2,3,4 in column 0 is 3.0
    assert c[0] == 3.0


def test_centroid_empty_history(pg_conn: psycopg.Connection, user_id: str) -> None:
    c = centroid_last(pg_conn, user_id, 5).unwrap()
    assert c is None


def test_centroid_dim_mismatch_returns_none(pg_conn: psycopg.Connection, user_id: str) -> None:
    vec = np.zeros(1024, dtype=np.float32)
    vec[0] = 1.0
    append_history(pg_conn, user_id, ParsedQuery(free_text="x"), vec).unwrap()
    c = centroid_last(pg_conn, user_id, 5, dim=8).unwrap()
    assert c is None


def test_history_isolated_per_user(pg_conn: psycopg.Connection, user_id: str) -> None:
    from humetric_core import User, new_user_id
    from humetric_store import insert_user

    other = new_user_id()
    insert_user(
        pg_conn,
        User(id=other, email="other@example.com", display_name="Other", created_at=0.0),
    ).unwrap()

    append_history(
        pg_conn,
        user_id,
        ParsedQuery(free_text="for first user"),
        np.zeros(1024, dtype=np.float32),
    ).unwrap()

    entries = read_history(pg_conn, other).unwrap()
    assert entries == []
