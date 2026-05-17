from __future__ import annotations

import time
from dataclasses import dataclass
from typing import cast

import numpy as np
import psycopg
from humetric_core import Err, Ok, ParsedQuery, Result
from humetric_store import (
    append_query_history,
    recent_query_embeddings,
    recent_query_history,
)

from humetric_orchestrator.errors import (
    HistoryReadFailed,
    HistoryWriteFailed,
    OrchestratorError,
)


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    """In-memory shape of a single history row, for the sidebar UI and tests.
    Backed by the `query_history` Postgres table since v1.5 — the JSONL file
    at ./data/history/queries.jsonl is no longer written or read."""

    ts: float
    parsed: ParsedQuery
    embedding: np.ndarray  # 1-D float32


def append_history(
    conn: psycopg.Connection,
    user_id: str,
    parsed: ParsedQuery,
    embedding: np.ndarray,
) -> Result[None, OrchestratorError]:
    vec = embedding.astype(np.float32).tolist()
    r = append_query_history(
        conn,
        user_id=user_id,
        ts=time.time(),
        parsed=parsed,
        embedding=vec,
    )
    if isinstance(r, Err):
        return Err(HistoryWriteFailed(path=f"query_history({user_id})", reason=str(r.error)))
    return Ok(None)


def read_history(
    conn: psycopg.Connection, user_id: str, n: int = 100
) -> Result[list[HistoryEntry], OrchestratorError]:
    """Return up to `n` most-recent history entries for this user, newest first."""
    r = recent_query_history(conn, user_id, n)
    if isinstance(r, Err):
        return Err(HistoryReadFailed(path=f"query_history({user_id})", reason=str(r.error)))
    out: list[HistoryEntry] = []
    for ts, parsed_dict, embedding in r.value:
        d = dict(parsed_dict)
        # Pydantic strict mode wants tuples, not lists, for tuple fields.
        for k in ("must_skills", "nice_skills", "target_entity_types"):
            v = d.get(k)
            if isinstance(v, list):
                d[k] = tuple(v)
        out.append(
            HistoryEntry(
                ts=ts,
                parsed=ParsedQuery(**d),
                embedding=np.asarray(embedding, dtype=np.float32),
            )
        )
    return Ok(out)


def centroid_last(
    conn: psycopg.Connection, user_id: str, n: int, *, dim: int | None = None
) -> Result[np.ndarray | None, OrchestratorError]:
    """Mean of the last `n` query embeddings for this user, or None if no
    history yet. The personalization reranker feature consumes this; if a
    user has not yet claimed a Person, callers may still get a centroid
    here as long as they've run any queries."""
    if n <= 0:
        return Ok(cast("np.ndarray | None", None))
    r = recent_query_embeddings(conn, user_id, n)
    if isinstance(r, Err):
        return Err(HistoryReadFailed(path=f"query_history({user_id})", reason=str(r.error)))
    rows = r.value
    if not rows:
        return Ok(cast("np.ndarray | None", None))
    mat = np.asarray(rows, dtype=np.float32)
    centroid = mat.mean(axis=0).astype(np.float32)
    if dim is not None and centroid.shape != (dim,):
        return Ok(cast("np.ndarray | None", None))
    return Ok(cast("np.ndarray | None", centroid))
