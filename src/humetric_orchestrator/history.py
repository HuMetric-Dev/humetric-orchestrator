from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from humetric_core import Err, Ok, ParsedQuery, Result

from humetric_orchestrator.errors import (
    HistoryReadFailed,
    HistoryWriteFailed,
    OrchestratorError,
)


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    ts: float
    parsed: ParsedQuery
    embedding: np.ndarray  # 1-D float32


def append_history(
    path: str | Path, parsed: ParsedQuery, embedding: np.ndarray
) -> Result[None, OrchestratorError]:
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return Err(HistoryWriteFailed(path=str(p), reason=f"mkdir: {e}"))

    record = {
        "ts": time.time(),
        "parsed": parsed.model_dump(),
        "embedding": embedding.astype(np.float32).tolist(),
    }
    try:
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError as e:
        return Err(HistoryWriteFailed(path=str(p), reason=str(e)))
    return Ok(None)


def read_history(path: str | Path) -> Result[list[HistoryEntry], OrchestratorError]:
    p = Path(path)
    if not p.exists():
        return Ok([])
    out: list[HistoryEntry] = []
    try:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                rec = json.loads(stripped)
                parsed_dict = dict(rec["parsed"])
                # Pydantic strict mode wants tuples, not lists, for tuple fields.
                for k in ("must_skills", "nice_skills"):
                    v = parsed_dict.get(k)
                    if isinstance(v, list):
                        parsed_dict[k] = tuple(v)
                out.append(
                    HistoryEntry(
                        ts=float(rec["ts"]),
                        parsed=ParsedQuery(**parsed_dict),
                        embedding=np.asarray(rec["embedding"], dtype=np.float32),
                    )
                )
    except (OSError, ValueError, KeyError) as e:
        return Err(HistoryReadFailed(path=str(p), reason=str(e)))
    return Ok(out)


def centroid_last(
    path: str | Path, n: int, *, dim: int | None = None
) -> Result[np.ndarray | None, OrchestratorError]:
    """Mean of the last `n` query embeddings, or None if no history yet."""
    r = read_history(path)
    if isinstance(r, Err):
        return r
    entries = r.value[-n:] if n > 0 else []
    if not entries:
        return Ok(cast("np.ndarray | None", None))
    mat = np.stack([e.embedding for e in entries])
    centroid = mat.mean(axis=0).astype(np.float32)
    if dim is not None and centroid.shape != (dim,):
        return Ok(cast("np.ndarray | None", None))
    return Ok(cast("np.ndarray | None", centroid))
