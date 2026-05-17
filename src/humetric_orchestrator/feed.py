from __future__ import annotations

from humetric_core import EntityType, Result

from humetric_orchestrator.backend import Explanation, LLMBackend
from humetric_orchestrator.errors import OrchestratorError


def write_feed(
    backend: LLMBackend,
    query_text: str,
    candidates: list[tuple[str, EntityType, str]],
) -> Result[list[Explanation], OrchestratorError]:
    """`candidates` = [(entity_id, entity_type, text_blob)]. Returns one
    Explanation per candidate, preserving input order so callers can group
    by `entity_type` for rendering without re-sorting."""
    return backend.write_explanations(query_text, candidates)
