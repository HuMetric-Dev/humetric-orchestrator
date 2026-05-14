from __future__ import annotations

from humetric_core import Result

from humetric_orchestrator.backend import Explanation, LLMBackend
from humetric_orchestrator.errors import OrchestratorError


def write_feed(
    backend: LLMBackend,
    query_text: str,
    candidates: list[tuple[str, str]],
) -> Result[list[Explanation], OrchestratorError]:
    """`candidates` = [(person_id, text_blob)]. Returns one explanation per candidate."""
    return backend.write_explanations(query_text, candidates)
