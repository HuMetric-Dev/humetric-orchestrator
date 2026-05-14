from __future__ import annotations

from humetric_core import ParsedQuery, Result

from humetric_orchestrator.backend import LLMBackend
from humetric_orchestrator.errors import OrchestratorError


def parse_query(backend: LLMBackend, text: str) -> Result[ParsedQuery, OrchestratorError]:
    """Single entrypoint for free-text → ParsedQuery. Delegates to the configured backend."""
    return backend.parse_query(text)
