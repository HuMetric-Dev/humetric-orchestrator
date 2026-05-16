from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from humetric_core import Err, Ok, ParsedQuery, Result

from humetric_orchestrator.errors import (
    BackendCallFailed,
    BackendMisconfigured,
    OrchestratorError,
    ParseRejected,
)

_PARSE_SYSTEM_PROMPT = """\
You parse free-text recruiting queries into a structured JSON object. Be \
literal about must-haves vs nice-to-haves. Only fields present in the schema \
are allowed. If the user did not state a value, leave the field as its default \
(null/empty). The `free_text` field should retain the user's original phrasing, \
useful for semantic search.\
"""

_PARSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "free_text": {"type": "string"},
        "must_skills": {"type": "array", "items": {"type": "string"}},
        "nice_skills": {"type": "array", "items": {"type": "string"}},
        "location": {"type": ["string", "null"]},
        "min_followers": {"type": ["integer", "null"]},
        "min_years_experience": {"type": ["integer", "null"]},
    },
    "required": ["free_text"],
    "additionalProperties": False,
}


@dataclass(frozen=True, slots=True)
class Explanation:
    person_id: str
    text: str


class LLMBackend(Protocol):
    name: str

    def parse_query(self, text: str) -> Result[ParsedQuery, OrchestratorError]: ...

    def write_explanations(
        self,
        query: str,
        candidates: list[tuple[str, str]],
    ) -> Result[list[Explanation], OrchestratorError]: ...


def _validate_parsed_query(payload: object) -> Result[ParsedQuery, OrchestratorError]:
    if not isinstance(payload, dict):
        return Err(ParseRejected(detail=f"expected object, got {type(payload).__name__}"))
    typed: dict[str, Any] = {str(k): v for k, v in payload.items()}
    # Pydantic strict mode rejects lists where tuples are declared; the LLM
    # always returns lists, so coerce at the boundary.
    for k in ("must_skills", "nice_skills"):
        v = typed.get(k)
        if isinstance(v, list):
            typed[k] = tuple(v)
    try:
        return Ok(ParsedQuery(**typed))
    except (TypeError, ValueError) as e:
        return Err(ParseRejected(detail=str(e)))


@dataclass(slots=True)
class FakeBackend:
    """A test backend that uses simple keyword rules. Useful for unit tests."""

    name: str = "fake"
    canned_explanation: str = "good match"

    def parse_query(self, text: str) -> Result[ParsedQuery, OrchestratorError]:
        tokens = [t.lower() for t in text.split()]
        skills_seen = [
            t for t in tokens if t in {"rust", "python", "go", "kafka", "react", "payments"}
        ]
        return _validate_parsed_query(
            {
                "free_text": text,
                "must_skills": tuple(skills_seen[:1]),
                "nice_skills": tuple(skills_seen[1:]),
                "location": None,
                "min_followers": None,
                "min_years_experience": None,
            }
        )

    def write_explanations(
        self, query: str, candidates: list[tuple[str, str]]
    ) -> Result[list[Explanation], OrchestratorError]:
        return Ok(
            [
                Explanation(person_id=pid, text=f"{self.canned_explanation}: {text[:60]}")
                for pid, text in candidates
            ]
        )


@dataclass(slots=True)
class AnthropicBackend:
    """Claude API backend via tool-use for structured query parsing."""

    api_key: str | None = None
    model: str = "claude-opus-4-7"
    name: str = "anthropic"

    def _client(self) -> Result[Any, OrchestratorError]:
        try:
            import anthropic
        except ImportError as e:
            return Err(BackendMisconfigured(backend=self.name, detail=f"import anthropic: {e}"))
        try:
            return Ok(anthropic.Anthropic(api_key=self.api_key))
        except (ValueError, TypeError, AttributeError, OSError) as e:
            return Err(BackendMisconfigured(backend=self.name, detail=str(e)))

    def parse_query(self, text: str) -> Result[ParsedQuery, OrchestratorError]:
        client_r = self._client()
        if isinstance(client_r, Err):
            return client_r
        client = client_r.value

        try:
            import anthropic
        except ImportError as e:
            return Err(BackendMisconfigured(backend=self.name, detail=f"import anthropic: {e}"))

        tools = [
            {
                "name": "emit_parsed_query",
                "description": "Emit the structured ParsedQuery from a recruiter's free-text query.",
                "input_schema": _PARSE_JSON_SCHEMA,
            }
        ]
        try:
            resp = client.messages.create(
                model=self.model,
                max_tokens=512,
                system=_PARSE_SYSTEM_PROMPT,
                tools=tools,
                tool_choice={"type": "tool", "name": "emit_parsed_query"},
                messages=[{"role": "user", "content": text}],
            )
        except (anthropic.APIError, anthropic.APIConnectionError, anthropic.APIStatusError) as e:
            return Err(BackendCallFailed(backend=self.name, detail=str(e)))

        for block in getattr(resp, "content", []):
            if getattr(block, "type", None) == "tool_use":
                return _validate_parsed_query(block.input)
        return Err(ParseRejected(detail="no tool_use block in response"))

    def write_explanations(
        self, query: str, candidates: list[tuple[str, str]]
    ) -> Result[list[Explanation], OrchestratorError]:
        if not candidates:
            return Ok([])
        client_r = self._client()
        if isinstance(client_r, Err):
            return client_r
        client = client_r.value

        bullets = "\n".join(f"- {pid}: {text[:300]}" for pid, text in candidates)
        prompt = (
            f"Query: {query}\n\n"
            f"Candidates:\n{bullets}\n\n"
            "For each candidate, write a 1-2 sentence explanation of why they match. "
            'Respond with a JSON object: {"explanations": [{"person_id": str, "text": str}, ...]}. '
            "No prose."
        )

        try:
            import anthropic
        except ImportError as e:
            return Err(BackendMisconfigured(backend=self.name, detail=f"import anthropic: {e}"))
        try:
            resp = client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
        except (anthropic.APIError, anthropic.APIConnectionError, anthropic.APIStatusError) as e:
            return Err(BackendCallFailed(backend=self.name, detail=str(e)))

        text = "".join(getattr(b, "text", "") for b in getattr(resp, "content", []))
        return _parse_explanations(text, candidates)


@dataclass(slots=True)
class OpenAIBackend:
    """OpenAI-compatible backend. Targets vLLM by setting `base_url`."""

    base_url: str | None = None
    api_key: str = "dummy"
    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    name: str = "openai"

    def _client(self) -> Result[Any, OrchestratorError]:
        try:
            import openai
        except ImportError as e:
            return Err(BackendMisconfigured(backend=self.name, detail=f"import openai: {e}"))
        try:
            return Ok(openai.OpenAI(api_key=self.api_key, base_url=self.base_url))
        except (ValueError, TypeError, AttributeError, OSError) as e:
            return Err(BackendMisconfigured(backend=self.name, detail=str(e)))

    def parse_query(self, text: str) -> Result[ParsedQuery, OrchestratorError]:
        client_r = self._client()
        if isinstance(client_r, Err):
            return client_r
        client = client_r.value

        try:
            import openai
        except ImportError as e:
            return Err(BackendMisconfigured(backend=self.name, detail=f"import openai: {e}"))

        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _PARSE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Return ONLY a JSON object matching this schema:\n"
                        f"{json.dumps(_PARSE_JSON_SCHEMA)}\n\nQuery: {text}",
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=512,
            )
        except (openai.APIError, openai.APIConnectionError, openai.APIStatusError) as e:
            return Err(BackendCallFailed(backend=self.name, detail=str(e)))

        try:
            content = resp.choices[0].message.content or "{}"
            payload = json.loads(content)
        except (ValueError, IndexError, AttributeError) as e:
            return Err(ParseRejected(detail=f"bad response: {e}"))
        return _validate_parsed_query(payload)

    def write_explanations(
        self, query: str, candidates: list[tuple[str, str]]
    ) -> Result[list[Explanation], OrchestratorError]:
        if not candidates:
            return Ok([])
        client_r = self._client()
        if isinstance(client_r, Err):
            return client_r
        client = client_r.value

        bullets = "\n".join(f"- {pid}: {text[:300]}" for pid, text in candidates)
        prompt = (
            f"Query: {query}\n\nCandidates:\n{bullets}\n\n"
            "For each candidate, write a 1-2 sentence explanation of why they match. "
            'Respond with a JSON object: {"explanations": [{"person_id": str, "text": str}, ...]}. '
            "No prose."
        )
        try:
            import openai
        except ImportError as e:
            return Err(BackendMisconfigured(backend=self.name, detail=f"import openai: {e}"))
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=2048,
            )
        except (openai.APIError, openai.APIConnectionError, openai.APIStatusError) as e:
            return Err(BackendCallFailed(backend=self.name, detail=str(e)))

        text = resp.choices[0].message.content or ""
        return _parse_explanations(text, candidates)


_PLACEHOLDER_EXPLANATION = "Surfaced by retrieval; the LLM did not return a specific rationale."


def _parse_explanations(
    blob: str, candidates: list[tuple[str, str]]
) -> Result[list[Explanation], OrchestratorError]:
    """Parse the LLM's explanations payload and align to `candidates` 1:1.

    Accepts either `{"explanations": [...]}` (current prompt) or a bare list
    (older prompt / lenient model). Out-of-order entries are reordered to match
    `candidates`; missing pids get a placeholder so the feed always has one
    row per candidate.
    """
    start_obj = blob.find("{")
    end_obj = blob.rfind("}")
    start_arr = blob.find("[")
    end_arr = blob.rfind("]")
    has_obj = start_obj >= 0 and end_obj > start_obj
    has_arr = start_arr >= 0 and end_arr > start_arr
    # Whichever outer bracket appears first wins; otherwise prefer whatever's present.
    prefer_arr = has_arr and (not has_obj or start_arr < start_obj)

    raw: object
    if prefer_arr:
        try:
            raw = json.loads(blob[start_arr : end_arr + 1])
        except ValueError as e:
            return Err(ParseRejected(detail=f"json: {e}"))
    elif has_obj:
        try:
            parsed = json.loads(blob[start_obj : end_obj + 1])
        except ValueError as e:
            return Err(ParseRejected(detail=f"json: {e}"))
        if not isinstance(parsed, dict):
            return Err(ParseRejected(detail="explanations payload is not an object"))
        raw = parsed.get("explanations", [])
    else:
        return Err(ParseRejected(detail="no JSON object or array in explanations response"))

    if not isinstance(raw, list):
        return Err(ParseRejected(detail="explanations field is not a list"))

    by_pid: dict[str, str] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        pid = item.get("person_id")
        text = item.get("text")
        if isinstance(pid, str) and isinstance(text, str):
            by_pid.setdefault(pid, text)

    out = [
        Explanation(person_id=pid, text=by_pid.get(pid, _PLACEHOLDER_EXPLANATION))
        for pid, _ in candidates
    ]
    return Ok(out)
