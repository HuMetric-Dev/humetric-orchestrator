from __future__ import annotations

from humetric_orchestrator.backend import (
    _EXPLANATIONS_GROUNDING_RULE,
    _build_explanations_prompt,
)


def test_grounding_rule_phrases_present() -> None:
    rule = _EXPLANATIONS_GROUNDING_RULE.lower()
    assert "only on the candidate text" in rule
    assert "do not invent" in rule
    assert "say what they do have" in rule


def test_built_prompt_includes_grounding_rule() -> None:
    prompt = _build_explanations_prompt(
        "rust engineer with distributed systems experience",
        [
            ("p:gh:alice", "person", "python dev in toronto"),
            ("p:gh:bob", "person", "react frontend"),
        ],
    )
    assert _EXPLANATIONS_GROUNDING_RULE in prompt


def test_built_prompt_truncates_candidate_text_at_300_chars() -> None:
    long_blob = "x" * 1000
    prompt = _build_explanations_prompt("q", [("p:gh:a", "person", long_blob)])
    assert "x" * 300 in prompt
    assert "x" * 301 not in prompt


def test_built_prompt_has_query_and_candidate_ids() -> None:
    prompt = _build_explanations_prompt("rust engineer", [("p:gh:alice", "person", "rust dev")])
    assert "Query: rust engineer" in prompt
    assert "p:gh:alice" in prompt


def test_built_prompt_labels_entity_types() -> None:
    prompt = _build_explanations_prompt(
        "ai labs and engineers",
        [
            ("o:gh:anthropic", "organization", "AI safety lab"),
            ("p:gh:alice", "person", "rust dev"),
        ],
    )
    assert "(organization)" in prompt
    assert "(person)" in prompt
