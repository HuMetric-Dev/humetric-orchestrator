from __future__ import annotations

from humetric_core import EntityType

from humetric_orchestrator.backend import _PLACEHOLDER_EXPLANATION, _parse_explanations


def _cands(*ids: str, et: EntityType = "person") -> list[tuple[str, EntityType, str]]:
    return [(i, et, f"text for {i}") for i in ids]


def test_parses_object_form_with_entity_id() -> None:
    blob = (
        'Sure! Here:\n{"explanations": ['
        '{"entity_id": "p:gh:alice", "text": "great match"},'
        '{"entity_id": "p:gh:bob",   "text": "ok match"}'
        "]}"
    )
    out = _parse_explanations(blob, _cands("p:gh:alice", "p:gh:bob")).unwrap()
    assert [e.entity_id for e in out] == ["p:gh:alice", "p:gh:bob"]
    assert all(e.entity_type == "person" for e in out)
    assert out[0].text == "great match"
    assert out[1].text == "ok match"


def test_accepts_legacy_person_id_key() -> None:
    # Older models may continue to emit "person_id" — accept both for now.
    blob = '[{"person_id": "p:gh:alice", "text": "great"}]'
    out = _parse_explanations(blob, _cands("p:gh:alice")).unwrap()
    assert out[0].text == "great"


def test_fills_placeholder_when_llm_skips_candidates() -> None:
    blob = '{"explanations": [{"entity_id": "p:gh:alice", "text": "great"}]}'
    out = _parse_explanations(blob, _cands("p:gh:alice", "p:gh:bob", "p:gh:carol")).unwrap()
    assert [e.entity_id for e in out] == ["p:gh:alice", "p:gh:bob", "p:gh:carol"]
    assert out[0].text == "great"
    assert out[1].text == _PLACEHOLDER_EXPLANATION
    assert out[2].text == _PLACEHOLDER_EXPLANATION


def test_reorders_to_candidate_order() -> None:
    blob = (
        '{"explanations": ['
        '{"entity_id": "p:gh:bob", "text": "b-text"},'
        '{"entity_id": "p:gh:alice", "text": "a-text"}'
        "]}"
    )
    out = _parse_explanations(blob, _cands("p:gh:alice", "p:gh:bob")).unwrap()
    assert [e.entity_id for e in out] == ["p:gh:alice", "p:gh:bob"]
    assert out[0].text == "a-text"
    assert out[1].text == "b-text"


def test_ignores_unknown_eids_in_payload() -> None:
    blob = (
        '{"explanations": ['
        '{"entity_id": "p:gh:alice", "text": "great"},'
        '{"entity_id": "p:gh:ghost", "text": "phantom"}'
        "]}"
    )
    out = _parse_explanations(blob, _cands("p:gh:alice", "p:gh:bob")).unwrap()
    assert [e.entity_id for e in out] == ["p:gh:alice", "p:gh:bob"]
    assert out[1].text == _PLACEHOLDER_EXPLANATION


def test_rejects_no_json_at_all() -> None:
    r = _parse_explanations("just prose, no braces", _cands("p:gh:alice"))
    assert r.is_err()


def test_skips_malformed_entries_without_failing() -> None:
    blob = (
        '{"explanations": ['
        '{"entity_id": "p:gh:alice", "text": "great"},'
        '"not an object",'
        '{"entity_id": 123, "text": "wrong type"}'
        "]}"
    )
    out = _parse_explanations(blob, _cands("p:gh:alice", "p:gh:bob")).unwrap()
    assert out[0].text == "great"
    assert out[1].text == _PLACEHOLDER_EXPLANATION


def test_preserves_organization_entity_type() -> None:
    blob = '[{"entity_id": "o:gh:anthropic", "text": "AI safety lab"}]'
    out = _parse_explanations(blob, _cands("o:gh:anthropic", et="organization")).unwrap()
    assert out[0].entity_type == "organization"
    assert out[0].text == "AI safety lab"
