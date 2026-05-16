from __future__ import annotations

from humetric_orchestrator.backend import _PLACEHOLDER_EXPLANATION, _parse_explanations


def _cands(*ids: str) -> list[tuple[str, str]]:
    return [(i, f"text for {i}") for i in ids]


def test_parses_object_form() -> None:
    blob = (
        'Sure! Here:\n{"explanations": ['
        '{"person_id": "gh:alice", "text": "great match"},'
        '{"person_id": "gh:bob",   "text": "ok match"}'
        "]}"
    )
    out = _parse_explanations(blob, _cands("gh:alice", "gh:bob")).unwrap()
    assert [e.person_id for e in out] == ["gh:alice", "gh:bob"]
    assert out[0].text == "great match"
    assert out[1].text == "ok match"


def test_parses_bare_array_form() -> None:
    blob = '[{"person_id": "gh:alice", "text": "great"}]'
    out = _parse_explanations(blob, _cands("gh:alice")).unwrap()
    assert out[0].text == "great"


def test_fills_placeholder_when_llm_skips_candidates() -> None:
    blob = '{"explanations": [{"person_id": "gh:alice", "text": "great"}]}'
    out = _parse_explanations(blob, _cands("gh:alice", "gh:bob", "gh:carol")).unwrap()
    assert [e.person_id for e in out] == ["gh:alice", "gh:bob", "gh:carol"]
    assert out[0].text == "great"
    assert out[1].text == _PLACEHOLDER_EXPLANATION
    assert out[2].text == _PLACEHOLDER_EXPLANATION


def test_reorders_to_candidate_order() -> None:
    blob = (
        '{"explanations": ['
        '{"person_id": "gh:bob", "text": "b-text"},'
        '{"person_id": "gh:alice", "text": "a-text"}'
        "]}"
    )
    out = _parse_explanations(blob, _cands("gh:alice", "gh:bob")).unwrap()
    assert [e.person_id for e in out] == ["gh:alice", "gh:bob"]
    assert out[0].text == "a-text"
    assert out[1].text == "b-text"


def test_ignores_unknown_pids_in_payload() -> None:
    blob = (
        '{"explanations": ['
        '{"person_id": "gh:alice", "text": "great"},'
        '{"person_id": "gh:ghost", "text": "phantom"}'
        "]}"
    )
    out = _parse_explanations(blob, _cands("gh:alice", "gh:bob")).unwrap()
    assert [e.person_id for e in out] == ["gh:alice", "gh:bob"]
    assert out[1].text == _PLACEHOLDER_EXPLANATION


def test_rejects_no_json_at_all() -> None:
    r = _parse_explanations("just prose, no braces", _cands("gh:alice"))
    assert r.is_err()


def test_skips_malformed_entries_without_failing() -> None:
    blob = (
        '{"explanations": ['
        '{"person_id": "gh:alice", "text": "great"},'
        '"not an object",'
        '{"person_id": 123, "text": "wrong type"}'
        "]}"
    )
    out = _parse_explanations(blob, _cands("gh:alice", "gh:bob")).unwrap()
    assert out[0].text == "great"
    assert out[1].text == _PLACEHOLDER_EXPLANATION
