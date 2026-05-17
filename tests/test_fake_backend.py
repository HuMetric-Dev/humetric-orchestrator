from __future__ import annotations

from humetric_orchestrator import FakeBackend, parse_query, write_feed


def test_fake_parse_extracts_skill_keywords() -> None:
    be = FakeBackend()
    parsed = parse_query(be, "rust engineer who knows kafka").unwrap()
    assert parsed.free_text == "rust engineer who knows kafka"
    assert "rust" in parsed.must_skills
    assert "kafka" in parsed.nice_skills
    assert parsed.target_entity_types == ("person",)


def test_fake_parse_with_no_known_skills() -> None:
    be = FakeBackend()
    parsed = parse_query(be, "amazing communicator").unwrap()
    assert parsed.must_skills == ()
    assert parsed.nice_skills == ()


def test_fake_parse_org_intent_routes_to_organization() -> None:
    be = FakeBackend()
    parsed = parse_query(be, "firms similar to anthropic").unwrap()
    assert parsed.target_entity_types == ("organization",)


def test_fake_parse_mixed_intent_routes_to_both() -> None:
    be = FakeBackend()
    parsed = parse_query(be, "rust engineers at firms hiring in SF").unwrap()
    assert parsed.target_entity_types == ("person", "organization")


def test_fake_feed_writer_returns_one_per_candidate() -> None:
    be = FakeBackend()
    out = write_feed(
        be,
        "rust engineer",
        [
            ("p:gh:alice", "person", "rust engineer building distributed systems"),
            ("p:gh:bob", "person", "react dev"),
        ],
    ).unwrap()
    assert [(e.entity_id, e.entity_type) for e in out] == [
        ("p:gh:alice", "person"),
        ("p:gh:bob", "person"),
    ]
    assert all("good match" in e.text for e in out)


def test_fake_feed_writer_carries_organization_type() -> None:
    be = FakeBackend()
    out = write_feed(
        be,
        "ai labs",
        [("o:gh:anthropic", "organization", "AI safety lab in San Francisco")],
    ).unwrap()
    assert out[0].entity_id == "o:gh:anthropic"
    assert out[0].entity_type == "organization"
