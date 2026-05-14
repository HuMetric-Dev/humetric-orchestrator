from __future__ import annotations

from humetric_orchestrator import FakeBackend, parse_query, write_feed


def test_fake_parse_extracts_skill_keywords() -> None:
    be = FakeBackend()
    parsed = parse_query(be, "rust engineer who knows kafka").unwrap()
    assert parsed.free_text == "rust engineer who knows kafka"
    assert "rust" in parsed.must_skills
    assert "kafka" in parsed.nice_skills


def test_fake_parse_with_no_known_skills() -> None:
    be = FakeBackend()
    parsed = parse_query(be, "amazing communicator").unwrap()
    assert parsed.must_skills == ()
    assert parsed.nice_skills == ()


def test_fake_feed_writer_returns_one_per_candidate() -> None:
    be = FakeBackend()
    out = write_feed(
        be,
        "rust engineer",
        [("gh:alice", "rust engineer building distributed systems"), ("gh:bob", "react dev")],
    ).unwrap()
    assert [e.person_id for e in out] == ["gh:alice", "gh:bob"]
    assert all("good match" in e.text for e in out)
