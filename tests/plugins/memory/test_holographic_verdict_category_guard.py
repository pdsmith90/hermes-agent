"""Verdict/queue header must match the category on ``fact_store`` action=update.

The prompts' CATEGORY HYGIENE rule (prefix and category move together in ONE
update) kept failing in prose: research-open-questions left fids 963/985/989
(2026-08-21) and three more (2026-08-29) as ANSWERED bodies in
category=open-question; consolidate rewrote fids 886/896 to "REFUTED: ..." on
2026-09-04 and left them as lessons. Each is the queue/answer hybrid that
scripts/store-stats.py alerts on and someone repaired by hand. This guard
mirrors store-stats' VERDICT_HEADER and refuses the write, naming the
re-issue shape. Category-only repairs and tags-only updates are untouched.
"""

import json

import pytest

from plugins.memory.holographic import HolographicMemoryProvider


LESSON = (
    "LESSON: Hermes errors.log on 2026-08-11 showed 2951 WARNING lines, 710 of "
    "them check_fn registry notices and 696 of those returned False. Background "
    "probe noise of this shape is cosmetic: the checks gate features that are "
    "not configured for this deployment, so the volume tells you nothing about "
    "health and should not be spectrally monitored as if it did."
)
REFUTED = (
    "REFUTED: (by fid 1056): the check_fn registry warnings are permanent "
    "not-configured verdicts, not faults, so monitoring their spectrum is "
    "meaningless; the 2026-08-27 fix logs each check's first failure once."
)
assert len(LESSON) > 200 and len(REFUTED) > 40


@pytest.fixture
def provider(tmp_path):
    p = HolographicMemoryProvider(
        config={"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": 64}
    )
    p.initialize(session_id="test-session")
    yield p
    p.shutdown()


def _call(provider, args):
    return json.loads(provider._handle_fact_store(args))


def _add(provider, content, category, tags=""):
    out = _call(provider, {"action": "add", "content": content, "category": category, "tags": tags})
    return out["fact_id"]


def _fact(provider, fid):
    return _call(provider, {"action": "get", "fact_id": fid})["fact"]


def test_refuted_body_without_category_is_refused_and_nothing_changes(provider):
    """The 2026-09-04 consolidate shape: REFUTED body, trust cut, category forgotten."""
    fid = _add(provider, LESSON, "lesson", tags="postmortem,check_fn")
    out = _call(provider, {"action": "update", "fact_id": fid, "content": REFUTED,
                           "tags": "postmortem,check_fn,refuted", "trust_delta": -0.2})
    assert "error" in out, out
    assert 'category="researched"' in out["error"]
    fact = _fact(provider, fid)
    assert fact["content"] == LESSON
    assert fact["category"] == "lesson"
    assert fact["tags"] == "postmortem,check_fn"
    assert fact["trust_score"] == pytest.approx(0.5)


def test_same_call_with_category_researched_succeeds(provider):
    fid = _add(provider, LESSON, "lesson")
    out = _call(provider, {"action": "update", "fact_id": fid, "content": REFUTED,
                           "category": "researched", "trust_delta": -0.2})
    assert out["updated"] is True
    assert set(out["changed"]) == {"content", "category", "trust_score"}
    assert out["category"] == "researched"


def test_verdict_body_on_an_already_researched_fact_needs_no_category(provider):
    fid = _add(provider, "CONFIRMED: an earlier verdict with enough body to be a real fact " * 2, "researched")
    out = _call(provider, {"action": "update", "fact_id": fid, "content": REFUTED})
    assert out["updated"] is True
    assert out["changed"] == ["content"]


def test_reverse_direction_question_header_in_researched_fact(provider):
    """A researched fact rewritten to open 'OPEN QUESTION:' must move to open-question."""
    fid = _add(provider, "CONFIRMED: something settled, with a body long enough to matter " * 2, "researched")
    q = "OPEN QUESTION: does the settled claim still hold after the 2026-09 refactor? (raised 2026-09-04)"
    out = _call(provider, {"action": "update", "fact_id": fid, "content": q})
    assert "error" in out and 'category="open-question"' in out["error"]
    out = _call(provider, {"action": "update", "fact_id": fid, "content": q, "category": "open-question"})
    assert out["updated"] is True and out["category"] == "open-question"


def test_category_only_repair_of_an_existing_hybrid_passes(provider):
    """The prescribed repair must never be blocked by the guard."""
    fid = _add(provider, LESSON, "lesson")
    provider._store.update_fact(fid, content=REFUTED)  # bypass the tool: manufacture a hybrid
    assert _fact(provider, fid)["category"] == "lesson"
    out = _call(provider, {"action": "update", "fact_id": fid, "category": "researched"})
    assert out["updated"] is True and out["changed"] == ["category"]


def test_tags_only_update_on_a_hybrid_is_untouched(provider):
    fid = _add(provider, LESSON, "lesson")
    provider._store.update_fact(fid, content=REFUTED)
    out = _call(provider, {"action": "update", "fact_id": fid, "tags": "aged"})
    assert out["updated"] is True and out["changed"] == ["tags"]


def test_plain_content_rewrite_on_a_lesson_is_untouched(provider):
    fid = _add(provider, LESSON, "lesson")
    out = _call(provider, {"action": "update", "fact_id": fid, "content": LESSON + " Addendum: still true on 2026-09-04."})
    assert out["updated"] is True and out["changed"] == ["content"]
