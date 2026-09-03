"""``fact_store`` action=update must say what actually changed.

2026-09-03: consolidate-synthesize wanted to strip stale queue tags from fid
1056, sent an update carrying only ``content`` (identical to the stored body)
and got ``{"updated": true}`` back. Nothing in that result said the tags were
untouched, so it re-sent the identical call eight times — about 8 of the run's
19 API calls, and the harness's "identical call" nudge did not stop it. The
result now lists the fields that differ before/after, echoes the row's current
category, tags and trust, and names an empty diff as a no-op.
"""

import json

import pytest

from plugins.memory.holographic import HolographicMemoryProvider


BODY = (
    "ANSWERED: fixed in code 2026-08-27 (fork commit ba3bc8beef), and the "
    "premise was wrong twice over. The checks return False because the "
    "features are not configured for this deployment, which is a correct and "
    "permanent verdict rather than a fault; the warning volume was ~27 a day "
    "across nine distinct checks, not 73 across five."
)
assert len(BODY) > 200


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


def _add(provider, content, tags=""):
    out = _call(
        provider,
        {"action": "add", "content": content, "category": "researched", "tags": tags},
    )
    return out["fact_id"]


def test_tagless_update_with_identical_content_is_named_a_noop(provider):
    """The 2026-09-03 shape: content re-sent verbatim, tags never passed."""
    fid = _add(provider, BODY, tags="alpha,beta,gamma")
    out = _call(provider, {"action": "update", "fact_id": fid, "content": BODY})
    assert out["updated"] is True
    assert out["fact_id"] == fid
    assert out["changed"] == []
    assert out["note"].startswith("no-op")
    assert "tags=" in out["note"]
    # The row is echoed so the caller can see the tags it did not change.
    assert out["tags"] == "alpha,beta,gamma"
    assert out["category"] == "researched"


def test_tags_update_lists_tags_and_echoes_the_new_value(provider):
    fid = _add(provider, BODY, tags="alpha,beta,gamma")
    out = _call(provider, {"action": "update", "fact_id": fid, "tags": "gamma"})
    assert out["updated"] is True
    assert out["changed"] == ["tags"]
    assert out["tags"] == "gamma"
    assert "note" not in out


def test_category_and_trust_changes_are_listed(provider):
    fid = _add(provider, BODY)
    out = _call(
        provider,
        {"action": "update", "fact_id": fid, "category": "lesson", "trust_delta": 0.2},
    )
    assert out["changed"] == ["category", "trust_score"]
    assert out["category"] == "lesson"
    assert out["trust_score"] == pytest.approx(0.7)
    assert "note" not in out


def test_missing_fact_reports_not_updated(provider):
    out = _call(provider, {"action": "update", "fact_id": 999999, "tags": "x"})
    assert out == {"updated": False, "fact_id": 999999}


def test_content_wipe_guard_still_refuses_before_any_write(provider):
    """Adding the before/after diff must not weaken the 2026-09-02 guard."""
    fid = _add(provider, BODY, tags="alpha")
    out = _call(
        provider, {"action": "update", "fact_id": fid, "content": "alpha", "tags": "alpha,done"}
    )
    assert "error" in out
    fact = _call(provider, {"action": "get", "fact_id": fid})["fact"]
    assert fact["content"] == BODY
    assert fact["tags"] == "alpha"
