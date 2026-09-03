"""Content-wipe guard on ``fact_store`` action=update.

Two real incidents, months apart, same shape — a model "updating" a fact
replaced its whole body because ``content`` is a full-replacement field:

  2026-08-10  consolidate rewrote three ANSWERED facts into one-liners taken
              from 60-char digest previews, destroying their sourced answers.
  2026-09-02  experiment-design was told to add a lifecycle tag ("designed" /
              "retired-experiment") WITHOUT touching content; it passed the
              tag word as ``content`` too, wiping fids 882 (769 chars) and
              901 (1419 chars) to 18- and 8-char strings. Its own prompt said
              "do NOT rewrite its content". Prose did not hold; this guard is
              the code-side invariant, per the remove_fact precedent.

The guard refuses an update whose ``content`` (a) equals one of the fact's
tags — supplied or already stored — or (b) shrinks a >200-char body below
40 chars. A refused call must change nothing and the error must tell the
model the correct re-issue shape (tags= without content).
"""

import json

import pytest

from plugins.memory.holographic import HolographicMemoryProvider


LONG_BODY = (
    "PARTIALLY-CONFIRMED: FFT-based spectral monitoring CAN serve as early "
    "warning for agent drift — the underlying mechanism is validated but the "
    "specific cron-duration application remains untested. Literature support "
    "spans spectral analysis of periodic variance and tool-usage drift "
    "detection at production scale, none of it rig-specific yet."
)
assert len(LONG_BODY) > 200


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


def _body(provider, fid):
    return _call(provider, {"action": "get", "fact_id": fid})["fact"]["content"]


# -- the 2026-09-02 shape: lifecycle tag passed as content ------------------


def test_content_equal_to_supplied_tag_refused_and_nothing_changes(provider):
    fid = _add(provider, LONG_BODY, tags="verified,needs-experiment")
    out = _call(
        provider,
        {
            "action": "update",
            "fact_id": fid,
            "content": "retired-experiment",
            "tags": "verified,needs-experiment,retired-experiment",
            "trust_delta": -0.1,
        },
    )
    assert "error" in out, f"expected refusal, got {out}"
    assert "content" in out["error"] and "tags" in out["error"]
    # The refusal is atomic: body, tags and trust all untouched.
    fact = _call(provider, {"action": "get", "fact_id": fid})["fact"]
    assert fact["content"] == LONG_BODY
    assert fact["tags"] == "verified,needs-experiment"


def test_content_equal_to_already_stored_tag_refused(provider):
    """The tag need not be in the same call — 'designed' may already be set."""
    fid = _add(provider, LONG_BODY, tags="needs-experiment,designed")
    out = _call(
        provider, {"action": "update", "fact_id": fid, "content": "designed"}
    )
    assert "error" in out, f"expected refusal, got {out}"
    assert _body(provider, fid) == LONG_BODY


# -- the 2026-08-10 shape: long body shrunk to a stub -----------------------


def test_shrinking_long_body_to_stub_refused(provider):
    fid = _add(provider, LONG_BODY)
    out = _call(
        provider,
        {"action": "update", "fact_id": fid, "content": "one-liner rewrite"},
    )
    assert "error" in out, f"expected refusal, got {out}"
    assert _body(provider, fid) == LONG_BODY


# -- what must keep working -------------------------------------------------


def test_tags_only_lifecycle_update_still_works(provider):
    """The correct STEP 3 call: tags replaced, content untouched."""
    fid = _add(provider, LONG_BODY, tags="verified,needs-experiment")
    out = _call(
        provider,
        {
            "action": "update",
            "fact_id": fid,
            "tags": "verified,needs-experiment,designed",
        },
    )
    assert out["updated"] is True
    fact = _call(provider, {"action": "get", "fact_id": fid})["fact"]
    assert fact["content"] == LONG_BODY
    assert "designed" in fact["tags"]


def test_full_rewrite_still_works(provider):
    fid = _add(provider, LONG_BODY)
    new_body = "REFUTED: " + LONG_BODY
    out = _call(
        provider, {"action": "update", "fact_id": fid, "content": new_body}
    )
    assert out["updated"] is True
    assert _body(provider, fid) == new_body


def test_short_fact_short_content_still_works(provider):
    """The guard protects substantial bodies; short notes stay editable."""
    fid = _add(provider, "tool note: use ruff")
    out = _call(
        provider, {"action": "update", "fact_id": fid, "content": "tool note: use ty"}
    )
    assert out["updated"] is True
    assert _body(provider, fid) == "tool note: use ty"
