"""``fact_store`` action=update names the store-managed markers it kept.

``update_fact`` re-applies every cross-job marker a tags write omits
(fact_markers full-replace protection). That is correct, but the tool result
never said so: it echoed the stored tags — marker included — and a model that
had just asked for tags WITHOUT the marker read that as a failed write. On
2026-09-06 experiment-design retired two facts, saw ``needs-experiment`` still
present, spent five calls trying to strip it, and ended the run on a text turn
with no report (incident #42). The result now carries ``markers_kept`` and a
note saying the write landed and the marker stays.
"""

import json

import pytest

from plugins.memory.holographic import HolographicMemoryProvider


CLAIM = (
    "PARTIALLY-CONFIRMED: ParetoBandit's geometric forgetting CAN be applied to "
    "agent context window management — the mechanism is validated for TTL "
    "auto-calibration, and the mapping to adaptive context sizing is "
    "literature-supported; awaiting empirical test: instrument the agent loop "
    "and compare tool-call error rates under adaptive versus fixed sizing."
)


@pytest.fixture
def provider(tmp_path):
    p = HolographicMemoryProvider(
        config={"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": 64}
    )
    p.initialize(session_id="cron_15b2f8e4bf08_20260906_052531")
    yield p
    p.shutdown()


def _call(provider, args):
    return json.loads(provider._handle_fact_store(args))


def _add(provider, tags):
    out = _call(provider, {"action": "add", "content": CLAIM,
                           "category": "researched", "tags": tags})
    return out["fact_id"]


def test_retire_without_the_marker_reports_it_kept(provider):
    """The 2026-09-06 shape: tags=<existing minus needs-experiment>+retired-experiment."""
    fid = _add(provider, "researched,ParetoBandit,needs-experiment")
    out = _call(provider, {"action": "update", "fact_id": fid,
                           "tags": "researched,ParetoBandit,retired-experiment",
                           "trust_delta": -0.1})
    assert out["updated"] is True
    assert "tags" in out["changed"]
    assert out["markers_kept"] == ["needs-experiment"]
    assert "needs-experiment" in out["tags"].split(",")
    assert "retired-experiment" in out["tags"].split(",")
    assert "not an error" in out["note"]
    assert "Do not re-send" in out["note"]


def test_resending_the_same_call_is_a_no_op_that_still_names_the_marker(provider):
    fid = _add(provider, "researched,needs-experiment")
    call = {"action": "update", "fact_id": fid, "tags": "researched,retired-experiment"}
    _call(provider, call)
    out = _call(provider, call)
    assert out["changed"] == []
    assert out["markers_kept"] == ["needs-experiment"]
    assert "no-op" in out["note"] and "KEPT" in out["note"]


def test_tags_that_include_the_marker_get_no_note(provider):
    fid = _add(provider, "researched,needs-experiment")
    out = _call(provider, {"action": "update", "fact_id": fid,
                           "tags": "researched,needs-experiment,designed"})
    assert "markers_kept" not in out
    assert "note" not in out


def test_content_only_update_never_reports_markers(provider):
    fid = _add(provider, "researched,needs-experiment")
    out = _call(provider, {"action": "update", "fact_id": fid,
                           "content": CLAIM + " Addendum: measured 2026-09-06."})
    assert out["changed"] == ["content"]
    assert "markers_kept" not in out


def test_plain_tags_are_never_called_markers(provider):
    # Dropping an ordinary descriptive tag is a normal edit, not a kept marker.
    fid = _add(provider, "researched,ParetoBandit,answered")
    out = _call(provider, {"action": "update", "fact_id": fid,
                           "tags": "researched,ParetoBandit"})
    assert out["tags"] == "researched,ParetoBandit"
    assert "markers_kept" not in out
