"""Tests for fact_markers — cross-job completion state that a full tags
replace must not be able to destroy.

`tags` is written with `tags = ?`. Twenty-one nightly cron jobs share that one
column and four of them retype the whole string from memory, so a job that has
never heard of `promoted` deletes it by writing the tags it happens to know.
Three losses are on record in the live store's fact_history:

  fid 919  `promote-candidate` and `promoted`, over two nights
  fid 685  `promote-candidate`
  fid 1268 `deep-dived`, fifteen hours after topic-deep-dive wrote it

Each was patched in the losing job's prompt; the marker table is the invariant
that does not depend on 21 prompts staying right. The load-bearing tests here
are the ones that would have caught those three (test_full_replace_*), the one
that proves removal is still possible (test_clear_*), and the one that proves a
fact with no markers is byte-for-byte unaffected (test_no_markers_*).
"""

import sqlite3

import pytest

from plugins.memory.holographic.store import CROSS_JOB_MARKERS, MemoryStore


@pytest.fixture(autouse=True)
def _clean_shared_registry():
    for entry in list(MemoryStore._shared.values()):
        try:
            entry["conn"].close()
        except sqlite3.Error:
            pass
    MemoryStore._shared.clear()
    yield
    for entry in list(MemoryStore._shared.values()):
        try:
            entry["conn"].close()
        except sqlite3.Error:
            pass
    MemoryStore._shared.clear()


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path / "memory_store.db")
    yield s
    s.close()


def _tags(store, fact_id):
    return store.get_fact(fact_id)["tags"]


def _tagset(store, fact_id):
    return {t.strip() for t in _tags(store, fact_id).split(",") if t.strip()}


# ----------------------------------------------------------------------
# THE PROTECTION
# ----------------------------------------------------------------------

def test_full_replace_cannot_delete_another_jobs_marker(store):
    """fid 919, reproduced: daily-trace-mining retypes tags, `promoted` stays."""
    fid = store.add_fact(
        "LESSON: deferrable tools must be re-issued as a direct call.",
        category="lesson",
        tags="trace-mining,recurring-failure,tool-routing,deferrable-tools",
    )
    # dream-and-promote marks the source promoted (the tags= append pattern
    # every cron prompt already uses).
    store.update_fact(
        fid,
        tags="trace-mining,recurring-failure,tool-routing,deferrable-tools,promoted",
        changed_by="dream-and-promote",
    )
    # daily-trace-mining then retypes the tag string from its own memory,
    # renaming a descriptive tag and dropping the marker it never knew about.
    store.update_fact(
        fid,
        tags="trace-mining,recurring-failure,tool-calling,deferrable-tools",
        changed_by="daily-trace-mining",
    )

    assert "promoted" in _tagset(store, fid)
    assert store.get_markers(fid) == ["promoted"]
    # The caller's own curation still went through — this is not a rollback.
    assert "tool-calling" in _tagset(store, fid)
    assert "tool-routing" not in _tagset(store, fid)


def test_full_replace_restores_several_markers_at_once(store):
    """fid 919 lost `promote-candidate` AND `promoted`. Both come back."""
    fid = store.add_fact(
        "LESSON: two markers on one fact, written by two different jobs.",
        category="lesson",
        tags="trace-mining,promote-candidate,promoted",
    )
    store.update_fact(fid, tags="trace-mining", changed_by="daily-trace-mining")
    assert _tagset(store, fid) >= {"promote-candidate", "promoted"}
    assert store.get_markers(fid) == ["promote-candidate", "promoted"]


def test_a_genuine_curation_still_cannot_drop_a_marker(store):
    """fid 1268: research-open-questions answering a question.

    Every signal a "is this a retype or a curation?" heuristic could read says
    curation — the category changes, the verdict tags are replaced wholesale,
    and the write is entirely correct. It still had no business dropping
    topic-deep-dive's `deep-dived`, which is why re-apply is unconditional.
    """
    fid = store.add_fact(
        "OPEN QUESTION: does the Hessian eigenvalue spread predict divergence?",
        category="open-question",
        tags="dream,unverified,spectral-analysis,Hessian,deep-dived",
    )
    store.update_fact(
        fid,
        category="researched",
        content="PARTIALLY-CONFIRMED: ANSWERED — the spread leads divergence.",
        tags="researched,answered,spectral-analysis,Hessian,verified",
        changed_by="research-open-questions",
    )
    assert "deep-dived" in _tagset(store, fid)
    assert store.get_fact(fid)["category"] == "researched"
    # The verdict transition itself is untouched.
    assert _tagset(store, fid) >= {"researched", "answered", "verified"}
    assert "unverified" not in _tagset(store, fid)


def test_descriptive_tags_are_not_sticky(store):
    """Only CROSS_JOB_MARKERS survive a replace. Everything else is curatable."""
    fid = store.add_fact(
        "LESSON: descriptive tags describe, and must stay freely rewritable.",
        category="lesson",
        tags="verified,answered,research-queue,highlight,promoted",
    )
    store.update_fact(fid, tags="promoted", changed_by="some-job")
    assert _tagset(store, fid) == {"promoted"}


def test_clearing_all_tags_still_keeps_markers(store):
    fid = store.add_fact(
        "LESSON: an empty tags string is the widest possible full replace.",
        category="lesson",
        tags="needs-experiment,spectral-analysis",
    )
    store.update_fact(fid, tags="", changed_by="over-eager-job")
    assert _tags(store, fid) == "needs-experiment"


# ----------------------------------------------------------------------
# REMOVAL — the protection must not make retirement impossible
# ----------------------------------------------------------------------

def test_clear_marker_removes_it_from_both_places(store):
    fid = store.add_fact(
        "LESSON: a promoted entry that the incumbent review later retires.",
        category="lesson",
        tags="hermes,promoted",
    )
    assert store.clear_marker(fid, "promoted", changed_by="dream-and-promote") is True
    assert store.get_markers(fid) == []
    assert _tagset(store, fid) == {"hermes"}


def test_cleared_marker_is_not_resurrected_by_the_next_write(store):
    """The failure mode a naive re-apply would have: removal that undoes itself."""
    fid = store.add_fact(
        "LESSON: a topic re-opened for a second dive.",
        category="lesson",
        tags="hermes,deep-dived",
    )
    store.clear_marker(fid, "deep-dived")
    store.update_fact(fid, tags="hermes,spectral-analysis", changed_by="another-job")
    assert "deep-dived" not in _tagset(store, fid)
    assert store.get_markers(fid) == []


def test_clear_then_reassert_via_tags_records_it_again(store):
    """A caller that puts the marker back is asserting it, not retyping it."""
    fid = store.add_fact(
        "LESSON: markers can be re-asserted after being cleared.",
        category="lesson",
        tags="hermes,deep-dived",
    )
    store.clear_marker(fid, "deep-dived")
    store.update_fact(fid, tags="hermes,deep-dived", changed_by="topic-deep-dive")
    assert store.get_markers(fid) == ["deep-dived"]


def test_clear_marker_on_unknown_fact_and_unset_marker(store):
    fid = store.add_fact(
        "LESSON: clearing what is not there is a no-op, not an error.",
        category="lesson",
        tags="hermes",
    )
    assert store.clear_marker(fid, "promoted") is False
    assert store.clear_marker(999999, "promoted") is False


# ----------------------------------------------------------------------
# NO MARKERS — behaviour must be byte-for-byte what it was before
# ----------------------------------------------------------------------

def test_no_markers_tags_string_is_passed_through_verbatim(store):
    """No reordering, no dedup, no whitespace normalisation.

    Five rows in the live store hold JSON-array-shaped tags and five use
    ", " separators; an unrelated write must not silently reformat them.
    """
    weird = '["paper", "GRACE",  "geodesy"] , grace ,, grace'
    fid = store.add_fact(
        "PAPER: a row whose tags field was written in an odd shape.",
        category="paper",
        tags="paper,GRACE",
    )
    store.update_fact(fid, tags=weird)
    assert _tags(store, fid) == weird
    assert store.get_markers(fid) == []


def test_empty_fact_markers_table_behaves_exactly_as_before(store):
    """The pre-backfill state: markers on disk, nothing in fact_markers."""
    fid = store.add_fact(
        "LESSON: a fact written before the marker table was populated.",
        category="lesson",
        tags="hermes,promoted",
    )
    with store._lock:
        store._conn.execute("DELETE FROM fact_markers")
        store._conn.commit()
    store.update_fact(fid, tags="hermes", changed_by="daily-trace-mining")
    # Exactly today's behaviour: the marker is gone, because nothing knew it
    # was one. This is what the backfill exists to fix.
    assert _tags(store, fid) == "hermes"


def test_fact_markers_self_creates_on_a_preexisting_database(tmp_path):
    """A store opened against a DB that predates the table must not fail.

    Built by dropping the table from a real store rather than hand-rolling a
    legacy schema: facts_fts is external-content, so a facts table without its
    index makes the very first UPDATE raise "database disk image is malformed"
    and the test would be measuring that instead of the migration.
    """
    db = tmp_path / "legacy.db"
    first = MemoryStore(db)
    fid = first.add_fact("LESSON: a row that predates the marker table.",
                         category="lesson", tags="hermes,promoted")
    with first._lock:
        first._conn.execute("DROP TABLE fact_markers")
        first._conn.commit()
    first.close()
    MemoryStore._shared.clear()

    s = MemoryStore(db)
    try:
        assert s.get_markers(fid) == []          # not backfilled yet
        s.update_fact(fid, tags="hermes,promoted", changed_by="dream-and-promote")
        assert s.get_markers(fid) == ["promoted"]
        s.update_fact(fid, tags="hermes", changed_by="daily-trace-mining")
        assert "promoted" in _tagset(s, fid)
    finally:
        s.close()


# ----------------------------------------------------------------------
# API
# ----------------------------------------------------------------------

def test_set_marker_records_and_renders(store):
    fid = store.add_fact(
        "LESSON: set_marker is the explicit form of the tags append.",
        category="lesson",
        tags="hermes",
    )
    assert store.set_marker(fid, "promote-candidate", set_by="daily-review") is True
    assert store.get_markers(fid) == ["promote-candidate"]
    assert _tagset(store, fid) == {"hermes", "promote-candidate"}
    # Idempotent — no duplicate tag, no duplicate row.
    store.set_marker(fid, "promote-candidate")
    assert _tags(store, fid).count("promote-candidate") == 1


def test_set_marker_on_a_fact_with_no_tags(store):
    fid = store.add_fact("LESSON: a fact stored with no tags at all.",
                         category="lesson")
    assert store.set_marker(fid, "promoted") is True
    assert _tags(store, fid) == "promoted"


def test_set_marker_rejects_an_unknown_marker(store):
    fid = store.add_fact("LESSON: unknown markers are refused, not stored.",
                         category="lesson", tags="hermes")
    with pytest.raises(ValueError):
        store.set_marker(fid, "not-a-real-marker")
    assert store.get_markers(fid) == []


def test_set_marker_on_a_missing_fact_returns_false(store):
    assert store.set_marker(999999, "promoted") is False


def test_facts_with_marker_enumerates_and_does_not_substring_match(store):
    """`needs-experiment` must not pick up `retired-experiment`, and
    `deep-dive` must not pick up `deep-dived` — the exact trap the
    scripts/deep-dive-topic.py comment records."""
    a = store.add_fact("HYPOTHESIS: claim A awaiting an empirical test.",
                       category="hypothesis", tags="needs-experiment")
    b = store.add_fact("HYPOTHESIS: claim B, retired as undesignable.",
                       category="hypothesis", tags="retired-experiment")
    c = store.add_fact("SYNTHESIS: topic C, examined in depth this run.",
                       category="synthesis", tags="deep-dived")
    d = store.add_fact("LESSON: source fact D, the origin of a dive.",
                       category="lesson", tags="deep-dive")

    assert [f["fact_id"] for f in store.facts_with_marker("needs-experiment")] == [a]
    assert [f["fact_id"] for f in store.facts_with_marker("retired-experiment")] == [b]
    assert [f["fact_id"] for f in store.facts_with_marker("deep-dived")] == [c]
    assert [f["fact_id"] for f in store.facts_with_marker("deep-dive")] == [d]
    assert store.facts_with_marker("promoted") == []


def test_facts_with_marker_carries_provenance(store):
    fid = store.add_fact("LESSON: provenance travels with the marker.",
                         category="lesson", tags="hermes")
    store.set_marker(fid, "designed", set_by="experiment-design")
    row = store.facts_with_marker("designed")[0]
    assert row["set_by"] == "experiment-design"
    assert row["set_at"]


def test_add_fact_registers_markers_at_birth(store):
    """topic-deep-dive writes tags="deep-dived,…" on the ADD, not an update."""
    fid = store.add_fact(
        "SYNTHESIS: the topic examined in depth, written with its marker.",
        category="synthesis",
        tags="deep-dived,spectral-analysis",
        source_session="topic-deep-dive",
    )
    assert store.get_markers(fid) == ["deep-dived"]
    assert store.facts_with_marker("deep-dived")[0]["set_by"] == "topic-deep-dive"


def test_remove_fact_takes_the_markers_with_it(store):
    fid = store.add_fact("HYPOTHESIS: a prunable row carrying a marker.",
                         category="hypothesis", tags="needs-experiment")
    assert store.remove_fact(fid) is True
    assert store.facts_with_marker("needs-experiment") == []


def test_every_marker_round_trips(store):
    """Guards the constant itself: nothing in the set may be unusable."""
    for i, marker in enumerate(sorted(CROSS_JOB_MARKERS)):
        fid = store.add_fact(f"LESSON: round-trip row number {i} for a marker.",
                             category="lesson", tags="hermes")
        assert store.set_marker(fid, marker) is True
        store.update_fact(fid, tags="hermes", changed_by="forgetful-job")
        assert marker in _tagset(store, fid)
        assert store.clear_marker(fid, marker) is True
        assert store.get_markers(fid) == []


# ----------------------------------------------------------------------
# COMPOSITION with the other two hooks on this write path
# ----------------------------------------------------------------------

def test_history_snapshot_records_the_pre_reapply_tags(store):
    """The snapshot must stay the BEFORE image, not the reconciled string."""
    fid = store.add_fact("LESSON: history keeps the row as it stood.",
                         category="lesson", tags="hermes,promoted")
    store.update_fact(fid, tags="hermes", changed_by="daily-trace-mining")
    with store._lock:
        rows = store._conn.execute(
            "SELECT tags FROM fact_history WHERE fact_id = ? ORDER BY history_id DESC",
            (fid,),
        ).fetchall()
    assert rows[0]["tags"] == "hermes,promoted"


def test_near_duplicate_suppression_does_not_register_markers(store):
    """A suppressed write inserted nothing, so it must claim nothing.

    add_fact returns the standing row's id; if _record_markers ran on the
    caller's tags the standing fact would silently acquire a marker from a
    write that never landed.
    """
    body = (
        "PARTIALLY-CONFIRMED: the reranker dominates query latency at roughly "
        "sixty-eight percent of wall clock, with generation second at twenty-seven "
        "percent and actual vector retrieval under one percent of the total spend."
    )
    first = store.add_fact(body, category="researched", tags="reranker")
    second = store.add_fact(body + " Measured again.", category="researched",
                            tags="reranker,needs-experiment")
    assert second == first                       # suppressed as a near-duplicate
    assert store.get_markers(first) == []
    assert store.facts_with_marker("needs-experiment") == []


def test_exact_duplicate_write_does_not_register_markers(store):
    body = "LESSON: the exact-duplicate branch returns before any marker work."
    first = store.add_fact(body, category="lesson", tags="hermes")
    second = store.add_fact(body, category="lesson", tags="hermes,promoted")
    assert second == first
    assert store.get_markers(first) == []


def test_supersession_demotion_leaves_the_targets_markers_intact(store):
    """_demote_superseded goes through record_feedback, which never writes tags."""
    old = store.add_fact(
        "PARTIALLY-CONFIRMED: the GPU reranker does not fit alongside the model.",
        category="researched", tags="reranker,needs-experiment,promote-candidate",
    )
    store.add_fact(
        f"CONFIRMED: the GPU reranker does fit; this corrects fid {old} — the "
        f"blocker was ollama KV over-reservation, not card capacity.",
        category="researched", tags="reranker,verified",
    )
    assert store.get_fact(old)["trust_score"] == pytest.approx(0.30, abs=1e-6)
    assert store.get_markers(old) == ["needs-experiment", "promote-candidate"]
    assert _tagset(store, old) >= {"needs-experiment", "promote-candidate"}


def test_a_correction_that_also_retypes_tags_keeps_both_behaviours(store):
    """Demotion and marker re-apply must compose on the same target row."""
    old = store.add_fact(
        "PARTIALLY-CONFIRMED: an earlier verdict that a later fact will retract.",
        category="researched", tags="hermes,designed",
    )
    store.add_fact(
        f"REFUTED: the earlier verdict was inverted; this refutes fid {old}.",
        category="researched", tags="hermes,verified",
    )
    store.update_fact(old, tags="hermes,superseded", changed_by="consolidate-synthesize")
    assert store.get_fact(old)["trust_score"] == pytest.approx(0.30, abs=1e-6)
    assert "designed" in _tagset(store, old)
