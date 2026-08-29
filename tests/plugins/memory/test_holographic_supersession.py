"""Tests for automatic demotion of facts a later fact explicitly retracts.

Ranking is ``cross_encoder_score * trust_score`` and the cross-encoder saturates
(measured 0.9997-0.9988 across eight on-topic facts), so trust is effectively the
only discriminator. Before this hook, writing a retraction did nothing to what it
retracted: on 2026-08-28 one query returned four retracted answers above both
current ones, and another returned first a verdict a later fact calls "INVERTED".

The dangerous direction is demoting the wrong row, so the wrong-direction and
passive-voice cases below are the load-bearing tests, not the happy path.
"""

import sqlite3

import pytest

from plugins.memory.holographic.store import MemoryStore


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


def _trust(store, fact_id):
    return store._conn.execute(
        "SELECT trust_score FROM facts WHERE fact_id = ?", (fact_id,)
    ).fetchone()["trust_score"]


class TestDemotesWhatItRetracts:
    @pytest.mark.parametrize(
        "phrasing",
        [
            "This supersedes fid {old} — measured again at n=20.",
            "PROJECT — ladder settled, superseding fid {old}'s keep-Q4 call.",
            "CORRECTS fid {old}, whose Playwright claim was never tested.",
            "Refutes fid {old}: the card had the capacity all along.",
            "This invalidates fid={old} and re-opens the question.",
            "supersedes FID #{old} after the control run.",
        ],
    )
    def test_each_retraction_phrasing_demotes_the_target(self, store, phrasing):
        old = store.add_fact("LESSON: the reranker does not fit on the GPU.")
        assert _trust(store, old) == pytest.approx(0.5)
        store.add_fact("LESSON: " + phrasing.format(old=old))
        # Two steps of _UNHELPFUL_DELTA, floored at 0.30.
        assert _trust(store, old) == pytest.approx(0.30)

    def test_demotion_is_recorded_in_fact_history(self, store):
        old = store.add_fact("LESSON: stale verdict.")
        new = store.add_fact(f"LESSON: this supersedes fid {old}.")
        rows = store._conn.execute(
            "SELECT op, changed_by_session FROM fact_history WHERE fact_id = ?", (old,)
        ).fetchall()
        assert [r["op"] for r in rows] == ["feedback", "feedback"]
        assert all(f"retracted by fid {new}" in r["changed_by_session"] for r in rows)

    def test_floor_holds_against_repeated_retractions(self, store):
        old = store.add_fact("LESSON: much-retracted verdict.")
        for i in range(4):
            store.add_fact(f"LESSON: attempt {i} — this supersedes fid {old}.")
        # Never driven below the search floor, however many facts retract it.
        assert _trust(store, old) == pytest.approx(0.30)


class TestNeverDemotesTheWrongRow:
    def test_passive_voice_does_not_demote_the_newer_fact(self, store):
        """`superseded by fid N` names the NEWER fact — demoting N inverts it."""
        current = store.add_fact("LESSON: the GPU reranker does fit.")
        store.add_fact(
            f"LESSON: the old capacity verdict is superseded by fid {current}."
        )
        assert _trust(store, current) == pytest.approx(0.5)

    def test_forward_reference_is_ignored(self, store):
        """A fact cannot retract one written after it."""
        writer = store.add_fact("LESSON: this supersedes fid 999999.")
        assert _trust(store, writer) == pytest.approx(0.5)

    def test_self_reference_is_ignored(self, store):
        first = store.add_fact("LESSON: placeholder to advance the id.")
        text = "LESSON: this supersedes fid {sid}."
        sid = store.add_fact(text.format(sid=first + 1))
        assert sid == first + 1
        assert _trust(store, sid) == pytest.approx(0.5)

    def test_bare_fid_mention_without_a_retraction_verb_is_ignored(self, store):
        old = store.add_fact("LESSON: a cited but still-valid finding.")
        store.add_fact(f"LESSON: this builds on fid {old} and agrees with it.")
        assert _trust(store, old) == pytest.approx(0.5)

    def test_missing_target_does_not_raise(self, store):
        real = store.add_fact("LESSON: a real fact.")
        store.remove_fact(real)
        # The write must still succeed even though the target is gone.
        assert store.add_fact(f"LESSON: supersedes fid {real}.") > 0


class TestWriteIsNeverLost:
    def test_duplicate_content_does_not_demote_twice(self, store):
        old = store.add_fact("LESSON: stale verdict.")
        text = f"LESSON: this supersedes fid {old}."
        first = store.add_fact(text)
        assert _trust(store, old) == pytest.approx(0.30)
        # Re-writing identical content returns the existing id and must not
        # push the target any further down.
        assert store.add_fact(text) == first
        assert _trust(store, old) == pytest.approx(0.30)

    def test_a_failing_demotion_still_returns_the_new_fact_id(self, store, monkeypatch):
        old = store.add_fact("LESSON: stale verdict.")

        def boom(*a, **kw):
            raise RuntimeError("record_feedback exploded")

        monkeypatch.setattr(MemoryStore, "record_feedback", boom)
        new = store.add_fact(f"LESSON: this supersedes fid {old}.")
        assert new > old
        assert store.get_fact(new) is not None
        assert _trust(store, old) == pytest.approx(0.5)
