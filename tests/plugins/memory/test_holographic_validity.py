"""Tests for fact validity windows (`valid_from` / `valid_until`).

Supersession used to be expressed as a trust demotion plus free-text prose
("supersedes fid N"). That answers "which do I rank first" and nothing else —
"was this true on August 12" was not a question the store could be asked, and
an expired fact was only ranked lower, never marked invalid.

The load-bearing cases here are the ones where the window and the trust
demotion come apart, because they share a call site but not a control flow:
a target already sitting at the trust floor gets no demotion and MUST still
get its window closed, and a second retraction must NOT move a window the
first one already closed.
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


def _window(store, fact_id):
    row = store._conn.execute(
        "SELECT created_at, valid_from, valid_until FROM facts WHERE fact_id = ?",
        (fact_id,),
    ).fetchone()
    return row["created_at"], row["valid_from"], row["valid_until"]


def _trust(store, fact_id):
    return store._conn.execute(
        "SELECT trust_score FROM facts WHERE fact_id = ?", (fact_id,)
    ).fetchone()["trust_score"]


class TestSchemaMigration:
    def test_migration_is_idempotent(self, tmp_path):
        """Opening an existing store twice must not raise or duplicate a column."""
        path = tmp_path / "memory_store.db"
        first = MemoryStore(path)
        fid = first.add_fact("LESSON: a fact written before the second open.")
        first.close()
        MemoryStore._shared.clear()

        second = MemoryStore(path)
        cols = [r[1] for r in second._conn.execute("PRAGMA table_info(facts)")]
        assert cols.count("valid_from") == 1
        assert cols.count("valid_until") == 1
        assert second.get_fact(fid) is not None
        second.close()

    def test_a_row_written_by_a_stale_process_is_repaired_on_next_open(self, tmp_path):
        """The invariant is re-established at open, not only at column-add.

        Adding a column does not restart the processes already using the store.
        On 2026-09-02 the migration ran at 00:17 and three facts written at
        00:48 through an MCP bridge started at 00:04 came back with valid_from
        NULL on an already-migrated store — the bridge was still INSERTing
        through the module it imported at startup. The gateway that runs the
        nightly cron is the same shape of process, so a backfill that only fires
        when the column is missing would never have repaired them.
        """
        path = tmp_path / "memory_store.db"
        first = MemoryStore(path)
        fid = first.add_fact("LESSON: written by a process holding the old module.")
        # Exactly what a pre-column INSERT leaves behind on a migrated store.
        first._conn.execute("UPDATE facts SET valid_from = NULL WHERE fact_id = ?", (fid,))
        first._conn.commit()
        assert _window(first, fid)[1] is None
        first.close()
        MemoryStore._shared.clear()

        reopened = MemoryStore(path)
        created, valid_from, _ = _window(reopened, fid)
        assert valid_from == created
        reopened.close()

    def test_legacy_rows_are_backfilled_from_created_at(self, tmp_path):
        """A store written before this column existed must not come back NULL.

        Simulated by dropping the columns from a live store — the same shape
        _init_db sees on the first open after the upgrade.
        """
        path = tmp_path / "memory_store.db"
        first = MemoryStore(path)
        fid = first.add_fact("LESSON: written by the pre-validity code path.")
        first.close()
        MemoryStore._shared.clear()

        raw = sqlite3.connect(path)
        raw.execute("ALTER TABLE facts DROP COLUMN valid_from")
        raw.execute("ALTER TABLE facts DROP COLUMN valid_until")
        raw.commit()
        raw.close()

        migrated = MemoryStore(path)
        created, valid_from, valid_until = _window(migrated, fid)
        assert valid_from == created
        assert valid_until is None
        migrated.close()


class TestWritePath:
    def test_add_fact_is_born_valid_at_its_own_created_at(self, store):
        fid = store.add_fact("LESSON: a newly written fact.")
        created, valid_from, valid_until = _window(store, fid)
        # Same statement, so SQLite evaluates CURRENT_TIMESTAMP once — these
        # must be byte-identical, not merely close.
        assert valid_from == created
        assert valid_until is None

    def test_retraction_closes_the_window_at_the_retracting_fact(self, store):
        old = store.add_fact("LESSON: the reranker does not fit on the GPU.")
        new = store.add_fact(f"LESSON: it does fit — this supersedes fid {old}.")

        _, old_from, old_until = _window(store, old)
        new_created, new_from, new_until = _window(store, new)

        assert old_until == new_created, "window must close at the retraction, not 'now'"
        assert old_from <= old_until, "a window may not close before it opened"
        assert new_until is None, "the correction is still valid"

    def test_successive_windows_abut_and_do_not_overlap(self, store):
        """Three generations of one verdict must tile the timeline."""
        first = store.add_fact("LESSON: generation one of the verdict.")
        second = store.add_fact(f"LESSON: generation two — supersedes fid {first}.")
        third = store.add_fact(f"LESSON: generation three — supersedes fid {second}.")

        w1, w2, w3 = (_window(store, f)[1:] for f in (first, second, third))
        assert w1[1] == w2[0], "gen 1 must end exactly where gen 2 begins"
        assert w2[1] == w3[0], "gen 2 must end exactly where gen 3 begins"
        assert w3[1] is None

    def test_window_closes_even_when_trust_cannot_move(self, store):
        """The case the demotion loop skips — and the reason the stamp is outside it.

        The loop breaks on its first iteration once the target is at the floor,
        so a second retraction demotes nothing. The window must still close,
        because until now the first retraction may not have been recorded at all
        (a fact can arrive already demoted, e.g. restored from fact_history).
        """
        old = store.add_fact("LESSON: a verdict about to be retracted twice.")
        for _ in range(3):
            store.record_feedback(old, helpful=False)
        assert _trust(store, old) == pytest.approx(0.20)
        assert _window(store, old)[2] is None

        new = store.add_fact(f"LESSON: the corrected verdict — supersedes fid {old}.")
        assert _window(store, old)[2] == _window(store, new)[0]
        # Trust was untouched: the loop's floor guard still holds.
        assert _trust(store, old) == pytest.approx(0.20)

    def test_first_close_wins(self, store):
        old = store.add_fact("LESSON: a much-retracted verdict.")
        first_retraction = store.add_fact(f"LESSON: attempt one supersedes fid {old}.")
        closed_at = _window(store, old)[2]
        assert closed_at == _window(store, first_retraction)[0]

        store.add_fact(f"LESSON: attempt two also supersedes fid {old}.")
        assert _window(store, old)[2] == closed_at, "a later retraction must not move it"


class TestNeverClosesTheWrongWindow:
    def test_passive_voice_does_not_close_the_newer_fact(self, store):
        current = store.add_fact("LESSON: the GPU reranker does fit.")
        store.add_fact(
            f"LESSON: the old capacity verdict is superseded by fid {current}."
        )
        assert _window(store, current)[2] is None

    def test_forward_reference_is_ignored(self, store):
        writer = store.add_fact("LESSON: this supersedes fid 999999.")
        assert _window(store, writer)[2] is None

    def test_bare_fid_mention_without_a_retraction_verb_is_ignored(self, store):
        old = store.add_fact("LESSON: a cited but still-valid finding.")
        store.add_fact(f"LESSON: this builds on fid {old} and agrees with it.")
        assert _window(store, old)[2] is None

    def test_duplicate_content_does_not_reopen_or_move_a_window(self, store):
        old = store.add_fact("LESSON: stale verdict.")
        text = f"LESSON: this supersedes fid {old}."
        first = store.add_fact(text)
        closed_at = _window(store, old)[2]
        assert store.add_fact(text) == first
        assert _window(store, old)[2] == closed_at


class TestInvalidateFact:
    def test_refuses_a_superseding_id_that_is_not_strictly_newer(self, store):
        a = store.add_fact("LESSON: the earlier fact.")
        b = store.add_fact("LESSON: the later fact.")
        assert store.invalidate_fact(b, superseded_by=a) is False
        assert store.invalidate_fact(a, superseded_by=a) is False
        assert _window(store, a)[2] is None
        assert _window(store, b)[2] is None

    def test_unknown_superseding_fact_writes_nothing(self, store):
        a = store.add_fact("LESSON: a fact whose retractor does not exist.")
        assert store.invalidate_fact(a, superseded_by=999999) is False
        assert _window(store, a)[2] is None

    def test_returns_false_on_a_window_already_closed(self, store):
        a = store.add_fact("LESSON: the earlier fact.")
        b = store.add_fact("LESSON: the later fact.")
        c = store.add_fact("LESSON: a third, later still.")
        assert store.invalidate_fact(a, superseded_by=b) is True
        assert store.invalidate_fact(a, superseded_by=c) is False
        assert _window(store, a)[2] == _window(store, b)[0]


class TestReadPath:
    def test_get_fact_exposes_the_window(self, store):
        old = store.add_fact("LESSON: the retracted verdict.")
        new = store.add_fact(f"LESSON: the correction — supersedes fid {old}.")

        expired = store.get_fact(old)
        current = store.get_fact(new)
        assert expired["valid_from"] == expired["created_at"]
        assert expired["valid_until"] == current["valid_from"]
        assert current["valid_until"] is None

    def test_list_facts_exposes_the_window(self, store):
        store.add_fact("LESSON: a listed fact.", category="lesson")
        (row,) = store.list_facts(category="lesson", limit=5)
        assert "valid_from" in row and "valid_until" in row
        assert row["valid_until"] is None
