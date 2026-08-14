"""Tests for fact_history tombstones and source_session provenance.

Every update_fact and every delete that proceeds must leave the prior row in
fact_history, so no write path can destroy the only copy of a fact. On
2026-08-14 an unattended consolidation run hard-deleted six protected facts;
two existed in no backup and were recovered only because state.db happened to
still hold the message rows that created them. fact_history makes that
recovery a SELECT.
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


def _history(store, fact_id):
    return store._conn.execute(
        "SELECT * FROM fact_history WHERE fact_id = ? ORDER BY history_id",
        (fact_id,),
    ).fetchall()


class TestDeleteTombstone:
    def test_delete_writes_history_row(self, store):
        fid = store.add_fact("RESEARCHED: verdict to prune", category="researched")
        assert store.remove_fact(fid) is True
        rows = _history(store, fid)
        assert len(rows) == 1
        assert rows[0]["op"] == "delete"
        assert rows[0]["content"] == "RESEARCHED: verdict to prune"
        assert rows[0]["category"] == "researched"

    def test_forced_protected_delete_still_leaves_tombstone(self, store):
        fid = store.add_fact("LESSON: hard-won", category="lesson")
        assert store.remove_fact(fid, force=True) is True
        rows = _history(store, fid)
        assert len(rows) == 1
        assert rows[0]["op"] == "delete"
        assert rows[0]["content"] == "LESSON: hard-won"

    def test_refused_protected_delete_writes_no_history(self, store):
        fid = store.add_fact("LESSON: refused", category="lesson")
        assert store.remove_fact(fid) is False
        assert _history(store, fid) == []
        assert store.get_fact(fid) is not None

    def test_missing_fact_writes_no_history(self, store):
        assert store.remove_fact(99999) is False
        assert _history(store, 99999) == []


class TestUpdateTombstone:
    def test_update_snapshots_prior_content(self, store):
        fid = store.add_fact("HYPOTHESIS: v1", category="hypothesis", tags="a,b")
        store.update_fact(fid, content="HYPOTHESIS: v2")
        rows = _history(store, fid)
        assert len(rows) == 1
        assert rows[0]["op"] == "update"
        assert rows[0]["content"] == "HYPOTHESIS: v1"
        assert rows[0]["tags"] == "a,b"
        assert store.get_fact(fid)["content"] == "HYPOTHESIS: v2"

    def test_tag_wipe_is_recoverable(self, store):
        # The 08-13/08-14 recurrence: a cron run replacing the whole tags
        # field. The prior tags must survive in history.
        fid = store.add_fact("RESEARCHED: tagged", category="researched",
                             tags="open-question,grace")
        store.update_fact(fid, tags="wrong-synthesized-tag")
        rows = _history(store, fid)
        assert rows[0]["tags"] == "open-question,grace"

    def test_successive_updates_keep_every_version(self, store):
        fid = store.add_fact("v1", category="general")
        store.update_fact(fid, content="v2")
        store.update_fact(fid, content="v3")
        store.remove_fact(fid)
        ops = [r["op"] for r in _history(store, fid)]
        contents = [r["content"] for r in _history(store, fid)]
        assert ops == ["update", "update", "delete"]
        assert contents == ["v1", "v2", "v3"]


class TestSourceSession:
    def test_add_fact_stamps_source_session(self, store):
        fid = store.add_fact("stamped", source_session="sess-abc123")
        row = store._conn.execute(
            "SELECT source_session FROM facts WHERE fact_id = ?", (fid,)
        ).fetchone()
        assert row["source_session"] == "sess-abc123"

    def test_default_is_empty_string(self, store):
        fid = store.add_fact("unstamped")
        row = store._conn.execute(
            "SELECT source_session FROM facts WHERE fact_id = ?", (fid,)
        ).fetchone()
        assert row["source_session"] == ""

    def test_history_carries_source_session(self, store):
        fid = store.add_fact("carried", category="general",
                             source_session="sess-xyz")
        store.remove_fact(fid)
        rows = _history(store, fid)
        assert rows[0]["source_session"] == "sess-xyz"

    def test_migration_adds_column_to_existing_db(self, tmp_path):
        # A database created before the column existed must gain it on open.
        db = tmp_path / "old.db"
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE facts (fact_id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " content TEXT NOT NULL UNIQUE, category TEXT DEFAULT 'general',"
            " tags TEXT DEFAULT '', trust_score REAL DEFAULT 0.5,"
            " retrieval_count INTEGER DEFAULT 0, helpful_count INTEGER DEFAULT 0,"
            " created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
            " updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.execute("INSERT INTO facts (content) VALUES ('pre-existing')")
        conn.commit()
        conn.close()

        s = MemoryStore(db)
        try:
            cols = {r[1] for r in s._conn.execute("PRAGMA table_info(facts)")}
            assert "source_session" in cols
            assert "hrr_vector" in cols
            fid = s.add_fact("post-migration", source_session="new-sess")
            s.remove_fact(fid)
            assert _history(s, fid)[0]["source_session"] == "new-sess"
        finally:
            s.close()
