"""MemoryStore(readonly=True) — the viewer's handle on the fact store.

The memory wiki runs the same FactRetriever pipeline as the agent to rank a
person's search, and that pipeline has two write paths a viewer must never
take: MemoryStore.__init__ migrates the schema (_init_db), and search() counts
every returned row in retrieval_count, which the metabolism reads as agent
usage. These tests pin that a read-only store takes neither, ranks exactly as
a writer store does, and never shares a connection with one.
"""
from __future__ import annotations

import hashlib
import sqlite3

import pytest

pytest.importorskip("numpy")  # retrieval module imports numpy indirectly

from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore


def _seed(path):
    store = MemoryStore(str(path))
    try:
        store.add_fact("The deployment rollback failed on stale migration state.",
                       category="project")
        store.add_fact("Rollback of the gateway deployment needs a quiet restart.",
                       category="lesson")
        store.add_fact("Compaction threshold tuned to 0.85.", category="tool")
    finally:
        store.close()


def _counts(path):
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return conn.execute(
            "SELECT fact_id, retrieval_count FROM facts ORDER BY fact_id").fetchall()
    finally:
        conn.close()


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_readonly_search_ranks_like_a_writer_and_counts_nothing(tmp_path):
    db = tmp_path / "memory_store.db"
    _seed(db)
    before = _counts(db)

    ro = MemoryStore(str(db), readonly=True)
    try:
        got = FactRetriever(ro).search("deployment rollback", limit=5)
    finally:
        ro.close()
    assert _counts(db) == before, "a read-only search bumped retrieval_count"

    rw = MemoryStore(str(db))
    try:
        want = FactRetriever(rw).search("deployment rollback", limit=5)
    finally:
        rw.close()
    assert [r["fact_id"] for r in got] == [r["fact_id"] for r in want]
    assert len(got) == 2
    # ...and the writer path still counts, so the flag is what made the difference.
    assert _counts(db) != before


def test_readonly_store_refuses_writes(tmp_path):
    db = tmp_path / "memory_store.db"
    _seed(db)
    ro = MemoryStore(str(db), readonly=True)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            ro.add_fact("A viewer must not be able to write this.", category="lesson")
    finally:
        ro.close()


def test_readonly_open_runs_no_schema_init(tmp_path):
    """_init_db migrates and sets WAL mode; opening read-only must do neither.

    A column the migration would re-add is dropped first, so a read-only open
    that ran _init_db would either fail on its ALTER or bring the column back.
    The seeding writer is closed first, so no WAL is pending and the file must
    stay byte-identical.
    """
    db = tmp_path / "memory_store.db"
    _seed(db)
    conn = sqlite3.connect(str(db))
    conn.execute("ALTER TABLE facts DROP COLUMN source_session")
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    digest = _digest(db)

    ro = MemoryStore(str(db), readonly=True)
    try:
        cols = {r[1] for r in ro._conn.execute("PRAGMA table_info(facts)")}
        assert "source_session" not in cols, "a read-only open migrated the schema"
        ro.mark_retrieved([1, 2, 3])
    finally:
        ro.close()
    assert _digest(db) == digest


def test_readonly_search_facts_counts_nothing(tmp_path):
    """search_facts bumps retrieval_count inline; on a read-only store it must
    skip that rather than raise 'attempt to write a readonly database'."""
    db = tmp_path / "memory_store.db"
    _seed(db)
    before = _counts(db)
    ro = MemoryStore(str(db), readonly=True)
    try:
        rows = ro.search_facts("rollback")
    finally:
        ro.close()
    assert len(rows) == 2
    assert _counts(db) == before


def test_readonly_and_writer_never_share_a_connection(tmp_path):
    db = tmp_path / "memory_store.db"
    _seed(db)
    rw = MemoryStore(str(db))
    ro = MemoryStore(str(db), readonly=True)
    try:
        assert ro._conn is not rw._conn
        # The writer keeps working while the viewer is open, and the viewer
        # sees the write.
        fid = rw.add_fact("Written while a viewer is open.", category="lesson")
        row = ro._conn.execute(
            "SELECT content FROM facts WHERE fact_id = ?", (fid,)).fetchone()
        assert row["content"] == "Written while a viewer is open."
        # A second viewer shares the first viewer's connection, not the writer's.
        ro2 = MemoryStore(str(db), readonly=True)
        try:
            assert ro2._conn is ro._conn
        finally:
            ro2.close()
    finally:
        ro.close()
    # Closing the viewer left the writer's connection alone.
    rw._conn.execute("SELECT 1").fetchone()
    rw.close()


def test_readonly_never_creates_a_missing_store(tmp_path):
    missing = tmp_path / "nowhere" / "memory_store.db"
    with pytest.raises(sqlite3.OperationalError):
        MemoryStore(str(missing), readonly=True)
    assert not missing.parent.exists()


def test_release_all_under_closes_readonly_connections(tmp_path):
    profile = tmp_path / "profile"
    profile.mkdir()
    db = profile / "memory_store.db"
    _seed(db)
    ro = MemoryStore(str(db), readonly=True)
    conn = ro._conn
    assert MemoryStore.release_all_under(profile) == 1
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")
