"""The scheduler states a job's fact writes; the job's prose does not.

On 2026-08-15 consolidate-synthesize reported "4 removals (871, 872, 893,
892)" when two of those were updates, one was refused by the protected-category
guard and one was never touched — while it silently hard-deleted a fifth fact
(879) that its report never mentions. The morning briefing copied the false
numbers forward. Prompting the job to self-verify in prose had already failed
on 2026-08-11 and -08-12.

``_fact_write_ledger`` is read back from the store after the run, so the agent
cannot author it. These tests pin the two properties that make it worth
trusting: it is exact per session, and it never takes a job's report down with
it when the store is unreadable.
"""

import sqlite3

import pytest

from plugins.memory.holographic.store import MemoryStore

SESSION = "cron_testjob_20260815_210000"
OTHER = "cron_otherjob_20260815_220000"


@pytest.fixture(autouse=True)
def _clean_shared_registry():
    MemoryStore._shared.clear()
    yield
    for entry in list(MemoryStore._shared.values()):
        try:
            entry["conn"].close()
        except sqlite3.Error:
            pass
    MemoryStore._shared.clear()


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    """Point HERMES_HOME at a throwaway store and return the bound helper."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    from cron import scheduler

    return scheduler._fact_write_ledger


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path / "memory_store.db")
    yield s
    s.close()


class TestLedgerAccuracy:
    def test_reports_create_update_and_delete(self, ledger, store):
        made = store.add_fact("LESSON: kept", category="lesson",
                              source_session=SESSION)
        edited = store.add_fact("HYPOTHESIS: v1", category="hypothesis",
                                source_session=OTHER)
        doomed = store.add_fact("OPEN QUESTION: prune me",
                                category="open-question", source_session=OTHER)
        store.update_fact(edited, content="HYPOTHESIS: v2", changed_by=SESSION)
        store.remove_fact(doomed, changed_by=SESSION)

        block = ledger(SESSION)
        assert f"**Created (1):** {made} (lesson)" in block
        assert f"**Updated (1):** {edited}" in block
        assert "**Removed (1):**" in block
        assert str(doomed) in block
        # The content head is the recovery pointer — without it the reader
        # cannot tell what was lost without querying fact_history by hand.
        assert "OPEN QUESTION: prune me" in block

    def test_counts_a_fact_it_created_that_was_later_deleted(self, ledger, store):
        # fid 900 on 2026-08-15: added by consolidate at 03:25, deleted by the
        # 05:05 job. Reading `facts` alone drops it from the creator's ledger.
        fid = store.add_fact("OPEN QUESTION: short-lived",
                             category="open-question", source_session=SESSION)
        store.remove_fact(fid, changed_by=OTHER)

        block = ledger(SESSION)
        assert f"**Created (1):** {fid} (open-question, since deleted)" in block
        # It was not this session that removed it.
        assert "**Removed" not in block

    def test_one_job_never_inherits_anothers_writes(self, ledger, store):
        mine = store.add_fact("mine", category="general", source_session=SESSION)
        theirs = store.add_fact("theirs", category="general",
                                source_session=OTHER)
        store.update_fact(theirs, content="theirs v2", changed_by=OTHER)

        block = ledger(SESSION)
        assert str(mine) in block
        assert "**Updated" not in block
        assert f"**Created (1):** {mine} (general)" in block

    def test_silent_run_says_so_explicitly(self, ledger, store):
        store.add_fact("unrelated", category="general", source_session=OTHER)
        assert "No fact-store writes recorded" in ledger(SESSION)

    def test_block_asserts_its_own_authority(self, ledger, store):
        store.add_fact("x", category="general", source_session=SESSION)
        block = ledger(SESSION)
        assert "system-generated" in block
        assert "The agent did not write this section" in block

    def test_block_fits_the_context_from_truncation_window(self, ledger, store):
        # context_from hands a downstream job the FIRST 8000 chars of the
        # report file, and real reports run 14-21K. The ledger is emitted
        # above the prompt for exactly this reason; if it ever moves back
        # below, the morning briefing stops seeing the numbers it is meant
        # to stop repeating.
        store.add_fact("y", category="general", source_session=SESSION)
        report = (
            "# Cron Job: j\n\n**Job ID:** j\n"
            f"{ledger(SESSION)}\n## Prompt\n\n" + ("X" * 15000)
        )
        assert 0 <= report.find("Fact-store ledger") < 8000


class TestLedgerNeverBreaksAJob:
    """A ledger that cannot be built must degrade to nothing, never raise."""

    def test_empty_session_id(self, ledger):
        assert ledger("") == ""

    def test_missing_database(self, ledger):
        # HERMES_HOME points at a tmp dir with no memory_store.db yet.
        assert ledger(SESSION) == ""

    def test_unreadable_database(self, ledger, tmp_path):
        (tmp_path / "memory_store.db").write_text("this is not a database")
        assert ledger(SESSION) == ""

    def test_store_without_history_table(self, ledger, tmp_path):
        # A store predating fact_history must not raise on the history queries.
        conn = sqlite3.connect(tmp_path / "memory_store.db")
        conn.execute(
            "CREATE TABLE facts (fact_id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " content TEXT NOT NULL UNIQUE, category TEXT DEFAULT 'general',"
            " source_session TEXT DEFAULT '')"
        )
        conn.commit()
        conn.close()
        assert ledger(SESSION) == ""

    def test_opens_the_store_read_only(self, ledger, store, tmp_path):
        # The scheduler must never be able to mutate the store while
        # describing it.
        store.add_fact("witness", category="general", source_session=SESSION)
        before = sqlite3.connect(tmp_path / "memory_store.db").execute(
            "SELECT COUNT(*) FROM facts"
        ).fetchone()[0]
        ledger(SESSION)
        after = sqlite3.connect(tmp_path / "memory_store.db").execute(
            "SELECT COUNT(*) FROM facts"
        ).fetchone()[0]
        assert before == after == 1
