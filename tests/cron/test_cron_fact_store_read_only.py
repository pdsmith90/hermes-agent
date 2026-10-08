"""Cron jobs declared read-only on the fact store cannot write to it (2026-10-08, cron review #153).

The morning-briefing prompt says READ-ONLY (action=list and action=get only), and on 2026-10-08 the
model called fact_store action=add anyway. A job whose jobs.json record carries
``fact_store_read_only: true`` now runs with gateway.session_context.CRON_FACT_STORE_READ_ONLY bound
by _CronRunScope, and the holographic provider refuses fact_store add/update/remove and
fact_feedback under it. Reads are untouched, and a job without the flag writes exactly as before.
"""

import json

import pytest

from cron.scheduler import _CronRunScope
from gateway.session_context import CRON_FACT_STORE_READ_ONLY
from plugins.memory.holographic import HolographicMemoryProvider
from tools.daemon_pool import DaemonThreadPoolExecutor


@pytest.fixture
def provider(tmp_path):
    p = HolographicMemoryProvider(
        config={"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": 64}
    )
    p.initialize(session_id="cron_testjob_20261008_073013")
    yield p
    p.shutdown()


def _call(provider, tool, **args):
    return json.loads(provider.handle_tool_call(tool, args))


def _scope(read_only):
    job = {"id": "testjob", "name": "briefing"}
    if read_only:
        job["fact_store_read_only"] = True
    return _CronRunScope(job, "testjob", "exec1")


def _in_tool_worker(fn):
    # A batch of tool calls runs on this pool; it must carry the run's ContextVars.
    pool = DaemonThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(fn).result(timeout=30)
    finally:
        pool.shutdown(wait=True)


def _seed(provider):
    return _call(provider, "fact_store", action="add",
                 content="LESSON: seeded before the run", category="lesson")["fact_id"]


def _snapshot(provider, fid):
    fact = _call(provider, "fact_store", action="get", fact_id=fid)["fact"]
    return {k: fact.get(k) for k in ("content", "category", "tags", "trust_score", "helpful_count")}


class TestReadOnlyJob:
    def test_every_write_is_refused_and_nothing_changes(self, provider):
        kept = _seed(provider)
        before = _snapshot(provider, kept)
        scope = _scope(read_only=True)
        scope.enter()
        try:
            results = _in_tool_worker(lambda: [
                _call(provider, "fact_store", action="add",
                      content="LESSON: the job's own lesson", category="lesson"),
                # the tail a local model leaks into a call must not slip a write past the guard
                _call(provider, "fact_store",
                      action="add>\n</function>\n</tool_call>", content="LESSON: tail", category="lesson"),
                _call(provider, "fact_store", action="update", fact_id=kept, trust_delta=0.1),
                _call(provider, "fact_store", action="remove", fact_id=kept),
                _call(provider, "fact_feedback", action="helpful", fact_id=kept),
            ])
        finally:
            scope.exit()
        for r in results:
            assert "read-only on the fact store" in r["error"], r
        facts = _call(provider, "fact_store", action="list", limit=50)["facts"]
        assert [f["fact_id"] for f in facts] == [kept]
        assert _snapshot(provider, kept) == before

    def test_reads_still_work(self, provider):
        kept = _seed(provider)
        scope = _scope(read_only=True)
        scope.enter()
        try:
            got, listed, found = _in_tool_worker(lambda: (
                _call(provider, "fact_store", action="get", fact_id=kept),
                _call(provider, "fact_store", action="list", category="lesson"),
                _call(provider, "fact_store", action="search", query="seeded run"),
            ))
        finally:
            scope.exit()
        assert got["found"] is True
        assert listed["count"] == 1
        assert "error" not in found

    def test_exit_lifts_the_refusal(self, provider):
        scope = _scope(read_only=True)
        scope.enter()
        assert CRON_FACT_STORE_READ_ONLY.get() is True
        scope.exit()
        assert CRON_FACT_STORE_READ_ONLY.get() is False
        assert "fact_id" in _call(provider, "fact_store", action="add",
                                  content="LESSON: after the run", category="lesson")


class TestOrdinaryJob:
    def test_a_job_without_the_flag_writes_as_before(self, provider):
        scope = _scope(read_only=False)
        scope.enter()
        try:
            added = _in_tool_worker(lambda: _call(
                provider, "fact_store", action="add", content="LESSON: a writer job's lesson", category="lesson"))
        finally:
            scope.exit()
        assert added["status"] == "added"
