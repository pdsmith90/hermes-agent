"""Trust writes land on clean decimals, so the 0.30 search floor means 0.30.

Trust moves by float addition and the decimal steps are not exact in binary:
0.7 + -0.4 is 0.29999999999999993. On 2026-09-27 eleven synthesis facts, each
demoted 0.5 -> 0.7 -> 0.3 by update_fact, sat at that value instead of on the
floor and so failed the `trust_score >= 0.3` predicate of every default search. The exact-equality asserts below are the point —
pytest.approx would pass on the bug.
"""

import sqlite3

import pytest

pytest.importorskip("numpy")  # retrieval module imports numpy indirectly

from plugins.memory.holographic.retrieval import FactRetriever
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


def test_demotion_to_the_floor_lands_on_it_and_stays_searchable(store, monkeypatch):
    # rerank_url="" does not disable the reranker while HERMES_RERANK_URL is
    # exported (retrieval.py reads `rerank_url or env`); a test must never
    # reach a live endpoint.
    monkeypatch.delenv("HERMES_RERANK_URL", raising=False)
    fid = store.add_fact(
        "SYNTHESIS: document ingest as of 2026-09-25 — the orphan rebuild is settled.",
        category="synthesis",
    )
    store.update_fact(fid, trust_delta=0.2)           # the cron's 0.5 -> 0.7
    store.update_fact(fid, trust_delta=-0.4)          # predecessor: 0.7 -> 0.3
    assert _trust(store, fid) == 0.3

    retriever = FactRetriever(store=store, dense_url="")
    hits = retriever.search("document ingest orphan rebuild", min_trust=0.3)
    assert fid in [h["fact_id"] for h in hits]


def test_feedback_steps_land_on_clean_decimals(store):
    fid = store.add_fact("LESSON: the reranker fits on one card.")
    # Down four -0.10 steps, then up eighteen +0.05 steps to the ceiling:
    # unrounded, 14 of its 22 stops are off the grid (0.30000000000000004,
    # 0.49999999999999994, 0.9500000000000003, ...), and the drift compounds.
    expected = 0.5
    for helpful in [False] * 4 + [True] * 18:
        expected = min(1.0, round(expected + (0.05 if helpful else -0.10), 2))
        out = store.record_feedback(fid, helpful=helpful)
        assert out["new_trust"] == expected
        assert _trust(store, fid) == expected


def test_clamping_at_the_bounds_still_holds(store):
    fid = store.add_fact("LESSON: trust is bounded to [0, 1].")
    store.update_fact(fid, trust_delta=5.0)
    assert _trust(store, fid) == 1.0
    assert store.record_feedback(fid, helpful=True)["new_trust"] == 1.0
    store.update_fact(fid, trust_delta=-5.0)
    assert _trust(store, fid) == 0.0
    assert store.record_feedback(fid, helpful=False)["new_trust"] == 0.0
