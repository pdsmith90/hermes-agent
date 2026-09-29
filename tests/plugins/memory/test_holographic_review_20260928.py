"""Regression tests for the 2026-09-28 memory-system review changes.

Three measured defects, one test class each — the numbers are from a 30-day
audit of Claude Code transcripts and a 56-probe retrieval eval on a 2,589-fact
snapshot (Hermes facts 3036/3037):

1. Ambient per-turn prefetch injected ACTIVITY daily digests: 18% of every row
   Claude Code saw, and three of the five most-injected facts were digests. A
   digest names every project and tool of its day, so it matches almost any
   prompt. ``prefetch()`` now over-fetches and drops that category before the
   five-row cut; explicit ``fact_store`` search/list are untouched.
2. The FTS candidate pool was a literal ``limit * 3`` and stopped scaling with
   the corpus (BM25 pool recall@24 0.82 → 0.73). It is the module constant
   ``_FTS_POOL_MULT`` now so the eval can A/B it; this pins that search() uses it.
3. Tag vocabulary reached the entity graph — ``verified:2026-09-13``,
   ``distilled``, ``PARTIALLY-CONFIRMED``, ``paper`` among the highest-degree
   entities — so ``about()``/``related_to()`` walked through metadata into
   unrelated facts. ``_is_entity_like`` rejects that vocabulary for NEW writes.
"""
from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("numpy")  # retrieval imports numpy indirectly

from plugins.memory.holographic import (
    HolographicMemoryProvider,
    _PREFETCH_EXCLUDE,
    _PREFETCH_LIMIT,
)
from plugins.memory.holographic import retrieval
from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore, _is_entity_like


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


@pytest.fixture
def provider(tmp_path):
    p = HolographicMemoryProvider(
        config={"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": 64}
    )
    p.initialize(session_id="test-session")
    yield p
    p.shutdown()


def _linked_names(store, fact_id):
    return {
        row[0]
        for row in store._conn.execute(
            "SELECT e.name FROM entities e JOIN fact_entities fe"
            " ON fe.entity_id = e.entity_id WHERE fe.fact_id = ?",
            (fact_id,),
        )
    }


class TestEntityTagTokens:
    @pytest.mark.parametrize(
        "name",
        [
            "verified", "distilled", "PARTIALLY-CONFIRMED", "verified:2026-09-13",
            "written:2026-09-12", "source:claude-code", "audited:2026-09-20",
            "claude-memory-distilled", "answered", "daily", "paper", "promoted",
            "promote-candidate", "needs-experiment", "memory-entry",
            "promotion-declined:2026-09-21", "coverage-checked:2026-09-13",
        ],
    )
    def test_tag_vocabulary_is_not_an_entity(self, name):
        assert not _is_entity_like(name)

    @pytest.mark.parametrize(
        "name",
        ["llama-swap", "LightRAG", "gfx1201", "Hermes", "GRACE-FO", "arXiv:2607.19083",
         "fact_store", "Claude Code", "Gauss-Newton", "R9700"],
    )
    def test_real_names_still_pass(self, name):
        assert _is_entity_like(name)

    def test_tags_path_links_subjects_not_stamps(self, store):
        fid = store.add_fact(
            "LESSON: on the desktop the llama-swap reranker lives on the first card.",
            category="lesson",
            tags="llama-swap,verified:2026-09-28,distilled,source:claude-code,gfx1201",
        )
        names = _linked_names(store, fid)
        assert "llama-swap" in names
        assert "gfx1201" in names
        lowered = {n.lower() for n in names}
        assert "distilled" not in lowered
        assert not any(n.startswith(("verified:", "source:")) for n in lowered)

    def test_verdict_words_in_prose_are_not_entities(self, store):
        fid = store.add_fact(
            "PARTIALLY-CONFIRMED: the Zernike basis suits the LightRAG index; "
            "the Paper we read says so and the Daily digest agreed.",
            category="researched",
        )
        lowered = {n.lower() for n in _linked_names(store, fid)}
        assert "lightrag" in lowered
        assert "partially-confirmed" not in lowered
        assert "paper" not in lowered
        assert "daily" not in lowered


class TestPrefetchExcludesActivity:
    def test_activity_digest_is_filtered_out(self, provider):
        provider._store.add_fact(
            "ACTIVITY 2026-09-27: Claude Code spent the day on ZORBLAT tuning "
            "across three projects and many Bash calls.",
            category="activity",
        )
        provider._store.add_fact(
            "LESSON: ZORBLAT tuning on the desktop needs the ZORBLAT flag set first.",
            category="lesson",
        )
        out = provider.prefetch("how do I do ZORBLAT tuning")
        assert "LESSON: ZORBLAT tuning" in out
        assert "ACTIVITY 2026-09-27" not in out

    def test_window_is_still_five_rows(self, provider):
        for i in range(_PREFETCH_LIMIT + 3):
            provider._store.add_fact(
                f"LESSON: ZORBLAT lesson number {i} about the ZORBLAT flag on the desktop.",
                category="lesson",
            )
        out = provider.prefetch("ZORBLAT flag")
        rows = [ln for ln in out.splitlines() if ln.startswith("- [")]
        assert len(rows) == _PREFETCH_LIMIT

    def test_exclusion_is_only_activity(self):
        assert _PREFETCH_EXCLUDE == frozenset({"activity"})


class TestPoolMultiplierIsUsed:
    def test_search_requests_limit_times_multiplier(self, store, monkeypatch):
        store.add_fact("LESSON: QUUX pool sizing fact for the multiplier test.",
                       category="lesson")
        r = FactRetriever(store, rerank_url="", dense_url="")
        seen = {}
        orig = r._fts_candidates

        def spy(query, category, min_trust, limit):
            seen["limit"] = limit
            return orig(query, category, min_trust, limit)

        monkeypatch.setattr(r, "_fts_candidates", spy)
        r.search("QUUX pool sizing", limit=8)
        assert seen["limit"] == 8 * retrieval._FTS_POOL_MULT
        assert retrieval._FTS_POOL_MULT >= 3
