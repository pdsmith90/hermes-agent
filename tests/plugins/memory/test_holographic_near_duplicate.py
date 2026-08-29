"""Tests for add_fact's near-duplicate guard (2026-08-29).

UNIQUE(content) only catches byte-identical writes, so one changed character
created a competing row: measured over 871 facts, 7 pairs at Jaccard >= 0.55 and
two effectively identical (89/468 at 1.00, 85/878 at 0.98). The copies split
trust and retrieval_count and crowd the limit-5 prefetch window with one
finding.

The dangerous direction is suppressing a genuinely new fact, so the
does-NOT-fire cases are the load-bearing tests here, not the happy path.
"""
from __future__ import annotations

import sqlite3

import pytest

from plugins.memory.holographic.store import (
    _NEAR_DUP_JACCARD,
    _NEAR_DUP_MIN_TOKENS,
    MemoryStore,
)


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


# A realistic fact: 40+ distinct tokens, well over _NEAR_DUP_MIN_TOKENS.
BASE = (
    "LESSON: the document-store reranker went back to the CPU on 2026-08-25 "
    "because ollama kept evicting bge-m3 against qwen2.5:7b eight to thirty-two "
    "times an hour, starving rerank into 240 second timeouts and a silent "
    "unranked fallback that made answers fabricate; only -ngl 0 closes the "
    "0.62 GiB gap and partial offload trims do not."
)


def _count(store):
    return store._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]


class TestSuppressesRestatements:
    def test_reworded_prefix_returns_the_existing_row(self, store):
        first = store.add_fact(BASE, category="lesson")
        again = store.add_fact(
            BASE.replace("LESSON:", "KEY PATTERN/LESSON:"), category="lesson"
        )
        assert again == first
        assert _count(store) == 1

    def test_byte_identical_still_returns_the_existing_row(self, store):
        first = store.add_fact(BASE, category="lesson")
        assert store.add_fact(BASE, category="lesson") == first
        assert _count(store) == 1

    def test_collision_is_logged_at_warning(self, store, caplog):
        first = store.add_fact(BASE, category="lesson")
        with caplog.at_level("WARNING", logger="plugins.memory.holographic.store"):
            store.add_fact(BASE + " Re-confirmed.", category="lesson")
        assert any(
            "near-duplicate write suppressed" in rec.getMessage()
            for rec in caplog.records
        ), "a dropped write must be greppable in agent.log"
        assert _count(store) == 1
        assert first


class TestDoesNotFire:
    """A false positive silently discards a real write. These matter most."""

    def test_a_different_fact_about_the_same_subsystem_is_inserted(self, store):
        first = store.add_fact(BASE, category="lesson")
        other = store.add_fact(
            "LESSON: the GPU reranker DOES fit as of 2026-08-28 — the "
            "blocker was ollama's KV over-reservation, not card capacity. "
            "OLLAMA_NUM_PARALLEL 2->1 plus OLLAMA_EMBEDDING_NUM_CTX 8192->2048 "
            "freed enough headroom for -ub 2048 with zero evictions.",
            category="lesson",
        )
        assert other != first
        assert _count(store) == 2

    def test_cross_category_restatement_is_inserted(self, store):
        """Promotion to memory-entry is how MEMORY.md is rendered — a
        same-content row in another lane is deliberately NOT a duplicate."""
        first = store.add_fact(BASE, category="lesson")
        promoted = store.add_fact(BASE + " Promoted.", category="memory-entry")
        assert promoted != first
        assert _count(store) == 2

    def test_a_fact_older_than_the_window_is_not_compared(self, store):
        first = store.add_fact(BASE, category="lesson")
        store._conn.execute(
            "UPDATE facts SET created_at = datetime('now', '-45 days') "
            "WHERE fact_id = ?",
            (first,),
        )
        store._conn.commit()
        again = store.add_fact(BASE + " Re-confirmed 45 days later.",
                               category="lesson")
        assert again != first
        assert _count(store) == 2

    def test_short_facts_bypass_the_guard(self, store):
        """Below _NEAR_DUP_MIN_TOKENS one word moves Jaccard more than 5%, so
        the guard abstains rather than guess — including on the near-opposite
        pair that the supersession tests write."""
        a = store.add_fact("LESSON: the reranker does not fit on the GPU.")
        b = store.add_fact("LESSON: the GPU reranker does fit.")
        assert len(set(BASE.split())) > _NEAR_DUP_MIN_TOKENS
        assert a != b
        assert _count(store) == 2


class TestComposesWithSupersession:
    def test_a_restated_retraction_does_not_demote_twice(self, store):
        """Re-writing a retraction must never walk its target down again.

        The near-duplicate guard used to supply this by suppressing the second
        write, but it can no longer: suppressing a write that names the fid it
        retracts also skipped the demotion, which lost the correction and left
        the retracted fact at 0.50 (see TestSupersessionIsNeverSuppressed). Both
        restatements are now INSERTED, and the invariant is held where it always
        really lived — the _SUPERSESSION_FLOOR clamp in _demote_superseded.
        """
        old = store.add_fact(
            "LESSON: the document-store reranker was moved to the GPU on 2026-08-20 and "
            "the eviction thrash it caused was never measured at the time, so "
            "the latency regression went unnoticed for five days on that host."
        )
        retraction = (
            "LESSON: the reranker is back on the CPU as of 2026-08-25, which "
            "supersedes fid {old} — ollama was evicting bge-m3 against "
            "qwen2.5:7b up to thirty-two times an hour and starving rerank "
            "into silent unranked fallback on every long query.".format(old=old)
        )
        new = store.add_fact(retraction)
        trust_after_one = store._conn.execute(
            "SELECT trust_score FROM facts WHERE fact_id = ?", (old,)
        ).fetchone()["trust_score"]
        assert trust_after_one == pytest.approx(0.30, abs=1e-6)

        # Same retraction, one word longer: now INSERTED (supersession writes
        # are exempt from the guard), and must still not demote past the floor.
        assert store.add_fact(retraction + " Confirmed.") != new
        trust_after_two = store._conn.execute(
            "SELECT trust_score FROM facts WHERE fact_id = ?", (old,)
        ).fetchone()["trust_score"]
        assert trust_after_two == pytest.approx(trust_after_one, abs=1e-6)
        assert _NEAR_DUP_JACCARD < 1.0


class TestSupersessionIsNeverSuppressed:
    """A correction that names the fid it retracts must always be written.

    Jaccard cannot see polarity: a long fact corrected by one word ("does" ->
    "does NOT") scores ~0.86-0.98 against the original and reads as a
    restatement. Before this exemption the guard dropped the correction AND
    skipped _demote_superseded with it, so the retracted fact kept its 0.50
    trust and went on outranking the answer that overturned it — silently
    undoing the supersession fix this store shipped in da2f371005.
    """

    def _polarity_flip(self, store):
        base = (
            "LESSON: the document-store reranker does fit on the GPU after the ollama KV "
            "fix, measured across twenty runs with the embedding context lowered "
            "and parallelism reduced to one, leaving headroom for both models."
        )
        old = store.add_fact(base, category="lesson")
        correction = (
            base.replace("does fit", "does NOT fit")
            + f" This supersedes fid {old}."
        )
        return old, correction

    def test_near_identical_correction_is_still_inserted(self, store):
        old, correction = self._polarity_flip(store)
        new = store.add_fact(correction, category="lesson")
        assert new != old, "the correction was swallowed as a near-duplicate"

    def test_and_it_still_demotes_its_target(self, store):
        old, correction = self._polarity_flip(store)
        store.add_fact(correction, category="lesson")
        trust = store._conn.execute(
            "SELECT trust_score FROM facts WHERE fact_id = ?", (old,)
        ).fetchone()["trust_score"]
        assert trust == pytest.approx(0.30)

    def test_a_restatement_with_no_fid_is_still_suppressed(self, store):
        """The exemption is keyed on the retraction marker, not on similarity."""
        base = (
            "LESSON: the document-store reranker does fit on the GPU after the ollama KV "
            "fix, measured across twenty runs with the embedding context lowered "
            "and parallelism reduced to one, leaving headroom for both models."
        )
        old = store.add_fact(base, category="lesson")
        assert store.add_fact(base + " Re-confirmed today.", category="lesson") == old
