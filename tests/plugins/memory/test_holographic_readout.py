"""Regression tests for the HRR readout defects in FactRetriever.

Two defects, both silent — they degraded ranking rather than raising, so the
rest of the suite passed while probe/reason/search returned near-random order:

1. WRONG COMPARISON TARGET. ``encode_fact`` BUNDLES its components, so a
   component is read out by ``similarity(fact, bind(x, ROLE))``. probe() and
   reason() instead compared the unbind residual against the content component /
   ROLE_CONTENT, testing a three-way composite that is not a bundle component at
   all; search() compared an unbound query vector against bound content.

   Note ``unbind`` itself was never the bug: similarity(unbind(m,k), r) is
   exactly similarity(m, bind(k,r)). related() was therefore already correct.

2. THE [0,1] SHIFT. Scoring was ``(sim + 1) / 2 * trust``. The shift floors an
   unrelated fact at ~0.5, which trust then multiplies, so a high-trust fact
   with no match outranked a lower-trust fact with a strong one. search() keeps
   the shift on purpose — there the HRR term is one of three ADDITIVELY blended
   signals, where a constant offset cannot invert ranking.
"""

import pytest

from plugins.memory.holographic import holographic as hrr
from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore

pytestmark = pytest.mark.skipif(
    not hrr._HAS_NUMPY, reason="numpy unavailable"
)

DIM = 1024


class TestPhaseAlgebraAssumptions:
    """The identities the fix relies on. If these break, the fix is invalid."""

    def test_phase_addition_preserves_similarity(self):
        a = hrr.encode_atom("alpha", DIM)
        b = hrr.encode_atom("beta", DIM)
        c = hrr.encode_atom("gamma", DIM)
        assert hrr.similarity(hrr.bind(a, c), hrr.bind(b, c)) == pytest.approx(
            hrr.similarity(a, b), abs=1e-12
        )

    def test_unbind_similarity_equals_bind_similarity(self):
        # Why related() was already correct, and why 'unbind' was never the bug.
        m = hrr.encode_atom("memory", DIM)
        k = hrr.encode_atom("key", DIM)
        r = hrr.encode_atom("role", DIM)
        assert hrr.similarity(hrr.unbind(m, k), r) == pytest.approx(
            hrr.similarity(m, hrr.bind(k, r)), abs=1e-12
        )

    def test_bundle_is_similar_to_its_components(self):
        role_e = hrr.encode_atom("__hrr_role_entity__", DIM)
        present = hrr.bind(hrr.encode_atom("grace-fo", DIM), role_e)
        absent = hrr.bind(hrr.encode_atom("nothing-like-it", DIM), role_e)
        other = hrr.bind(hrr.encode_atom("sgp4", DIM), role_e)
        fact = hrr.bundle(present, other)
        assert hrr.similarity(fact, present) > 0.3
        assert abs(hrr.similarity(fact, absent)) < 0.1


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path / "memory_store.db")
    yield s
    s.close()


@pytest.fixture
def populated(store):
    """Three facts about GRACE-FO, three unrelated — with the DECOYS carrying
    maximum trust, which is what defeated the old (sim+1)/2*trust scoring."""
    for content in [
        'The "GRACE-FO" mission measures gravity.',
        'Reprocessing "GRACE-FO" data for mascons.',
        '"GRACE-FO" and "SGP4" both feed the pipeline.',
    ]:
        fid = store.add_fact(content)
        store._conn.execute(
            "UPDATE facts SET trust_score = 0.5 WHERE fact_id = ?", (fid,)
        )
    for content in [
        "Ocean tides in icy moons are dissipative.",
        "The kanban board tracks open work.",
        "Coffee consumption peaked on Tuesday.",
    ]:
        fid = store.add_fact(content)
        store._conn.execute(
            "UPDATE facts SET trust_score = 1.0 WHERE fact_id = ?", (fid,)
        )
    store._conn.commit()
    for fid, content in store._conn.execute("SELECT fact_id, content FROM facts"):
        store._compute_hrr_vector(fid, content)
    return store


class TestProbeFindsTheRightFacts:
    def test_probe_ranks_entity_facts_above_high_trust_decoys(self, populated):
        results = FactRetriever(populated).probe("GRACE-FO", limit=3)
        assert len(results) == 3
        for fact in results:
            assert "GRACE-FO" in fact["content"], (
                f"decoy outranked a real hit: {fact['content']!r} "
                f"(score={fact['score']:.4f})"
            )

    def test_unrelated_fact_cannot_be_rescued_by_trust(self, populated):
        """The defect-2 regression: an unrelated trust-1.0 fact must not
        outscore a matching trust-0.5 fact."""
        results = FactRetriever(populated).probe("GRACE-FO", limit=10)
        by_content = {f["content"]: f["score"] for f in results}
        best_decoy = max(s for c, s in by_content.items() if "GRACE-FO" not in c)
        worst_hit = min(s for c, s in by_content.items() if "GRACE-FO" in c)
        assert worst_hit > best_decoy

    def test_absent_entity_scores_near_zero(self, populated):
        results = FactRetriever(populated).probe("no-such-entity", limit=10)
        assert max(f["score"] for f in results) < 0.1


class TestReasonIntersects:
    def test_reason_requires_all_entities(self, populated):
        results = FactRetriever(populated).reason(["GRACE-FO", "SGP4"], limit=1)
        assert results
        assert "GRACE-FO" in results[0]["content"]
        assert "SGP4" in results[0]["content"]

    def test_reason_uses_and_not_or(self, populated):
        """The fact with BOTH entities must rank above facts with only one.

        Asserted against the both-fact's own rank, not against results[0] —
        comparing the rest of the list to results[0] is a tautology, since
        results[0] is the maximum by construction.
        """
        results = FactRetriever(populated).reason(["GRACE-FO", "SGP4"], limit=10)
        both = [
            f for f in results
            if "GRACE-FO" in f["content"] and "SGP4" in f["content"]
        ]
        assert both, "the fact containing both entities was not returned"
        only_one = [
            f for f in results
            if ("GRACE-FO" in f["content"]) ^ ("SGP4" in f["content"])
        ]
        assert only_one, "fixture no longer has a single-entity fact to compare"
        assert min(f["score"] for f in both) > max(f["score"] for f in only_one)


class TestSearchComparesLikeWithLike:
    """search()'s HRR term only re-ranks WITHIN the FTS5 candidate set, and
    carries weight 0.3 against fts+jaccard's 0.7 — so these pin the MECHANISM
    (which vector the query is compared as), not a ranking outcome that FTS
    would mask either way.
    """

    def test_query_is_compared_bound_to_role_content(self, populated):
        # Isolate the HRR term: score == (sim + 1) / 2 * trust, so sim is
        # recoverable exactly and can be checked against both candidate forms.
        retriever = FactRetriever(
            populated, fts_weight=0.0, jaccard_weight=0.0, hrr_weight=1.0
        )
        dim = retriever.hrr_dim
        role_content = hrr.encode_atom("__hrr_role_content__", dim)

        query = 'The "GRACE-FO" mission measures gravity.'
        results = retriever.search(query, limit=10)
        assert results, "FTS returned no candidates for the fixture query"

        hit = next(f for f in results if f["content"] == query)
        row = populated._conn.execute(
            "SELECT hrr_vector FROM facts WHERE content = ?", (query,)
        ).fetchone()
        fact_vec = hrr.bytes_to_phases(row["hrr_vector"])

        observed = 2.0 * (hit["score"] / hit["trust_score"]) - 1.0
        bound = hrr.similarity(
            hrr.bind(hrr.encode_text(query, dim), role_content), fact_vec
        )
        unbound = hrr.similarity(hrr.encode_text(query, dim), fact_vec)

        assert observed == pytest.approx(bound, abs=1e-9), (
            "search() is not binding the query to ROLE_CONTENT"
        )
        assert observed != pytest.approx(unbound, abs=1e-6)

    def test_bound_comparison_is_the_one_that_discriminates(self, populated):
        """The bound form must rank a fact first against its own text; the
        unbound form (the bug) must not."""
        role_content = hrr.encode_atom("__hrr_role_content__", DIM)
        rows = populated._conn.execute(
            "SELECT content, hrr_vector FROM facts WHERE hrr_vector IS NOT NULL"
        ).fetchall()
        vectors = [(r["content"], hrr.bytes_to_phases(r["hrr_vector"])) for r in rows]

        bound_hits = unbound_hits = 0
        for content, _ in vectors:
            qv = hrr.encode_text(content, DIM)
            for vec, which in ((hrr.bind(qv, role_content), "b"), (qv, "u")):
                top = max(vectors, key=lambda cv: hrr.similarity(vec, cv[1]))[0]
                if top == content:
                    if which == "b":
                        bound_hits += 1
                    else:
                        unbound_hits += 1

        assert bound_hits == len(vectors), "bound form failed self-retrieval"
        assert unbound_hits < len(vectors), (
            "unbound form also self-retrieves perfectly — fixture too small "
            "to demonstrate the defect"
        )
