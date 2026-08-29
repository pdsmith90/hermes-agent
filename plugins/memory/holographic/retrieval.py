"""Hybrid keyword/BM25 retrieval for the memory store.

Ported from KIK memory_agent.py — combines FTS5 full-text search with
Jaccard similarity reranking and trust-weighted scoring.
"""

from __future__ import annotations

import json
import logging
import math
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .store import MemoryStore

try:
    from . import holographic as hrr
except ImportError:
    import holographic as hrr  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


class FactRetriever:
    """Multi-strategy fact retrieval with trust-weighted scoring."""

    def __init__(
        self,
        store: MemoryStore,
        temporal_decay_half_life: int = 0,  # days, 0 = disabled
        fts_weight: float = 0.4,
        jaccard_weight: float = 0.3,
        hrr_weight: float = 0.3,
        hrr_dim: int = 1024,
        rerank_url: str = "",
        rerank_model: str = "qwen3-rerank",
        rerank_timeout: float = 8.0,
        rerank_max_query_chars: int = 1500,
        rerank_max_doc_chars: int = 3000,
    ):
        self.store = store
        self.half_life = temporal_decay_half_life
        self.hrr_dim = hrr_dim

        # Optional cross-encoder rerank of the FTS candidate pool (see search()).
        # DISABLED unless a URL is supplied, either here or via HERMES_RERANK_URL.
        # When disabled, or when the endpoint errors/times out, the additive
        # fts/jaccard/hrr blend below is used unchanged.
        self.rerank_url = rerank_url or os.environ.get("HERMES_RERANK_URL", "")
        self.rerank_model = rerank_model
        self.rerank_timeout = rerank_timeout
        # A reranker server has a hard per-sequence cap (llama.cpp RANK pooling
        # cannot split a sequence across ubatches). The QUERY is the term that
        # actually overruns it: prefetch() passes the caller's whole turn text,
        # which can be thousands of tokens, while stored facts are far smaller.
        # Truncating costs nothing — a cross-encoder judges relevance from the
        # head of the query — and turns a hard failure into a bounded request.
        self.rerank_max_query_chars = rerank_max_query_chars
        self.rerank_max_doc_chars = rerank_max_doc_chars

        # Auto-redistribute weights if numpy unavailable
        if hrr_weight > 0 and not hrr._HAS_NUMPY:
            fts_weight = 0.6
            jaccard_weight = 0.4
            hrr_weight = 0.0

        self.fts_weight = fts_weight
        self.jaccard_weight = jaccard_weight
        self.hrr_weight = hrr_weight

    def search(
        self,
        query: str,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
    ) -> list[dict]:
        """Hybrid search: FTS5 candidates → Jaccard rerank → trust weighting.

        Pipeline:
        1. FTS5 search: Get limit*3 candidates from SQLite full-text search
        2. Jaccard boost: Token overlap between query and fact content
        3. Trust weighting: final_score = relevance * trust_score
        4. Temporal decay (optional): decay = 0.5^(age_days / half_life)
        5. Cross-encoder rerank (optional): score = ce * (0.5 + 0.5*trust) *
           decay — same pool, reordered; the trust term is clamped there
           because the cross-encoder saturates. See stage 3 below.

        Returns list of dicts with fact data + 'score' field, sorted by score desc.
        """
        # Stage 1: Get FTS5 candidates (more than limit for reranking headroom)
        candidates = self._fts_candidates(query, category, min_trust, limit * 3)

        if not candidates:
            return []

        # Stage 2: Rerank with Jaccard + trust + optional decay
        query_tokens = self._tokenize(query)
        # Hoisted: constant across candidates, and encode_text(query) was
        # previously recomputed for every one. Bound to ROLE_CONTENT because a
        # fact's content component is bind(encode_text(content), ROLE_CONTENT);
        # comparing an UNBOUND query against it is quasi-orthogonal by
        # construction, which made this term near-constant noise.
        #
        # DELIBERATE DIVERGENCE from upstream 89f74d58f, which hoists this the
        # same way but leaves the query UNBOUND. The binding is the whole point
        # of this fix — do not take upstream's form on the next sync.
        #
        # Upstream's laziness IS kept: encode on the first candidate that
        # actually carries a vector, so migrated stores (FTS candidates whose
        # hrr_vector was never backfilled) pay nothing. Guarded by
        # test_search_without_vectors_never_encodes.
        query_vec = None
        scored = []

        for fact in candidates:
            content_tokens = self._tokenize(fact["content"])
            tag_tokens = self._tokenize(fact.get("tags", ""))
            all_tokens = content_tokens | tag_tokens

            jaccard = self._jaccard_similarity(query_tokens, all_tokens)
            fts_score = fact.get("fts_rank", 0.0)

            # HRR similarity. The [0,1] shift is KEPT here, unlike probe/
            # related/reason: this term is one of three ADDITIVELY blended
            # signals, so a constant offset cannot invert ranking, and removing
            # it would perturb the tuned fts/jaccard/hrr weights. Do not
            # "unify" this with the max(sim, 0.0) used by the others.
            if self.hrr_weight > 0 and fact.get("hrr_vector"):
                fact_vec = hrr.bytes_to_phases(fact["hrr_vector"], dim=self.hrr_dim)
                if query_vec is None:
                    role_content = hrr.encode_atom(
                        "__hrr_role_content__", self.hrr_dim
                    )
                    query_vec = hrr.bind(
                        hrr.encode_text(query, self.hrr_dim), role_content
                    )
                hrr_sim = (hrr.similarity(query_vec, fact_vec) + 1.0) / 2.0  # shift to [0,1]
            else:
                hrr_sim = 0.5  # neutral

            # Combine FTS5 + Jaccard + HRR
            relevance = (self.fts_weight * fts_score
                        + self.jaccard_weight * jaccard
                        + self.hrr_weight * hrr_sim)

            # Trust weighting
            score = relevance * fact["trust_score"]

            # Optional temporal decay. Stashed on the fact (and popped off with
            # hrr_vector below) because stage 3 recomputes the score from
            # scratch and has to apply the SAME factor: until 2026-08-29 it
            # silently dropped it, which made temporal_decay_half_life dead
            # config whenever the reranker was up — i.e. always, in production.
            # Verified on a copy of the live store: top-5 was byte-identical at
            # half_life 0 and 60 with the reranker reachable.
            decay = self._temporal_decay(
                fact.get("updated_at") or fact.get("created_at")
            )
            fact["_decay"] = decay
            score *= decay

            fact["score"] = score
            scored.append(fact)

        # Sort by score descending, return top limit
        scored.sort(key=lambda x: x["score"], reverse=True)

        # Stage 3 (optional): cross-encoder rerank of the SAME limit*3 pool.
        # Measured against a live store: reranking the existing limit*3 pool
        # changed ~47% of top-5 in ~520 ms, while widening to limit*6 gave the
        # SAME ~47% at ~1190 ms — so the pool is deliberately NOT widened.
        # The additive blend above still decides which limit*3 candidates get
        # here; this only reorders them. Any failure returns None -> blend order.
        if self.rerank_url and len(scored) > 1:
            ce = self._rerank_scores(query, [f["content"] for f in scored])
            if ce is not None:
                for fact, ce_score in zip(scored, ce):
                    # ce_score is already P(yes) in [0,1] (RANK pooling softmaxes
                    # in-graph for QWEN3). Trust weighting is preserved so a
                    # low-trust fact cannot win on relevance alone, and the
                    # decay factor from stage 2 is carried through — dropping it
                    # here is what made half_life inert.
                    #
                    # The trust term is CLAMPED to [0.5, 1.0] on THIS path only
                    # (2026-08-29). Within one topic the cross-encoder is
                    # near-binary: measured 0.9997-0.9988 across seven on-topic
                    # facts, a 0.0009 spread against a 0.20 trust spread, so a
                    # raw 0.30-1.00 multiplier decided the order by itself. It
                    # ranked fid 986 (trust 0.70, text reads "RETIRED", ce
                    # 0.9599) first and fid 1305 (trust 0.50, the current
                    # answer, ce 0.9990) seventh — burying a real 4% relevance
                    # deficit the reranker HAD detected. `0.5 + 0.5*trust` says
                    # trust may at most halve a fact's score: a 0.30 lineage row
                    # still loses ~13% to a 0.50 current one and cannot win back
                    # a 0.1% ce edge, while a stale-but-trusted row no longer
                    # beats a fresher correction on trust alone. The additive
                    # blend above is deliberately NOT clamped — its relevance
                    # term has real dynamic range, so trust cannot swamp it
                    # there, and the tuned fts/jaccard/hrr weights assume the
                    # raw multiplier.
                    fact["score"] = (
                        ce_score
                        * (0.5 + 0.5 * fact["trust_score"])
                        * fact.get("_decay", 1.0)
                    )
                scored.sort(key=lambda x: x["score"], reverse=True)

        results = scored[:limit]
        # Strip raw HRR bytes and the internal decay stash — callers expect
        # JSON-serializable dicts holding only store columns plus "score".
        for fact in results:
            fact.pop("hrr_vector", None)
            fact.pop("_decay", None)
        # Surfacing counts as retrieval: this path serves per-turn prefetch
        # injection and the fact_store search action, neither of which was
        # reflected in retrieval_count before.
        try:
            self.store.mark_retrieved([f["fact_id"] for f in results])
        except Exception:
            logger.debug("mark_retrieved failed", exc_info=True)
        return results

    def _rerank_scores(self, query: str, documents: list[str]) -> list[float] | None:
        """POST to a llama.cpp /v1/rerank endpoint. None on ANY failure.

        Returning None (never raising) is deliberate: search() is on the hot
        path for prefetch() every turn and for every cron fact_store search, so
        a down or slow reranker must degrade to the blend, not break retrieval.
        """
        payload = json.dumps(
            {
                "model": self.rerank_model,
                "query": query[: self.rerank_max_query_chars],
                "documents": [d[: self.rerank_max_doc_chars] for d in documents],
            }
        ).encode()
        try:
            req = urllib.request.Request(
                self.rerank_url, payload, {"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=self.rerank_timeout) as resp:
                data = json.load(resp)
            scores = [0.0] * len(documents)
            for item in data["results"]:
                scores[item["index"]] = float(item["relevance_score"])
            return scores
        except (urllib.error.URLError, OSError, KeyError, ValueError, TypeError):
            return None

    def probe(
        self,
        entity: str,
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Compositional entity query using HRR algebra.

        Tests whether bind(entity, ROLE_ENTITY) is one of the components bundled
        into each fact's vector. This is NOT keyword search — it uses algebraic
        structure to find facts where the entity plays a structural role.

        Falls back to FTS5 search if numpy unavailable.
        """
        if not hrr._HAS_NUMPY:
            # Fallback to keyword search on entity name
            return self.search(entity, category=category, limit=limit)

        conn = self.store._conn

        # Encode entity as role-bound vector
        role_entity = hrr.encode_atom("__hrr_role_entity__", self.hrr_dim)
        entity_vec = hrr.encode_atom(entity.lower(), self.hrr_dim)
        probe_key = hrr.bind(entity_vec, role_entity)

        # Try category-specific bank first, then all facts
        if category:
            bank_name = f"cat:{category}"
            bank_row = conn.execute(
                "SELECT vector FROM memory_banks WHERE bank_name = ?",
                (bank_name,),
            ).fetchone()
            if bank_row:
                # Score against probe_key itself. The previous
                # unbind(bank_vec, probe_key) was the bind-vs-bundle error:
                # encode_fact BUNDLES its components, and unbind inverts bind,
                # not bundle, so the residual was noise. Note this makes the
                # branch deliberately equivalent to the direct scoring below —
                # it is kept so the memory_banks table retains a reader; do not
                # "simplify" it away without also retiring _rebuild_bank.
                # (bank_row is intentionally no longer read here; the row's
                # existence is what selects this branch.)
                return self._score_facts_by_vector(
                    probe_key, category=category, limit=limit
                )

        # Score against individual fact vectors directly
        where = "WHERE hrr_vector IS NOT NULL"
        params: list = []
        if category:
            where += " AND category = ?"
            params.append(category)

        rows = conn.execute(
            f"""
            SELECT fact_id, content, category, tags, trust_score,
                   retrieval_count, helpful_count, created_at, updated_at,
                   hrr_vector
            FROM facts
            {where}
            """,
            params,
        ).fetchall()

        if not rows:
            # Final fallback: keyword search
            return self.search(entity, category=category, limit=limit)

        # NOTE: upstream 89f74d58f hoists a loop-invariant role_content atom
        # here. This fix removes the only consumer (the content_vec compare
        # below), so the hoist is dropped with it rather than left dead.
        scored = []
        for row in rows:
            fact = dict(row)
            fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"), dim=self.hrr_dim)
            # Bundle-membership test: high iff bind(entity, ROLE_ENTITY) is one
            # of the components bundled into this fact vector. (Also drops a
            # full encode_text() of every fact body, per probe call.)
            sim = hrr.similarity(fact_vec, probe_key)
            # max(...,0), not (sim+1)/2: the shift floors an unrelated fact at
            # ~0.5, which trust then multiplies, so a trust-1.0 fact with no
            # match outranked a trust-0.5 fact with a strong one.
            fact["score"] = max(sim, 0.0) * fact["trust_score"]
            scored.append(fact)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def related(
        self,
        entity: str,
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Discover facts that share structural connections with an entity.

        Unlike probe (which finds facts *about* an entity), related finds
        facts that are connected through shared context — e.g., other entities
        mentioned alongside this one, or content that overlaps structurally.

        Falls back to FTS5 search if numpy unavailable.
        """
        if not hrr._HAS_NUMPY:
            return self.search(entity, category=category, limit=limit)

        conn = self.store._conn

        # Two ways the entity can be structurally present, both hoisted out of
        # the per-fact loop (the role atoms were previously recomputed per row).
        # Testing the content role works because encode_text bundles its token
        # atoms before binding to ROLE_CONTENT, and phase addition preserves
        # similarity — so a single token bound to ROLE_CONTENT stays similar to
        # the whole bound bag-of-words.
        entity_vec = hrr.encode_atom(entity.lower(), self.hrr_dim)
        role_entity = hrr.encode_atom("__hrr_role_entity__", self.hrr_dim)
        role_content = hrr.encode_atom("__hrr_role_content__", self.hrr_dim)
        as_entity = hrr.bind(entity_vec, role_entity)
        as_content = hrr.bind(entity_vec, role_content)

        # Get all facts with vectors
        where = "WHERE hrr_vector IS NOT NULL"
        params: list = []
        if category:
            where += " AND category = ?"
            params.append(category)

        rows = conn.execute(
            f"""
            SELECT fact_id, content, category, tags, trust_score,
                   retrieval_count, helpful_count, created_at, updated_at,
                   hrr_vector
            FROM facts
            {where}
            """,
            params,
        ).fetchall()

        if not rows:
            return self.search(entity, category=category, limit=limit)

        # Score each fact by how much the entity's atom appears in its vector.
        # This catches both role-bound entity matches AND content word matches.
        # The role atoms upstream 89f74d58f hoists here are already hoisted
        # above, together with the as_entity/as_content bind keys this loop
        # actually compares against — re-encoding them here would be dead.
        scored = []
        for row in rows:
            fact = dict(row)
            fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"), dim=self.hrr_dim)

            # Take the max — entity could appear in either role. This algebra
            # was already correct in its unbind form: similarity(unbind(m,k), r)
            # is exactly similarity(m, bind(k,r)), since both reduce to
            # mean(cos(m-k-r)). Written as a bundle-membership test for clarity.
            best_sim = max(
                hrr.similarity(fact_vec, as_entity),
                hrr.similarity(fact_vec, as_content),
            )

            # See probe(): the [0,1] shift let trust swamp the signal.
            fact["score"] = max(best_sim, 0.0) * fact["trust_score"]
            scored.append(fact)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def reason(
        self,
        entities: list[str],
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Multi-entity compositional query — vector-space JOIN.

        Given multiple entities, algebraically intersects their structural
        connections to find facts related to ALL of them simultaneously.
        This is compositional reasoning that no embedding DB can do.

        Example: reason(["peppi", "backend"]) finds facts where peppi AND
        backend both play structural roles — without keyword matching.

        Falls back to FTS5 search if numpy unavailable.
        """
        if not hrr._HAS_NUMPY or not entities:
            # Fallback: search with all entities as keywords
            query = " ".join(entities)
            return self.search(query, category=category, limit=limit)

        conn = self.store._conn
        role_entity = hrr.encode_atom("__hrr_role_entity__", self.hrr_dim)

        # One bundle-component key per entity. (Previously named
        # "entity_residuals" — they were never residuals, they are bind keys.)
        probe_keys = [
            hrr.bind(hrr.encode_atom(entity.lower(), self.hrr_dim), role_entity)
            for entity in entities
        ]

        # Get all facts with vectors
        where = "WHERE hrr_vector IS NOT NULL"
        params: list = []
        if category:
            where += " AND category = ?"
            params.append(category)

        rows = conn.execute(
            f"""
            SELECT fact_id, content, category, tags, trust_score,
                   retrieval_count, helpful_count, created_at, updated_at,
                   hrr_vector
            FROM facts
            {where}
            """,
            params,
        ).fetchall()

        if not rows:
            query = " ".join(entities)
            return self.search(query, category=category, limit=limit)

        # Score each fact by how much EACH entity is structurally present.
        # A fact scores high only if ALL entities have structural presence
        # (AND semantics via min, vs OR which would use mean/max).
        scored = []
        for row in rows:
            fact = dict(row)
            fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"), dim=self.hrr_dim)

            min_sim = min(hrr.similarity(fact_vec, key) for key in probe_keys)
            # See probe(): the [0,1] shift let trust swamp the signal.
            fact["score"] = max(min_sim, 0.0) * fact["trust_score"]
            scored.append(fact)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def contradict(
        self,
        category: str | None = None,
        threshold: float = 0.3,
        limit: int = 10,
    ) -> list[dict]:
        """Find potentially contradictory facts via entity overlap + content divergence.

        Two facts contradict when they share entities (same subject) but have
        low content-vector similarity (different claims). This is automated
        memory hygiene — no other memory system does this.

        Returns pairs of facts with a contradiction score.
        Falls back to empty list if numpy unavailable.
        """
        if not hrr._HAS_NUMPY:
            return []

        conn = self.store._conn

        # Get all facts with vectors and their linked entities
        where = "WHERE f.hrr_vector IS NOT NULL"
        params: list = []
        if category:
            where += " AND f.category = ?"
            params.append(category)

        rows = conn.execute(
            f"""
            SELECT f.fact_id, f.content, f.category, f.tags, f.trust_score,
                   f.created_at, f.updated_at, f.hrr_vector
            FROM facts f
            {where}
            """,
            params,
        ).fetchall()

        if len(rows) < 2:
            return []

        # Guard against O(n²) explosion on large fact stores.
        # At 500 facts, that's ~125K comparisons — acceptable.
        # Above that, only check the most recently updated facts.
        _MAX_CONTRADICT_FACTS = 500
        if len(rows) > _MAX_CONTRADICT_FACTS:
            rows = sorted(rows, key=lambda r: r["updated_at"] or r["created_at"], reverse=True)
            rows = rows[:_MAX_CONTRADICT_FACTS]

        # Build entity sets per fact
        fact_entities: dict[int, set[str]] = {}
        for row in rows:
            fid = row["fact_id"]
            entity_rows = conn.execute(
                """
                SELECT e.name FROM entities e
                JOIN fact_entities fe ON fe.entity_id = e.entity_id
                WHERE fe.fact_id = ?
                """,
                (fid,),
            ).fetchall()
            fact_entities[fid] = {r["name"].lower() for r in entity_rows}

        # Compare all pairs: high entity overlap + low content similarity = contradiction
        facts = [dict(r) for r in rows]
        contradictions = []

        for i in range(len(facts)):
            for j in range(i + 1, len(facts)):
                f1, f2 = facts[i], facts[j]
                ents1 = fact_entities.get(f1["fact_id"], set())
                ents2 = fact_entities.get(f2["fact_id"], set())

                if not ents1 or not ents2:
                    continue

                # Entity overlap (Jaccard)
                entity_overlap = len(ents1 & ents2) / len(ents1 | ents2) if (ents1 | ents2) else 0.0

                if entity_overlap < 0.3:
                    continue  # Not enough entity overlap to be contradictory

                # Content similarity via HRR vectors
                v1 = hrr.bytes_to_phases(f1["hrr_vector"], dim=self.hrr_dim)
                v2 = hrr.bytes_to_phases(f2["hrr_vector"], dim=self.hrr_dim)
                content_sim = hrr.similarity(v1, v2)

                # High entity overlap + low content similarity = potential contradiction
                # contradiction_score: higher = more contradictory
                contradiction_score = entity_overlap * (1.0 - (content_sim + 1.0) / 2.0)

                if contradiction_score >= threshold:
                    # Strip hrr_vector from output (not JSON serializable)
                    f1_clean = {k: v for k, v in f1.items() if k != "hrr_vector"}
                    f2_clean = {k: v for k, v in f2.items() if k != "hrr_vector"}
                    contradictions.append({
                        "fact_a": f1_clean,
                        "fact_b": f2_clean,
                        "entity_overlap": round(entity_overlap, 3),
                        "content_similarity": round(content_sim, 3),
                        "contradiction_score": round(contradiction_score, 3),
                        "shared_entities": sorted(ents1 & ents2),
                    })

        contradictions.sort(key=lambda x: x["contradiction_score"], reverse=True)
        return contradictions[:limit]

    def _score_facts_by_vector(
        self,
        target_vec: "np.ndarray",
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Score facts by similarity to a target vector.

        Called only by probe(), and must use the same scoring scale it does —
        max(sim, 0.0) * trust, not the [0,1] shift. Mixing the two would make
        probe() return incompatible scores depending on whether a category bank
        row happened to exist.
        """
        conn = self.store._conn

        where = "WHERE hrr_vector IS NOT NULL"
        params: list = []
        if category:
            where += " AND category = ?"
            params.append(category)

        rows = conn.execute(
            f"""
            SELECT fact_id, content, category, tags, trust_score,
                   retrieval_count, helpful_count, created_at, updated_at,
                   hrr_vector
            FROM facts
            {where}
            """,
            params,
        ).fetchall()

        scored = []
        for row in rows:
            fact = dict(row)
            fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"), dim=self.hrr_dim)
            sim = hrr.similarity(target_vec, fact_vec)
            fact["score"] = max(sim, 0.0) * fact["trust_score"]
            scored.append(fact)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def _fts_candidates(
        self,
        query: str,
        category: str | None,
        min_trust: float,
        limit: int,
    ) -> list[dict]:
        """Get raw FTS5 candidates from the store.

        Uses the store's database connection directly for FTS5 MATCH
        with rank scoring. Normalizes FTS5 rank to [0, 1] range.
        """
        conn = self.store._conn

        # Build query - FTS5 rank is negative (lower = better match)
        # We need to join facts_fts with facts to get all columns
        params: list = []
        where_clauses = ["facts_fts MATCH ?"]
        # FTS5 defaults to AND-between-tokens, which kills recall on
        # natural-language queries ("what happened with the deployment
        # rollback"). Sanitize: drop stopwords, OR-join content tokens, so
        # any significant term can match.
        params.append(self._sanitize_fts_query(query))

        if category:
            where_clauses.append("f.category = ?")
            params.append(category)

        where_clauses.append("f.trust_score >= ?")
        params.append(min_trust)

        where_sql = " AND ".join(where_clauses)

        sql = f"""
            SELECT f.*, facts_fts.rank as fts_rank_raw
            FROM facts_fts
            JOIN facts f ON f.fact_id = facts_fts.rowid
            WHERE {where_sql}
            ORDER BY facts_fts.rank
            LIMIT ?
        """
        params.append(limit)

        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception:
            # FTS5 MATCH can fail on malformed queries — fall back to empty
            return []

        if not rows:
            return []

        # Normalize FTS5 rank: rank is negative, lower = better
        # Convert to positive score in [0, 1] range
        raw_ranks = [abs(row["fts_rank_raw"]) for row in rows]
        max_rank = max(raw_ranks) if raw_ranks else 1.0
        max_rank = max(max_rank, 1e-6)  # avoid div by zero

        results = []
        for row, raw_rank in zip(rows, raw_ranks):
            fact = dict(row)
            fact.pop("fts_rank_raw", None)
            fact["fts_rank"] = raw_rank / max_rank  # normalize to [0, 1]
            results.append(fact)

        return results

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Simple whitespace tokenization with lowercasing.

        Strips common punctuation. No stemming/lemmatization (Phase 1).
        """
        if not text:
            return set()
        # Split on whitespace, lowercase, strip punctuation
        tokens = set()
        for word in text.lower().split():
            cleaned = word.strip(".,;:!?\"'()[]{}#@<>")
            if cleaned:
                tokens.add(cleaned)
        return tokens

    # Stopwords dropped before FTS5 OR-expansion. Short English function
    # words that carry no retrieval signal and force false-negative AND
    # matches when left in the query.
    _FTS_STOPWORDS = frozenset({
        "a", "about", "above", "after", "again", "all", "am", "an", "and",
        "any", "are", "as", "at", "be", "because", "been", "before", "being",
        "between", "both", "but", "by", "can", "could", "did", "do", "does",
        "doing", "don", "down", "during", "each", "few", "for", "from",
        "further", "had", "has", "have", "having", "he", "her", "here",
        "hers", "herself", "him", "himself", "his", "how", "i", "if", "in",
        "into", "is", "it", "its", "itself", "just", "me", "more", "most",
        "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
        "only", "or", "other", "our", "ours", "ourselves", "out", "over",
        "own", "same", "she", "should", "so", "some", "such", "than", "that",
        "the", "their", "theirs", "them", "themselves", "then", "there",
        "these", "they", "this", "those", "through", "to", "too", "under",
        "until", "up", "very", "was", "we", "were", "what", "when", "where",
        "which", "while", "who", "whom", "why", "will", "with", "would",
        "you", "your", "yours", "yourself", "yourselves",
    })

    @classmethod
    def _sanitize_fts_query(cls, query: str) -> str:
        """Convert a natural-language query to an FTS5-safe OR expression.

        FTS5 treats a multi-word MATCH argument as AND-joined by default,
        which tanks recall on prose queries. This helper:
          - tokenizes the query
          - drops stopwords and short (<2 char) tokens
          - strips FTS5 special characters from each token
          - OR-joins the survivors

        If nothing remains (pathological query), falls back to the raw
        query so the caller sees zero results instead of a SQL error.
        """
        if not query:
            return ""
        # Strip FTS5 operator characters from EACH token to avoid
        # accidentally creating a malformed query.
        #
        # '-' is TRANSLATED TO A SPACE, not deleted. The unicode61 tokenizer
        # splits on it, so a deleted hyphen welds two indexed tokens into one
        # that cannot exist in the index: "llama-swap" -> "llamaswap" -> zero
        # rows, silently (tokens are OR-joined, so the query drops its highest
        # -IDF term instead of erroring). 93% of this store's facts contain a
        # hyphenated identifier, so the effect was corpus-wide: "guo-grace" 0
        # hits against 83 linked facts, "grace-fo" 0 against 33. Mapping to a
        # space keeps the surrounding phrase quotes, so the token pair is
        # matched as an adjacent phrase: '"guo grace"' -> 88 facts.
        _FTS_SPECIAL = '"()*^:+'
        tokens: list[str] = []
        for raw in query.lower().split():
            cleaned = raw.strip(".,;:!?\"'()[]{}#@<>") .translate(
                str.maketrans("-", " ", _FTS_SPECIAL)
            ).strip()
            if len(cleaned) < 2:
                continue
            if cleaned in cls._FTS_STOPWORDS:
                continue
            # FTS5 phrase-literal each token to ensure no special chars
            # sneak through as operators.
            tokens.append(f'"{cleaned}"')
        if not tokens:
            # Fallback: raw query (likely returns 0, but never crashes)
            return query
        return " OR ".join(tokens)

    @staticmethod
    def _jaccard_similarity(set_a: set, set_b: set) -> float:
        """Jaccard similarity coefficient: |A ∩ B| / |A ∪ B|."""
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def _temporal_decay(self, timestamp_str: str | None) -> float:
        """Exponential decay: 0.5^(age_days / half_life_days).

        Returns 1.0 if decay is disabled or timestamp is missing.
        """
        if not self.half_life or not timestamp_str:
            return 1.0

        try:
            if isinstance(timestamp_str, str):
                # Parse ISO format timestamp from SQLite
                ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            else:
                ts = timestamp_str

            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400
            if age_days < 0:
                return 1.0

            return math.pow(0.5, age_days / self.half_life)
        except (ValueError, TypeError):
            return 1.0
