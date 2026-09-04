"""Hybrid keyword/BM25 retrieval for the memory store.

Ported from KIK memory_agent.py — combines FTS5 full-text search with
Jaccard similarity reranking and trust-weighted scoring.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .store import MemoryStore

try:
    from . import holographic as hrr
    from . import embeddings
    from . import entailment
except ImportError:
    import holographic as hrr  # type: ignore[no-redef]
    import embeddings  # type: ignore[no-redef]
    import entailment  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

# --- query-adaptive dense weighting --------------------------------------
# The 2026-09-02 eval's finding was that the FTS candidate pool has recall 1.00
# on entity/lexical/supersession probes and 0.36 on paraphrase ones. So the
# dense lane must widen the pool on paraphrase-shaped questions WITHOUT feeding
# the cross-encoder two dozen extra distractors on the questions keyword search
# already answers perfectly — that is how you fix paraphrase and lose entity in
# the same commit. fid 1385 (ANSWERED) says the same thing: union the two
# candidate lists, and weight them ADAPTIVELY rather than at one static alpha.
#
# The routing is by query SHAPE, which is the cheapest signal that actually
# separates the two regimes and needs no model: a question carrying a quoted
# string, a path, an identifier, an ALL-CAPS acronym or a fid reference is one
# whose answer shares literal tokens with the stored fact, and BM25 is already
# at its ceiling there. Plain prose is where keyword search has nothing to
# match on.
#
# TUNED 2026-09-04 against the 46-probe set on a backfilled snapshot of the
# live store, comparing production-with-dense against production-without-dense
# on the SAME corpus (the 09-02 baseline is not a valid control here — the store
# grew 939 -> 1011 facts in between, and those 72 rows are distractors of their
# own). k_lexical swept 8/4/2/0:
#
#   k_lex   entity h@5   lexical h@5   paraphrase h@5
#     8        0.71         0.77           0.64
#     4        0.71         0.85           0.64      <- the knee
#     2        0.71         0.85           0.64
#     0        0.79         0.85           0.57
#   (no dense: 0.79         0.85           0.29)
#
# 8 was over-injecting: two extra dense rows displaced the gold answer of a
# lexical probe out of the top 5 for nothing. 4 restores lexical to its
# no-dense value exactly while keeping every paraphrase gain, and measures
# identically to 2 while leaving more headroom for a prose question the shape
# router sends down the lexical path anyway.
#
# k_lexical=0 — dense candidates ONLY for prose queries — is the one setting
# that also restores entity h@5 to 0.79. It is not the default because that
# 0.79 -> 0.71 is a SINGLE probe (p09) whose gold moved from rank 5 to rank 6
# when the cross-encoder judged a dense candidate more relevant than four of
# the five keyword hits; entity hit@8 is 0.79 at every k, so the fact is still
# retrieved either way. Paying 0.07 of paraphrase recall to move one already-
# retrieved fact up one rank is the wrong trade. Set k_lexical=0 if you want
# the stricter no-regression guarantee.
#
# NOTE ON _DENSE_SHARE: it has NO effect while the cross-encoder is up, because
# stage 3 rescores the entire pool and the blend only decides the pre-rerank
# order. It matters solely on the reranker-down fallback path. Do not try to
# tune retrieval quality with it — tune k.
_DENSE_SHARE = {"lexical": 0.10, "semantic": 0.50}
_DENSE_K = {"lexical": 4, "semantic": 24}

# Reciprocal-rank-fusion constant for stage 3 (2026-09-05).
#
# Until now the cross-encoder REPLACED the blend ordering outright, throwing
# away BM25 + Jaccard + HRR + dense. Measured over the 56-probe set with the
# pools and ce scores captured from one live pass, that cost more than it
# bought: entity hit@5 0.79 against 0.93 for the same pool unreranked, and
# overall hit@1 0.41 against 0.54. The cross-encoder is very good at "is this
# passage about this question" and blind to which token was the POINT of a
# literal query, which is exactly what BM25 knows.
#
# Fusing the two rankings instead of replacing one with the other beats
# replacement on every metric and regresses no probe type:
#
#          hit@1  hit@5  hit@8   MRR   entity  lexical  paraphrase  supersession
#   replace  0.41   0.80   0.86  0.555   0.79     0.85      0.79        0.80
#   RRF K=5  0.52   0.89   0.95  0.660   1.00     0.85      0.88        0.80
#
# K is flat over 0-6 on every metric (h@5 0.89, entity 1.00, paraphrase 0.88)
# and decays back toward replacement above ~15, so this is a plateau rather
# than a tuned point. 5 is chosen inside it rather than at the K=0 boundary,
# where rank 1 scores twice rank 2 and the ordering would be fragile to any
# change in pool character; 5 also takes the plateau's best hit@8. A bootstrap
# over 400 resamples of the probe set puts P(RRF beats replacement on hit@5) at
# 0.983 for every K in 2-5, against 0.56 at the conventional K=60.
#
# WHY RANKS AND NOT A WEIGHTED SUM OF SCORES: the two scales are not
# comparable and not stably comparable. The blend is an additive relevance
# times raw trust times decay; ce is P(yes) from a softmax that measured
# 0.9997-0.9988 across seven on-topic facts. Any convex mix needs a calibration
# that would have to be re-fit whenever the reranker model changes. Rank fusion
# needs none, which is the whole reason it is the standard choice here.
_RRF_K = 5.0

# Token shapes that say "this query names something literally".
#
# NO APOSTROPHE RULE. An early version treated any ' as a quoted literal and
# routed "Wasn't there a day when...", "the author's method" and "the server's
# media folders" — three of the fourteen paraphrase probes — to the keyword
# profile. Possessives and contractions are the most ordinary prose there is;
# this is the same mistake the 2026-07-29 entity-extraction fix deleted an
# apostrophe rule to undo. Double quotes stay: they are deliberate.
#
# ALL-CAPS is >=3 letters, not >=2, because "AI" appears in prose questions
# while GPU / OOM / FTS5 / RAG name things. Measured over the 46-probe set,
# that one change moved paraphrase routing from 8/14 to 13/14 correct.
_LEXICAL_TOKEN_RE = re.compile(
    r"""
      \w*[_/\\]\w+                 # snake_case, a/path, a\path
    | \w+\.\w+                     # file.py, module.attr, 0.6B
    | [A-Za-z]+\d|\d+[A-Za-z]+     # gfx1201, Q8, 27b, v5
    | \b[A-Z]{3,}\b                # FTS5, HRR, GPU, OOM
    | \b[a-z]+[A-Z]\w*             # camelCase
    | \b[A-Z][a-z]+[A-Z]\w*        # CamelCase / FactRetriever
    """,
    re.VERBOSE,
)


def _query_shape(query: str) -> str:
    """"lexical" if the query names something literally, else "semantic".

    THE ERRORS ARE NOT SYMMETRIC, and not in the direction that first looks
    obvious. Routing a keyword question to the semantic profile widens a pool
    whose recall is already 1.00 — the gold fact is still in it, and only its
    RANK is at risk, which the cross-encoder gets a second say on. Routing a
    paraphrase question to the keyword profile leaves it with dense_k=8 and a
    0.10 share, i.e. leaves the 0.36 pool recall this whole track exists to fix
    almost exactly where it was. Recall lost at candidate generation cannot be
    won back downstream; rank can. So the tie goes to "semantic".

    Calibrated against the 46-probe set (variant C of five tried): 13/14
    paraphrase probes route semantic, and 25/32 of the rest route lexical. The
    seven that do not are entity/lexical questions phrased in pure prose, which
    is exactly the population the dense lane is harmless on.
    """
    if '"' in query:
        return "lexical"
    if re.search(r"\bfid[\s=:#]*\d", query, re.IGNORECASE):
        return "lexical"
    if _LEXICAL_TOKEN_RE.search(query):
        return "lexical"
    return "semantic"


def _search_meta(
    shape: str,
    results: "list[dict]",
    reranked: bool,
    dense_raw: "dict[int, float]",
    top_ce: float = 0.0,
    ce_per_row: "list[float] | None" = None,
) -> dict:
    """Per-QUERY confidence signals for the abstention gate (Track 2).

    `reranked` is the load-bearing field. search() returns scores on two
    different scales: the additive blend's `relevance * trust * decay`, and the
    cross-encoder's `ce * (0.5 + 0.5*trust) * decay`, where ce saturates near
    1.0 for anything on-topic. A floor calibrated on one is meaningless against
    the other, so a caller must know which it got — and the honest behaviour
    when the reranker was down is to make no confidence claim at all rather
    than a miscalibrated one.
    """
    return {
        "shape": shape,
        "reranked": reranked,
        "dense_candidates": len(dense_raw),
        "n_results": len(results),
        "top_score": float(results[0].get("score", 0.0)) if results else 0.0,
        # PER-ROW raw cross-encoder scores, aligned to `results`. Added
        # 2026-09-04 because rank fusion made `score` unusable as a per-row
        # relevance signal: an RRF score is 1/(K+rank) + 1/(K+rank), so it is
        # bounded AWAY from zero no matter how irrelevant the row is — rank 5
        # of a small pool still scores ~0.18. Any consumer that floored on
        # `score` was silently disabled by that change, which is exactly what
        # happened to claude-recall-hook.py's SCORE_FLOOR. Empty when the
        # cross-encoder did not run, for the same reason top_ce is 0.0 there:
        # the blend lives on a different scale and no floor calibrated on one
        # means anything against the other.
        # The abstention floor goes HERE, not on top_score. See the _ce stash
        # in search() for the measurement that settles it. This is the BEST
        # cross-encoder score among the returned rows, not rank 0's — see the
        # assignment in search() for why the distinction became load-bearing.
        "top_ce": top_ce,
        "ce": list(ce_per_row or []),
    }


def _env_float(name: str, default: float) -> float:
    """Positive float from the environment, or *default* when unset or unusable."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = 0.0
    if value <= 0:
        logger.warning("%s=%r is not a positive number; using %.1f", name, raw, default)
        return default
    return value


# --- abstention (Track 2) -------------------------------------------------
# Every 2026 memory benchmark punishes confident recall of nothing, and this
# store had no way to say "I have nothing". It is not a filter: rows below the
# floor are STILL RETURNED, and the caller decides. That is what separates this
# from PROBE_FLOOR in the MCP bridge, which does drop rows.
#
# ONE constant, read by BOTH doors — the prose one in scripts/hermes_memory_mcp.py
# and the JSON one in the fact_store tool. The bridge's existing PROBE_FLOOR is
# the counter-example: a bridge-local number whose calibration survives only in a
# comment, which drifts the moment one side is recalibrated.
#
# CALIBRATED 2026-09-04 by scripts/memory-retrieval-eval.py --stage abstention
# over 54 answerable and 10 provably-unanswerable probes on a backfilled
# snapshot. See that function for why the floor sits on the raw cross-encoder
# score rather than on fact["score"]. Rerun it after any change to the probe
# set, the reranker, or the embedding model.
#
# WHAT THIS FLOOR DETECTS, AND WHAT IT DOES NOT. The measured distribution is
# bimodal in a way that matters:
#
#   unanswerable, "nothing like this is stored":  0.000 0.000 0.001 0.004
#                                                 0.007 0.025
#   unanswerable, FALSE PREMISE:                  0.893 0.949 0.982 0.982
#   answerable:                                   0.007 0.023 | 0.189 … 1.000
#
# A cross-encoder scores "is this document about the same thing as this query",
# not "does this document answer this query". A question whose premise is
# invented but whose vocabulary is entirely real — "in our vLLM-versus-
# llama-swap throughput comparison, how much faster was vLLM?", "what MMLU
# score did qwen36-27b-cron get?" — is maximally on-topic by construction, and
# a genuinely related fact scores 0.98 against it. No floor on this signal can
# catch that class, and pretending otherwise would be worse than silence.
# Detecting a fabricated premise needs entailment, not similarity.
#
# So the floor is set at the MAX-MARGIN point of the empty band between the
# "nothing stored" cluster and the answerable tail, rather than at the Youden-J
# maximum. Youden still picks 0.984 — it catches nine of ten negatives, but it
# also abstains on 20% of questions the store CAN answer, which is a far more
# damaging error: the caller loses a real answer, while a missed abstention
# still hands them the rows to judge.
#
# RECALIBRATED 2026-09-05 for the rank-fusion change, on 56 answerable and 10
# unanswerable probes. Two things moved: the ranking (so a different row can be
# at rank 0) and, because of that, the signal itself, which is now the max ce
# over the RETURNED rows rather than rank 0's ce — see the top_ce assignment in
# search(). The band is (0.0242, 0.0386]: everything in it flags the same five
# negatives and abstains on the same single answerable probe. 0.031 is its
# midpoint. At 0.031:
#
#   * 55 of 56 answerable questions pass
#   * the one that does not (p39) is a probe where retrieval genuinely failed
#     to return the gold fact at all, so the gate is right about it and its
#     true cost on this set is ZERO false abstentions
#   * 5 of 10 unanswerable are flagged
#
# WHICH five is the interesting part, and it is NOT the absent/false-premise
# split the probe file labels. Flagged: p48, p49, p50, p51 (absent) and p55
# (false-premise, "Jellyfin on pdnas"). Missed: p47 (ABSENT, ce 0.951) and
# p52/p53/p54/p56 (false-premise). What actually predicts the score is whether
# the QUESTION'S VOCABULARY IS IN THE CORPUS:
#
#   p55 asks about Jellyfin — a media server nothing here has ever mentioned,
#       so every token is off-corpus and ce is 0.0000 even though the question
#       is a false premise rather than a simple gap.
#   p47 asks how worn the NVMe holding the GGUF files is — pure absence, no
#       SMART data anywhere in the store, but NVMe / GGUF / model files are
#       saturated in-corpus vocabulary, so ce is 0.9510.
#
# So the scope limit above is better stated as: a cross-encoder scores TOPIC
# OVERLAP, and cannot tell "we have no data on this property of a thing we talk
# about constantly" from "here is that data". Absence and false premise are the
# same failure wearing two labels. Closing it needs entailment — does this
# passage SUPPORT this claim — not a better threshold. Do not try to close it
# by raising the floor: the next answerable probe sits at 0.0386, so every step
# up buys unanswerable coverage with real answers.
#
# SECOND SCOPE LIMIT, measured on the live store 2026-09-04 after activation:
# ce depends on how SPECIFIC the query is, not only on what the store holds.
# The same unanswerable question asked two ways, against the same corpus and
# the same top hit (fid 1202):
#
#   "Is fail2ban (or anything like it) running on pdsrv, and what jail
#    settings — bantime, findtime, maxretry — does it apply to SSH logins?"   0.023  -> abstains
#   "Is fail2ban running on pdsrv and what are its jail settings"             0.164  -> does not
#
# A vague query is a loose target, so a merely-adjacent fact clears it. The
# probe set is written in long, specific questions, so the floor is calibrated
# on that distribution and UNDER-fires on short ones. It never over-fires from
# this: brevity raises ce, it does not lower it, so the gate stays silent
# rather than wrongly abstaining. Fixing it properly means calibrating per
# query-length band; do that only if short-query abstention turns out to matter.
#
# Set HERMES_ABSTAIN_FLOOR=0 to disable the gate entirely.
#
# NOT COMPARABLE to the 0.107 this shipped with on 2026-09-04: that floor sat
# on rank 0's ce under a different ranking. Any note quoting 0.107 predates the
# fusion change.
_ABSTAIN_FLOOR_DEFAULT = 0.031
ABSTAIN_FLOOR = _env_float("HERMES_ABSTAIN_FLOOR", _ABSTAIN_FLOOR_DEFAULT)


def no_confident_match(meta: dict, floor: "float | None" = None) -> "dict | None":
    """The abstention verdict for a search, or None to make no claim.

    None means "say nothing", and it covers three distinct situations that all
    warrant silence rather than a warning:

      * no results at all — the doors already have their own wording for that;
      * the cross-encoder did not run (down, timed out, or disabled), so the
        only calibrated signal is missing. An uncalibrated warning is worse
        than none: it would fire on the blend's numbers, which live on a
        different scale entirely;
      * a floor of 0, which is how the gate ships disabled before calibration.

    A dict is returned ONLY when the store should admit it has no confident
    answer, so a door can treat truthiness as the whole decision.
    """
    f = ABSTAIN_FLOOR if floor is None else float(floor)
    if f <= 0 or not meta.get("n_results") or not meta.get("reranked"):
        return None
    top = float(meta.get("top_ce", 0.0))
    if top >= f:
        return None
    return {"top_ce": round(top, 3), "floor": round(f, 3)}


# The shape the entailment leg is allowed to run on. NOT a tuning knob — see
# the measurement in entailment.py. Every false abstention the unrestricted rule
# produced was a paraphrase probe, and four of the five questions the
# cross-encoder misses are lexical-shaped, so this single restriction keeps 4/5
# of the catches for 1/8 of the cost and halves the number of model calls.
_ENTAIL_SHAPE = "lexical"

# The judge is asked "do these facts ANSWER the question?", so it may only be
# shown something that IS a question. A bare noun phrase has no answer, and the
# model correctly says so: measured 8 of 8 bare entity names ("CubeOH",
# "LightRAG", "R9700", ...) abstaining with 8 good rows returned and top_ce
# 0.987-1.000. That query form is not an edge case — it is what
# consolidate-synthesize STEP 1(b) issues all night, and what about()/probe()
# callers type.
#
# The calibration set is the reason this was invisible: all 56 answerable
# probes are question-mark-terminated natural-language questions of ten words
# or more, so the noun-phrase population sits entirely outside the 1/56 false
# abstention figure. This guard does not narrow that set — every probe in it
# still qualifies — it just declines to judge the population that was never
# measured. The shape gate is NOT the fix: asked directly, the judge rejects
# bare noun phrases on the semantic path too.
_INTERROGATIVE = re.compile(
    r"^\s*(what|which|how|why|when|where|who|whose|whom|is|are|was|were|do|does"
    r"|did|can|could|should|would|will|has|have|had|am|any|if|tell|explain)\b",
    re.IGNORECASE,
)
_ENTAIL_MIN_WORDS = 6


def _is_question(query: str) -> bool:
    """Is this something a yes/no "does it answer it" judgement can apply to?"""
    q = (query or "").strip()
    if len(q.split()) < _ENTAIL_MIN_WORDS:
        return False
    return "?" in q or bool(_INTERROGATIVE.match(q))


def no_entailed_answer(
    query: str,
    results: "list[dict]",
    meta: dict,
    url: "str | None" = None,
) -> "dict | None":
    """The second abstention leg: the rows are on topic but do not answer it.

    Composes with no_confident_match rather than replacing it. A door wanting
    the full measured rule writes:

        verdict = no_confident_match(meta) or no_entailed_answer(q, rows, meta)

    which is exactly the "ce floor OR lexical-gated entailment" row of the
    table in entailment.py: 9/10 unanswerable, 1/56 false abstentions.

    Returns None — make no claim — in every situation except a model that
    actually ran and actually said no. That covers the lane being switched off,
    the query not being lexical-shaped, THE QUERY NOT BEING A QUESTION, an
    empty result set, an unreachable or slow model, and an unparseable reply. NOT running on the per-turn prefetch
    path is the caller's job, and the reason this is a free function rather
    than a step inside search(): a ~1.4 s model call must be something a door
    opts into, not something every turn inherits.
    """
    if not results or meta.get("shape") != _ENTAIL_SHAPE:
        return None
    # Only judge something that is actually a question — see _is_question.
    if not _is_question(query):
        return None
    # The cross-encoder must have run. Without it the pool ordering came from
    # the blend, so `results` is a different population from the one this was
    # calibrated on, and its own floor already declines to speak.
    if not meta.get("reranked"):
        return None
    answered = entailment.answers_question(query, results, url=url)
    if answered is not False:          # True, or None for "no judgement"
        return None
    return {"reason": "entailment", "shape": meta.get("shape"),
            "top_ce": round(float(meta.get("top_ce", 0.0)), 3)}


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
        rerank_timeout: float | None = None,
        rerank_max_query_chars: int = 1500,
        rerank_max_doc_chars: int = 3000,
        dense_url: str | None = None,
        dense_timeout: float | None = None,
        rerank_fusion: bool = True,
    ):
        self.store = store
        self.half_life = temporal_decay_half_life
        self.hrr_dim = hrr_dim

        # Dense candidate lane (2026-09-04). THREE-STATE, unlike rerank_url
        # directly below: None = decide from the environment, "" = off, and a
        # string = that endpoint. rerank_url's `or os.environ.get(...)` form
        # cannot express "off" while the variable is exported, which is the trap
        # the retrieval eval documents at length — every rung of its ablation
        # silently became the production rung. The dense rung must be able to
        # say no.
        #
        # Note what does NOT gate this: a config flag. The lane is inert until
        # fact_embeddings has rows (see MemoryStore._embeddings_active), so a
        # store that was never backfilled — every test's tmp_path store — never
        # reaches the network no matter what this URL says.
        self.dense_url = embeddings.resolve_url(dense_url)
        self.dense_timeout = (
            float(dense_timeout)
            if dense_timeout is not None
            else embeddings.resolve_timeout()
        )

        # Optional cross-encoder rerank of the FTS candidate pool (see search()).
        # DISABLED unless a URL is supplied, either here or via HERMES_RERANK_URL.
        # When disabled, or when the endpoint errors/times out, the additive
        # fts/jaccard/hrr blend below is used unchanged.
        self.rerank_url = rerank_url or os.environ.get("HERMES_RERANK_URL", "")
        self.rerank_model = rerank_model
        # 8 s suits an interactive turn. The gateway's cron lane instead pays a
        # 3-8 s reranker cold start on each job's first turn (llama-swap
        # idle-unloads qwen3-rerank between jobs) and was losing that turn's
        # recall to this limit, so its unit sets HERMES_RERANK_TIMEOUT higher.
        self.rerank_timeout = (
            float(rerank_timeout)
            if rerank_timeout is not None
            else _env_float("HERMES_RERANK_TIMEOUT", 8.0)
        )
        # A reranker server has a hard per-sequence cap (llama.cpp RANK pooling
        # cannot split a sequence across ubatches). The QUERY is the term that
        # actually overruns it: prefetch() passes the caller's whole turn text,
        # which can be thousands of tokens, while stored facts are far smaller.
        # Truncating costs nothing — a cross-encoder judges relevance from the
        # head of the query — and turns a hard failure into a bounded request.
        self.rerank_max_query_chars = rerank_max_query_chars
        self.rerank_max_doc_chars = rerank_max_doc_chars
        # False reproduces the pre-2026-09-05 behaviour exactly — the
        # cross-encoder replacing the blend order. It exists for the retrieval
        # eval's ablation ladder, in the same way hrr_weight=0 does; production
        # never sets it.
        self.rerank_fusion = rerank_fusion

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
        with_meta: bool = False,
    ):
        """Hybrid search: FTS5 ∪ dense candidates → blend → trust weighting.

        Pipeline:
        0. Dense candidates (optional): embed the query once and take the top
           _DENSE_K[shape] by cosine over the whole corpus, UNIONed with stage 1.
           This is candidate GENERATION, not reranking — see the module header.
        1. FTS5 search: Get limit*3 candidates from SQLite full-text search
        2. Jaccard boost: Token overlap between query and fact content
        3. Trust weighting: final_score = relevance * trust_score
        4. Temporal decay (optional): decay = 0.5^(age_days / half_life)
        5. Cross-encoder rerank (optional): the same pool is scored
           ce * (0.5 + 0.5*trust) * decay — the trust term is clamped there
           because the cross-encoder saturates — and the resulting ranking is
           FUSED with the blend ranking by RRF rather than replacing it. See
           stage 3 and _RRF_K.

        The returned 'score' is therefore an RRF score whenever the reranker
        was reachable, and a trust-weighted relevance otherwise. Neither is a
        confidence measure; no_confident_match() reads the raw cross-encoder
        score for that, and deliberately makes no claim when reranked is False.

        Returns list of dicts with fact data + 'score' field, sorted by score
        desc. With *with_meta*, returns (results, meta) instead — meta carries
        the confidence signals the abstention gate needs (see the two retrieval
        doors: scripts/hermes_memory_mcp.py and the fact_store tool). It is a
        separate return shape rather than an extra key on every fact because
        these are properties of the QUERY, not of any one row.
        """
        shape = _query_shape(query)

        # Stage 1: Get FTS5 candidates (more than limit for reranking headroom)
        candidates = self._fts_candidates(query, category, min_trust, limit * 3)

        # Stage 0: dense candidates, UNIONed in. An FTS row wins any collision
        # because it is the one carrying fts_rank; the dense score is looked up
        # by fact_id below and so survives the dedupe either way.
        dense_raw: dict[int, float] = {}
        if self.dense_url:
            dense_rows, dense_raw = self._dense_candidates(
                query, category, min_trust, _DENSE_K[shape]
            )
            if dense_rows:
                seen = {int(f["fact_id"]) for f in candidates}
                candidates = candidates + [
                    f for f in dense_rows if int(f["fact_id"]) not in seen
                ]

        if not candidates:
            return ([], _search_meta(shape, [], False, dense_raw, 0.0)) if with_meta else []

        # Min-max, not the divide-by-max that _fts_candidates uses on BM25 rank.
        # Cosines from a retrieval-tuned embedder occupy a narrow high band —
        # measured 0.29 between two unrelated three-word strings, not 0.0 — so
        # dividing by the max would leave the whole pool crammed into the top
        # third of [0,1] and waste most of the dense term's dynamic range.
        # Min-max also gives a candidate with no stored vector the right neutral
        # value: 0.0, i.e. "as weak as the weakest thing the dense lane scored",
        # rather than "infinitely bad". That state is transitional anyway — the
        # nightly heal job exists to empty it.
        dense_norm: dict[int, float] = {}
        if dense_raw:
            lo, hi = min(dense_raw.values()), max(dense_raw.values())
            span = hi - lo
            dense_norm = {
                fid: ((val - lo) / span if span > 1e-9 else 1.0)
                for fid, val in dense_raw.items()
            }
        # The dense TERM only enters the blend when the lane actually produced
        # scores. Without this the weights would be silently rescaled by
        # (1 - share) on every search that failed to embed, quietly changing
        # ranking on the exact path that is supposed to degrade to the old one.
        dense_share = _DENSE_SHARE[shape] if dense_norm else 0.0
        blend_scale = 1.0 - dense_share

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

            # Combine FTS5 + Jaccard + HRR, then give the dense term its share.
            # The three tuned weights are SCALED rather than replaced, so their
            # ratios — including an hrr_weight=0 ablation, and the numpy-absent
            # 0.6/0.4 redistribution above — survive untouched, and
            # dense_share=0 reproduces the pre-2026-09-04 score exactly.
            relevance = (self.fts_weight * fts_score
                        + self.jaccard_weight * jaccard
                        + self.hrr_weight * hrr_sim)
            if dense_share:
                relevance = blend_scale * relevance + dense_share * dense_norm.get(
                    int(fact["fact_id"]), 0.0
                )

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

        # Stage 3 (optional): cross-encoder rerank of the SAME pool — the FTS
        # limit*3 candidates plus whatever the dense lane unioned in. The FTS
        # half is deliberately NOT widened for the reranker's benefit: measured
        # against a live store, reranking limit*3 changed ~47% of top-5 in
        # ~520 ms, while limit*6 gave the SAME ~47% at ~1190 ms. The dense
        # rows are a different matter — they are there for RECALL (a fact the
        # keyword index could not see at all), and the reranker is what lets
        # them compete on relevance rather than on the blend's dense share.
        # This stage only reorders; any failure returns None -> blend order.
        reranked = False
        if self.rerank_url and len(scored) > 1:
            ce = self._rerank_scores(query, [f["content"] for f in scored])
            if ce is not None:
                reranked = True
                # `scored` is in blend order at this point, so a candidate's
                # index IS its blend rank — RRF's first ranking, for free.
                ce_final = []
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
                    ce_final.append(
                        ce_score
                        * (0.5 + 0.5 * fact["trust_score"])
                        * fact.get("_decay", 1.0)
                    )
                    # Stashed for the abstention gate, and popped with _decay
                    # below. THE SCORE ABOVE CANNOT BE USED AS A CONFIDENCE
                    # SIGNAL: it multiplies relevance by two policy terms, and
                    # measured on the live store 2026-09-04 they dominate it.
                    # A correct answer at rank 0 scored 0.242 (ce 0.40, 18 days
                    # old) while the top hit for a question the store provably
                    # cannot answer scored 0.725 (ce 0.98, one day old) — the
                    # ranking is right in both cases, but any floor that
                    # abstains on the second also abstains on the first. ce is
                    # the only term here that is a relevance judgement.
                    fact["_ce"] = ce_score
                    fact["_ce_final"] = ce_final[-1]

                if self.rerank_fusion:
                    # Reciprocal rank fusion of the blend order and the ce
                    # order (see _RRF_K).
                    #
                    # THE TIEBREAK IS LOAD-BEARING, and it is not the stable
                    # sort's. Symmetric fusion ties EXACTLY whenever the two
                    # lists merely swap a pair — blend (1,2) against ce (2,1)
                    # both give 1/(K+1) + 1/(K+2) — and a stable sort hands
                    # every one of those to the blend, which has already had
                    # its say inside the score. Double-counting it that way
                    # reinstates the exact failure the trust clamp was added to
                    # fix: the live F1 shape where a stale trust-0.70 row
                    # ("RETIRED", ce 0.9599) sat above the trust-0.50 row that
                    # answered the question (ce 0.9990). ce_final is the term
                    # that has not been counted twice, so it breaks the tie.
                    ce_rank = [0] * len(scored)
                    for pos, idx in enumerate(
                        sorted(range(len(scored)),
                               key=lambda i: ce_final[i], reverse=True)
                    ):
                        ce_rank[idx] = pos + 1
                    for idx, fact in enumerate(scored):
                        fact["score"] = (
                            1.0 / (_RRF_K + idx + 1)
                            + 1.0 / (_RRF_K + ce_rank[idx])
                        )
                    scored.sort(
                        key=lambda x: (x["score"], x.pop("_ce_final")), reverse=True
                    )
                else:
                    for fact, final in zip(scored, ce_final):
                        fact["score"] = final
                        fact.pop("_ce_final", None)
                    scored.sort(key=lambda x: x["score"], reverse=True)

        results = scored[:limit]
        # Strip raw HRR bytes and the internal decay stash — callers expect
        # JSON-serializable dicts holding only store columns plus "score".
        # MAX over the returned rows, not rank 0's. The gate answers "is any of
        # what I am about to show you actually an answer", and since 2026-09-05
        # rank 0 is decided by RRF, which can seat a blend-favoured row the
        # cross-encoder scored near zero at the top of an otherwise good result
        # set. Reading rank 0 alone made the floor a function of the ranking
        # policy: recalibrating rank 0's ce after the fusion change moved the
        # floor 0.107 -> 0.047 and still put three answerable probes below it
        # whose gold was sitting in the returned rows with ce ~1.0. Max over
        # the returned rows is invariant to
        # how those rows are ordered, which is the property that keeps the
        # calibration alive across the next ranking change. It is NOT max over
        # the whole pool: a relevant fact the caller never sees is not a reason
        # to claim confidence.
        top_ce = max((float(f.get("_ce", 0.0)) for f in results), default=0.0)
        ce_per_row = [float(f.get("_ce", 0.0)) for f in results] if reranked else []
        for fact in results:
            fact.pop("hrr_vector", None)
            fact.pop("_decay", None)
            fact.pop("_ce", None)
        # Surfacing counts as retrieval: this path serves per-turn prefetch
        # injection and the fact_store search action, neither of which was
        # reflected in retrieval_count before.
        try:
            self.store.mark_retrieved([f["fact_id"] for f in results])
        except Exception:
            logger.debug("mark_retrieved failed", exc_info=True)
        if with_meta:
            return results, _search_meta(
                shape, results, reranked, dense_raw, top_ce, ce_per_row)
        return results

    def _dense_candidates(
        self,
        query: str,
        category: str | None,
        min_trust: float,
        k: int,
    ) -> "tuple[list[dict], dict[int, float]]":
        """Top-*k* facts by embedding cosine, plus the raw cosine of each.

        ([], {}) on ANY failure — no vectors stored, no numpy, the embedding
        endpoint down or slow. That is the whole latency guard: search() then
        proceeds on FTS candidates alone, which is precisely how it behaved
        before this lane existed. A dense lane that could raise would have put
        an HTTP call between the user and every single turn's prefetch.

        Brute force over the full corpus, deliberately (see
        MemoryStore.embedding_matrix). The trust/category filter is applied to
        the id set BEFORE the top-k cut, not after, or a category search would
        return fewer than k rows whenever the global top-k happened to sit
        outside it.
        """
        loaded = self.store.embedding_matrix()
        if loaded is None:
            return [], {}
        ids, matrix = loaded
        numpy = hrr._np()

        vector = embeddings.embed_one(
            query, url=self.dense_url, timeout=self.dense_timeout
        )
        if not vector:
            return [], {}
        q = numpy.asarray(vector, dtype=numpy.float32)
        norm = float(numpy.linalg.norm(q))
        if norm <= 0:
            return [], {}
        # Stored vectors are normalised on the way in (embeddings.to_blob), so
        # normalising the query is all that is left to make this dot product a
        # cosine.
        sims = matrix @ (q / norm)

        conn = self.store._conn
        where = "trust_score >= ?"
        params: list = [min_trust]
        if category:
            where += " AND category = ?"
            params.append(category)
        try:
            allowed = {
                int(r[0])
                for r in conn.execute(
                    f"SELECT fact_id FROM facts WHERE {where}", params
                ).fetchall()
            }
        except Exception:
            return [], {}
        if not allowed:
            return [], {}

        mask = numpy.fromiter(
            (int(fid) in allowed for fid in ids), dtype=bool, count=len(ids)
        )
        if not mask.any():
            return [], {}
        cand_ids = ids[mask]
        cand_sims = sims[mask]
        k = min(int(k), int(cand_ids.shape[0]))
        top = numpy.argsort(-cand_sims)[:k]
        chosen = [int(cand_ids[i]) for i in top]
        raw = {int(cand_ids[i]): float(cand_sims[i]) for i in top}

        placeholders = ",".join("?" * len(chosen))
        try:
            rows = conn.execute(
                f"SELECT * FROM facts WHERE fact_id IN ({placeholders})", chosen
            ).fetchall()
        except Exception:
            return [], {}
        by_id = {int(r["fact_id"]): dict(r) for r in rows}
        # Preserve cosine order, and drop any id the SELECT could not resolve
        # (a vector whose fact was deleted between the two statements).
        out = []
        for fid in chosen:
            fact = by_id.get(fid)
            if fact is None:
                raw.pop(fid, None)
                continue
            # No fts_rank: this row did not come from the keyword index, and
            # search() reads it with .get(..., 0.0). Inventing a rank here would
            # give a dense-only hit a free BM25 score.
            out.append(fact)
        return out, raw

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
