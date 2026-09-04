# Holographic Memory Provider

Local SQLite fact store with FTS5 search, trust scoring, entity resolution, and HRR-based compositional retrieval.

## Requirements

None — uses SQLite (always available). NumPy optional for HRR algebra.

## Setup

```bash
hermes memory setup    # select "holographic"
```

Or manually:
```bash
hermes config set memory.provider holographic
```

## Config

Config in `config.yaml` under `plugins.hermes-memory-store`:

| Key | Default | Description |
|-----|---------|-------------|
| `db_path` | `$HERMES_HOME/memory_store.db` | SQLite database path |
| `auto_extract` | `false` | Auto-extract facts at session end |
| `default_trust` | `0.5` | Default trust score for new facts |
| `hrr_dim` | `1024` | HRR vector dimensions |

## Optional: cross-encoder reranking

`fact_store(action='search')` and per-turn prefetch both go through
`FactRetriever.search()`, which ranks an FTS5 candidate pool with an additive
FTS + Jaccard + HRR blend. That blend is bag-of-words, so a query containing a
domain acronym can rank an unrelated fact that merely shares the word.

A cross-encoder reranker can reorder the pool instead. It is **disabled by
default**; set an OpenAI-compatible `/v1/rerank` endpoint to enable:

```bash
export HERMES_RERANK_URL=http://localhost:18000/v1/rerank
```

| Key | Default | Description |
|-----|---------|-------------|
| `rerank_url` | `$HERMES_RERANK_URL`, else empty | Rerank endpoint. Empty = disabled |
| `rerank_model` | `qwen3-rerank` | `model` field sent in the request |
| `rerank_timeout` | `$HERMES_RERANK_TIMEOUT`, else `8.0` | Per-request timeout, seconds |

Behaviour and constraints:

- **Two timeouts, keep them ordered.** The agent-side prefetch budget
  (`HERMES_MEMORY_PREFETCH_TIMEOUT`, default 8 s, `agent/memory_manager.py`)
  caps the whole recall step; `rerank_timeout` caps just the rerank call
  inside it. Keep the rerank timeout below the budget so a slow reranker
  still falls back to the additive blend in time — with the two equal (the
  old 8 s/8 s pairing) a cold-started reranker cost the turn its recall
  outright. A gateway serving cron can afford larger values than an
  interactive session; set both in its systemd drop-in.

- **Fails open.** Any error, timeout, or malformed response falls back to the
  additive blend. `search()` is on the hot path for every turn's prefetch and
  every cron fact search, so a down reranker must not break retrieval.
- **The FTS pool is not widened for the reranker's sake.** Only the `limit*3`
  FTS pool plus the dense candidates (below) are reordered. Measured against a
  live store, reranking the FTS pool alone changed ~47% of top-5 in ~520 ms;
  widening it to `limit*6` changed the same ~47% but cost ~1190 ms. The dense
  lane widens the pool for a different reason — recall, not rank.
- **Trust weighting is preserved** — the final score is
  `relevance_score * trust_score`, so a low-trust fact cannot win on relevance
  alone.
- Scores are used as-is. A llama.cpp reranker with RANK pooling already returns
  a probability in `[0,1]`; do not apply a further sigmoid or min-max.

Verified against `llama-server --reranking` serving Qwen3-Reranker. Note that
`--reranking` alone implies `--embedding` and `--pooling rank`, and that RANK
pooling cannot split a sequence across ubatches — so `--ubatch-size` is a hard
cap on `template + query + longest single document`, not on their sum.

## Dense candidate lane (embeddings)

Keyword search cannot retrieve a fact that shares no words with the question.
Measured on a 939-fact store (2026-09-02), the FTS candidate pool held the
gold fact for 100% of entity/lexical questions and 36% of paraphrased ones —
and every stage above FTS, the reranker included, only reorders that pool.
`search()` therefore also embeds the query and UNIONs the top-k facts by cosine
into the pool before the blend and the rerank run. Brute force over the whole
corpus (1000 × 1024 float32 is 4 MB and one matmul); no vector database.

**Activation is the data, not a flag.** The lane is inert while the
`fact_embeddings` table is empty — a fresh store, and every test store, makes
no network call. Populating the table switches it on:

```bash
~/.hermes/hermes-agent/venv/bin/python ~/.hermes/scripts/memory-index-heal.py
```

After that `add_fact`/`update_fact` embed each new or rewritten fact
themselves (outside the write lock; a failure leaves the row absent for the
nightly heal, never fails the write). `DELETE FROM fact_embeddings` turns the
lane off again and is also how you force a re-embed after a model swap —
rows for a different `model` count as missing.

| Knob | Default | Description |
|------|---------|-------------|
| `HERMES_EMBED_URL` | `http://127.0.0.1:18000/v1/embeddings` | Endpoint. **Empty is authoritative OFF** (unlike `HERMES_RERANK_URL`) |
| `HERMES_EMBED_MODEL` | `jina-embed` | `model` field sent in the request; stored with each vector |
| `HERMES_EMBED_TIMEOUT` | `8.0` | Per-request timeout; the gateway drop-in sets 15 for cold loads |
| `HERMES_ABSTAIN_FLOOR` | calibrated constant | Cross-encoder floor below which search admits "no confident match" |

The weighting is query-adaptive: a question carrying a quoted string, path,
identifier, ALL-CAPS acronym or fid reference takes 8 dense candidates at a
0.10 share of the blend; plain prose takes 24 at 0.50. See `_query_shape` in
`retrieval.py` for the routing and its measured error asymmetry. Rung
`+dense` of `scripts/memory-retrieval-eval.py` isolates the lane's effect.

## Abstention

`search(..., with_meta=True)` returns `(rows, meta)`; `meta["top_ce"]` is the
raw cross-encoder score of the best row and `no_confident_match(meta)` is the
verdict both retrieval doors render (a `NO-CONFIDENT-MATCH` line in the MCP
bridge, sibling keys in the `fact_store` JSON). Rows are never filtered by it.
The floor sits on the cross-encoder score and not on `score` because `score`
multiplies relevance by trust and temporal decay — a correct 18-day-old answer
measured 0.24 while a wrong 1-day-old one measured 0.73. Calibrate with
`memory-retrieval-eval.py --stage abstention` against negative probes.

## Tools

| Tool | Description |
|------|-------------|
| `fact_store` | 9 actions: add, search, probe, related, reason, contradict, update, remove, list |
| `fact_feedback` | Rate facts as helpful/unhelpful (trains trust scores) |
