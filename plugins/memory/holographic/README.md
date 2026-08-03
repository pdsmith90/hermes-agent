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
| `rerank_timeout` | `5.0` | Per-request timeout, seconds |

Behaviour and constraints:

- **Fails open.** Any error, timeout, or malformed response falls back to the
  additive blend. `search()` is on the hot path for every turn's prefetch and
  every cron fact search, so a down reranker must not break retrieval.
- **The candidate pool is not widened.** Only the existing `limit*3` FTS pool is
  reordered. Measured against a live store, reranking that pool changed ~47% of
  top-5 in ~520 ms; widening to `limit*6` changed the same ~47% but cost
  ~1190 ms.
- **Trust weighting is preserved** — the final score is
  `relevance_score * trust_score`, so a low-trust fact cannot win on relevance
  alone.
- Scores are used as-is. A llama.cpp reranker with RANK pooling already returns
  a probability in `[0,1]`; do not apply a further sigmoid or min-max.

Verified against `llama-server --reranking` serving Qwen3-Reranker. Note that
`--reranking` alone implies `--embedding` and `--pooling rank`, and that RANK
pooling cannot split a sequence across ubatches — so `--ubatch-size` is a hard
cap on `template + query + longest single document`, not on their sum.

## Tools

| Tool | Description |
|------|-------------|
| `fact_store` | 9 actions: add, search, probe, related, reason, contradict, update, remove, list |
| `fact_feedback` | Rate facts as helpful/unhelpful (trains trust scores) |
