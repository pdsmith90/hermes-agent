"""Dense embedding client for the fact store's candidate lane.

WHY THIS EXISTS. The 2026-09-02 four-rung retrieval eval (46 probes, 939 facts)
found that every arm above raw BM25 reorders the SAME FTS5 candidate pool, and
that pool's recall@24 is 1.00 on entity/lexical/supersession probes and 0.36 on
paraphrase ones. The cross-encoder was already extracting 100% of what the pool
could yield on paraphrase (hit@5 0.357 against a 0.357 ceiling) — so the
constraint was never ranking, it was candidate generation. Keyword search cannot
retrieve a fact that shares no keywords with the question. This module is the
second candidate generator.

It talks to the SAME llama-swap endpoint everything else on this box talks to
(127.0.0.1:18000), using the `jina-embed` entry that has had no consumer since
the OpenViking sidecar was disabled on 2026-08-19. No new resident model, no new
service, no vector database: 1008 facts x 1024 dims of float32 is 4 MB, and a
brute-force cosine over that is a single numpy matmul.

FAILURE IS ALWAYS SILENT. Every function here returns None rather than raising.
The read path degrades to FTS-only candidates (exactly the pre-2026-09 behaviour)
and the write path leaves the embedding row absent for the nightly heal job to
backfill — the same self-heal philosophy as hermes-numpy-ensure. An embedding
call must never be able to fail a fact write or break a search.

SERVING CONSTRAINTS, measured 2026-09-04 against the live llama-swap entry:

  * the GGUF is jina-embeddings-v5-small-retrieval-Q8_0 (610 MB), served with
    `-c 8192 --parallel 4 -b 2048 -ub 2048`, so ONE sequence gets 8192/4 = 2048
    tokens. Facts cap at 2710 chars (~700 tokens) and clear it; document chunks
    from the docs index do not, on their own, so MAX_CHARS truncates.
  * output is 1024-dim and ALREADY L2-normalised (measured norm exactly 1.0),
    but normalise() does not assume it — a re-quantised or swapped model must
    not silently turn cosine into an unnormalised dot product.
  * throughput ~2200 tok/s. Batch 32 embedded 5611 tokens in 2.52 s; batch 64
    took 7.69 s for 11060, i.e. no better per token and a longer tail on any
    single failure. BATCH is 32 for that reason.
  * warm single-query latency is 14-500 ms. `jina-embed` carries `ttl: 600` in
    the llama-swap fleet, so an idle-unloaded model costs a load on the first
    call after ten quiet minutes; that is what DEFAULT_TIMEOUT has to cover, and
    the same cold-start shape that cost the reranker a turn's recall on
    2026-09-03 (see HERMES_RERANK_TIMEOUT in the gateway drop-in).
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

try:
    from . import holographic as hrr
except ImportError:  # pragma: no cover - direct-module import path
    import holographic as hrr  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://127.0.0.1:18000/v1/embeddings"
DEFAULT_MODEL = "jina-embed"
DEFAULT_DIM = 1024

# Covers a cold llama-swap load of the 610 MB GGUF, not just a warm call. A
# too-tight budget here does not error — it silently drops the dense half of the
# candidate pool, which is invisible in logs and looks exactly like "the dense
# lane did not help". Overridable per-process for the cron lane, which has
# nobody waiting on it.
DEFAULT_TIMEOUT = 8.0

# One sequence gets 2048 tokens (see module docstring). At the corpus's measured
# ~4 chars/token that is ~8000 chars; 6000 leaves headroom for token-dense text
# (paths, hex, base64-looking strings) where the ratio is far worse. Truncation
# is the right failure here: an embedding of the head of a chunk is a usable
# vector, and a 400-error is not.
MAX_CHARS = 6000

# 32 inputs per request. See the throughput measurement in the module docstring.
BATCH = 32


def resolve_url(explicit: "str | None" = None) -> str:
    """The embeddings endpoint, or "" when the dense lane is off.

    DELIBERATELY NOT the `rerank_url or os.environ.get(...)` idiom used for the
    cross-encoder one file over. That form makes `rerank_url=""` mean "fall back
    to the environment", so a caller cannot turn the stage OFF while
    HERMES_RERANK_URL is exported — which is exactly the trap the retrieval eval
    documents at length (every rung of its ablation silently became the `full`
    rung, and four identical rows read as "the components add nothing"). Here an
    explicit "" is authoritative OFF and None means "decide from the
    environment", so the eval's `+dense` rung can be built honestly.
    """
    if explicit is not None:
        return explicit.strip()
    return os.environ.get("HERMES_EMBED_URL", DEFAULT_URL).strip()


def resolve_model() -> str:
    return os.environ.get("HERMES_EMBED_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def resolve_timeout() -> float:
    raw = os.environ.get("HERMES_EMBED_TIMEOUT", "").strip()
    if not raw:
        return DEFAULT_TIMEOUT
    try:
        value = float(raw)
    except ValueError:
        value = 0.0
    if value <= 0:
        logger.warning(
            "HERMES_EMBED_TIMEOUT=%r is not a positive number; using %.1f",
            raw,
            DEFAULT_TIMEOUT,
        )
        return DEFAULT_TIMEOUT
    return value


def available() -> bool:
    """Can the dense lane run in this process at all?

    numpy gates it for the same reason it gates the HRR term: the read path is a
    matmul over the whole corpus, and a pure-Python fallback for that would be
    slower than the keyword search it is meant to widen.
    """
    return hrr._HAS_NUMPY


def embed(
    texts: "list[str]",
    *,
    url: "str | None" = None,
    model: str = "",
    timeout: "float | None" = None,
) -> "list[list[float]] | None":
    """Embed *texts* in order. None on ANY failure, never raises.

    `url` follows resolve_url()'s three-state contract — None means "decide from
    the environment", "" is authoritative OFF. It is deliberately NOT
    `url or resolve_url()`: that idiom is what makes rerank_url="" fail to
    disable the reranker, and writing it here would have made an eval rung that
    asked for no dense lane get one anyway.

    Batched at BATCH per request; a failure in any batch fails the whole call,
    because a partial result would silently produce a half-populated candidate
    pool that looks like a working dense lane.
    """
    if not texts:
        return []
    endpoint = resolve_url(url)
    if not endpoint:
        return None
    model = model or resolve_model()
    timeout = resolve_timeout() if timeout is None else float(timeout)

    out: list[list[float]] = []
    for start in range(0, len(texts), BATCH):
        chunk = [(t or "")[:MAX_CHARS] for t in texts[start : start + BATCH]]
        payload = json.dumps(
            {
                "model": model,
                # "float", explicitly: llama.cpp's default encoding for this
                # endpoint is not guaranteed across builds and the base64 form
                # has been unreliable here.
                "encoding_format": "float",
                "input": chunk,
            }
        ).encode()
        try:
            req = urllib.request.Request(
                endpoint, payload, {"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.load(resp)
            # The server may return `data` in any order; `index` is authoritative.
            items = sorted(data["data"], key=lambda d: int(d["index"]))
            if len(items) != len(chunk):
                logger.debug(
                    "embed: asked for %d vectors, got %d", len(chunk), len(items)
                )
                return None
            out.extend([float(x) for x in item["embedding"]] for item in items)
        except (urllib.error.URLError, OSError, KeyError, ValueError, TypeError):
            logger.debug("embed: request failed", exc_info=True)
            return None
    return out


def embed_one(text: str, **kwargs) -> "list[float] | None":
    """Single-text convenience. None on failure, same as embed()."""
    vectors = embed([text], **kwargs)
    if not vectors:
        return None
    return vectors[0]


def to_blob(vector: "list[float]") -> bytes:
    """L2-normalise and serialise as native-order float32.

    Normalising on the way IN is what lets the read path score the whole corpus
    with one dot product instead of a per-row division. jina-embed already
    returns unit vectors, so this is a no-op in production — it exists so that
    swapping the model can never silently turn the cosine into a magnitude
    comparison.

    No format prefix, unlike holographic.phases_to_bytes: fact_embeddings stores
    `dim` in its own column, so the blob is never ambiguous and the reader can
    assert the length instead of sniffing it.
    """
    numpy = hrr._np()
    arr = numpy.asarray(vector, dtype=numpy.float32)
    norm = float(numpy.linalg.norm(arr))
    if norm > 0:
        arr = arr / norm
    return arr.astype(numpy.float32).tobytes()


def from_blob(blob: bytes, dim: int):
    """Deserialise a float32 blob written by to_blob(). None if it is not one.

    A wrong-length blob means the stored vector was written by a different model
    than the one now configured; returning None drops that row from the dense
    scan rather than corrupting the matmul with a reshaped array.
    """
    numpy = hrr._np()
    itemsize = numpy.dtype(numpy.float32).itemsize
    if not blob or len(blob) != dim * itemsize:
        return None
    return numpy.frombuffer(blob, dtype=numpy.float32)
