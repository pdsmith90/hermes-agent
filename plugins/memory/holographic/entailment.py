"""Entailment check for the abstention gate's blind spot.

WHY THIS EXISTS. The abstention gate floors the cross-encoder score, and a
cross-encoder answers "is this passage ABOUT the same thing as this query", not
"does it ANSWER it". Measured over the 66-probe set, that misses half the
unanswerable questions — and not the half the labels predict. At floor 0.031 it
flags p48/p49/p50/p51 (absent) and p55 (a FALSE PREMISE), while missing p47 (a
PURE ABSENCE, ce 0.9510) and p52/p53/p54/p56. What actually predicts the score
is whether the question's VOCABULARY is in the corpus:

  * p55 asks about Jellyfin on pdnas — a media server nothing here has ever
    mentioned, so every token is off-corpus and ce is 0.0000.
  * p47 asks how worn the NVMe holding the GGUF files is — no SMART data exists
    anywhere in the store, but NVMe / GGUF / model-files is saturated in-corpus
    vocabulary, so ce is 0.9510.

Absence and false premise are the same failure wearing two labels, and no floor
on a similarity signal separates them. This module asks a generative model the
question the cross-encoder cannot: do these rows actually answer it.

MEASURED 2026-09-04, 66 probes, top-5 production retrieval:

  rule                                  negatives  of the 5 ce misses  false abst.
  ce floor 0.031 alone                       5/10                 0/5        0/56
  entailment alone                          10/10                 5/5        8/56
  ce floor OR lexical-gated entailment       9/10                 4/5        1/56

THE GATE IS WHAT MAKES IT AFFORDABLE, and it is not a tuning knob. Every one of
the eight false abstentions under the unrestricted rule is a PARAPHRASE probe:
a gold fact that answers in different words reads as "not an answer" to a strict
judge. Four of the five questions the cross-encoder misses are lexical-shaped.
So restricting the call to lexical-shaped queries keeps four of five catches for
one of eight costs — and halves the number of calls as a side effect. Do not
"improve" this by removing the shape gate.

TWO THINGS THAT MEASURED THE OPPOSITE OF EXPECTATION, recorded so they are not
retried:

  * The SIMPLE prompt won. Asking the model to state the question's
    presupposition and then judge support for THAT scored the same 10/10
    detection but 37.5% false abstention against 14.3%, and took 4.4 s against
    1.1 s. It made the model stricter about everything rather than sharper
    about premises. PROMPT below is the simple form.
  * The CHEAP model is enough. qwen35-4b-util is twice as strict as
    ornith35-a3b-eval unrestricted (32.1% against 16.1% false abstention), but
    at the gated operating point the two are IDENTICAL — 9/10, 4/5, 1/56 —
    because the gate excludes exactly the paraphrase queries where the small
    model over-fires. qwen35-4b-util is also a member of llama-swap's
    `retrieval` group (swap: false), so it co-loads beside qwen3-rerank and
    jina-embed instead of evicting whatever else is resident. That is the whole
    reason this can sit on an interactive path at all.

FAILURE IS ALWAYS SILENT, as in embeddings.py: every function returns None
rather than raising, and None means "make no claim". A model that is down, slow,
refusing to load, or answering in an unparseable format must leave the gate
exactly as it was before this module existed.

DISABLED BY DEFAULT. HERMES_ENTAIL_URL unset means off, following
HERMES_RERANK_URL rather than the dense lane's data-driven activation — there is
no table whose emptiness can make this inert, so the URL is the switch, and
tests that do not set it never reach the network.

ONE HAZARD WORTH NAMING. qwen35-4b-util's group has `swap: false`, which means
requesting it CO-LOADS rather than evicting. When a 27B is already resident
there may be no room, and llama-one-card.sh REFUSES rather than splitting (a
change made 2026-09-04 after the OOM/segfault cascade recorded in fid 1451).
A refusal surfaces here as a failed request and therefore as None, which is the
correct behaviour — but it does mean this lane is expected to be quiet during
the nightly cron window, when the 27B holds the card.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen35-4b-util"
# Cold load of the 4B measured at 2.5 s, warm calls at ~1.4 s. 20 s covers a
# cold load behind a busy card without hanging an interactive turn.
DEFAULT_TIMEOUT = 20.0
# Facts are capped so a long fact cannot push the prompt past the served
# context. 5 x 900 chars plus the frame is well inside qwen35-4b-util's window.
MAX_FACTS = 5
MAX_FACT_CHARS = 900
MAX_QUERY_CHARS = 1000
# Generous, and deliberately not 120. A reasoning-capable model spends its
# budget on hidden tokens and returns finish=length with EMPTY content — the
# trap that voided 32 of 66 verdicts in the 2026-09-04 judge run. Thinking is
# also switched off explicitly below; this is the belt to that pair of braces.
MAX_TOKENS = 512

PROMPT = """You are auditing a memory store.

QUESTION:
{query}

RETRIEVED FACTS:
{facts}

Do the retrieved facts actually ANSWER the question?
Reply with exactly one line: VERDICT: YES or VERDICT: NO"""


def resolve_url(explicit: "str | None" = None) -> str:
    """Three-state, like embeddings.resolve_url and unlike rerank_url.

    An explicit "" is authoritative OFF even when the environment variable is
    exported; None means "decide from the environment". The default is OFF, so
    every test and every process that has not opted in stays off the network.
    """
    if explicit is not None:
        return explicit.strip()
    return os.environ.get("HERMES_ENTAIL_URL", "").strip()


def resolve_model() -> str:
    return os.environ.get("HERMES_ENTAIL_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def resolve_timeout() -> float:
    try:
        return float(os.environ.get("HERMES_ENTAIL_TIMEOUT", DEFAULT_TIMEOUT))
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT


# Exactly YES or NO, as a whole word, right after the label. Not a substring
# test: "NO" is inside NONE, UNKNOWN, NOT ENOUGH INFORMATION and CANNOT
# DETERMINE, every one of which is a REFUSAL to judge and was being read as a
# confident no — an abstention invented by the parser, which is the one thing
# this module's contract forbids.
_VERDICT_RE = re.compile(r"VERDICT\s*:?\s*(YES|NO)\b", re.IGNORECASE)


def _parse(text: str) -> "bool | None":
    """The model's verdict, or None when it did not give exactly one.

    UNANIMITY, not last-wins. Last-wins was justified by a model that restates
    the instruction BEFORE answering, and it is wrong in the other direction:
    this module's own PROMPT ends "...VERDICT: YES or VERDICT: NO", so any
    reply that echoes that line AFTER its answer — or merely says "the
    alternative verdict would be NO" — ends on a NO window and silently
    inverted a YES into an abstention.

    Disagreeing verdicts therefore mean "no judgement", not "the last one".
    That is the safe direction on both counts: an echo can no longer
    manufacture an abstention, and a genuine NO buried in an echo is merely
    lost rather than inverted. An unparseable reply stays None — one reply in
    66 was unparseable in the measured run, and reading it as NO would be a
    false abstention the model never gave.
    """
    if not text:
        return None
    verdicts = {m.group(1).upper() == "YES" for m in _VERDICT_RE.finditer(text)}
    return verdicts.pop() if len(verdicts) == 1 else None


def answers_question(
    query: str,
    facts: "list[dict]",
    url: "str | None" = None,
    model: "str | None" = None,
    timeout: "float | None" = None,
) -> "bool | None":
    """True if the model judges that *facts* answer *query*. None on any failure.

    None is not "no". It is "this lane made no judgement", and every caller must
    treat it as leaving the decision to the cross-encoder floor alone.
    """
    endpoint = resolve_url(url)
    if not endpoint or not query or not facts:
        return None
    rendered = "\n\n".join(
        f"[fid {f.get('fact_id')}] {str(f.get('content', ''))[:MAX_FACT_CHARS]}"
        for f in facts[:MAX_FACTS]
    )
    payload = json.dumps({
        "model": model or resolve_model(),
        "messages": [{"role": "user", "content": PROMPT.format(
            query=query[:MAX_QUERY_CHARS], facts=rendered)}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        # Qwen3-family thinking under a long prompt exhausts the completion
        # budget and returns empty content; the documented escape hatch is a
        # chat-template kwarg, and it is ignored by models without one.
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    try:
        req = urllib.request.Request(
            endpoint, payload, {"Content-Type": "application/json"})
        with urllib.request.urlopen(
            req, timeout=timeout if timeout is not None else resolve_timeout()
        ) as resp:
            body = json.load(resp)
        choice = body["choices"][0]
        return _parse(choice["message"].get("content") or "")
    except Exception:
        # BROAD ON PURPOSE, and not a lint slip. This function's entire
        # contract is "never raises, None means no judgement", and an
        # enumerated tuple cannot hold that line: http.client.HTTPException
        # derives from Exception rather than OSError (so a truncated body or a
        # garbled status line escaped), and walking
        # body["choices"][0]["message"] raises TypeError or AttributeError on
        # any response shaped differently than expected. Seven such shapes were
        # found escaping. What escapes does not merely disable this lane — the
        # fact_store door wraps the whole action in `except Exception` and
        # returns a tool error, so a malformed reply from an OPTIONAL
        # second-opinion model would discard a search that had already
        # succeeded and ranked its rows.
        logger.debug("entailment check failed", exc_info=True)
        return None
