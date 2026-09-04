"""The entailment leg of the abstention gate.

The cross-encoder floor scores topic overlap, so it misses half the
unanswerable probes — and not the half the absent/false-premise labels predict.
This leg asks a generative model whether the returned rows actually answer the
question. Everything here guards a contract rather than a score: silence on
failure, the shape gate, and composition with the existing floor.
"""
import json
import pytest

from plugins.memory.holographic import entailment
from plugins.memory.holographic.retrieval import (
    no_confident_match,
    no_entailed_answer,
)

ROWS = [{"fact_id": 7, "content": "LESSON: the reranker runs on the GPU."}]
# Every query used here must be a QUESTION — the lane declines anything else
# (see _is_question). Bare noun phrases are covered by TestQueryForm below.
Q = "Which GPU does the document-store reranker actually run on today?"
META = {"shape": "lexical", "reranked": True, "n_results": 1, "top_ce": 0.98}


class TestSwitch:
    """Disabled by default, and "" is authoritative off."""

    def test_off_unless_the_url_is_set(self, monkeypatch):
        monkeypatch.delenv("HERMES_ENTAIL_URL", raising=False)
        assert entailment.resolve_url() == ""
        assert no_entailed_answer(Q, ROWS, META) is None

    def test_empty_string_beats_an_exported_variable(self, monkeypatch):
        """The rerank_url trap, not repeated.

        `x or os.environ.get(...)` cannot express "off" while the variable is
        exported, which is what once made every rung of the retrieval ablation
        silently become the production rung.
        """
        monkeypatch.setenv("HERMES_ENTAIL_URL", "http://stub/v1/chat/completions")
        assert entailment.resolve_url() == "http://stub/v1/chat/completions"
        assert entailment.resolve_url("") == ""
        assert no_entailed_answer(Q, ROWS, META, url="") is None

    def test_no_network_call_when_off(self, monkeypatch):
        monkeypatch.delenv("HERMES_ENTAIL_URL", raising=False)

        def explode(*a, **k):                       # pragma: no cover
            raise AssertionError("reached the network with the lane off")

        monkeypatch.setattr(entailment.urllib.request, "urlopen", explode)
        assert entailment.answers_question("q", ROWS) is None


class TestGate:
    """The shape gate is the thing that makes this affordable. Guard it."""

    @pytest.fixture
    def says_no(self, monkeypatch):
        calls = []

        def fake(query, facts, url=None, model=None, timeout=None):
            calls.append(query)
            return False

        monkeypatch.setattr(entailment, "answers_question", fake)
        return calls

    def test_flags_a_lexical_query_the_model_rejects(self, says_no):
        v = no_entailed_answer("what MMLU score did X get", ROWS, META)
        assert v and v["reason"] == "entailment"
        assert len(says_no) == 1

    def test_semantic_queries_are_not_even_asked(self, says_no):
        """Not just unflagged — the model must not be CALLED.

        Every false abstention in the measured run was a paraphrase probe, and
        the saving in model calls is half of them. A version that asked and
        then discarded the answer would keep the cost and lose the point.
        """
        meta = {**META, "shape": "semantic"}
        assert no_entailed_answer("Wasn't there a day when the review reported nothing?", ROWS, meta) is None
        assert says_no == []

    def test_silent_when_the_reranker_did_not_run(self, says_no):
        assert no_entailed_answer(Q, ROWS, {**META, "reranked": False}) is None
        assert says_no == []

    def test_silent_on_an_empty_result_set(self, says_no):
        assert no_entailed_answer(Q, [], META) is None
        assert says_no == []


class TestNoneIsNotNo:
    """A lane that cannot judge must leave the gate exactly as it found it."""

    @pytest.mark.parametrize("verdict", [True, None])
    def test_only_an_explicit_no_flags(self, monkeypatch, verdict):
        monkeypatch.setattr(entailment, "answers_question",
                            lambda *a, **k: verdict)
        assert no_entailed_answer(Q, ROWS, META) is None

    def test_a_dead_model_is_silence_not_abstention(self, monkeypatch):
        monkeypatch.setenv("HERMES_ENTAIL_URL", "http://stub/v1/chat/completions")

        def boom(*a, **k):
            raise OSError("connection refused")

        monkeypatch.setattr(entailment.urllib.request, "urlopen", boom)
        assert entailment.answers_question("q", ROWS) is None
        assert no_entailed_answer(Q, ROWS, META) is None


class TestParsing:
    def test_verdicts(self):
        assert entailment._parse("VERDICT: YES") is True
        assert entailment._parse("VERDICT: NO") is False
        assert entailment._parse("  verdict:  no  ") is False

    def test_disagreeing_verdicts_are_no_judgement(self):
        """UNANIMITY, not last-wins — last-wins was directional and wrong.

        PROMPT itself ends "...VERDICT: YES or VERDICT: NO", so any reply that
        echoes that line AFTER answering ends on a NO window. Under last-wins
        that inverted a YES into an abstention the model never gave.
        """
        assert entailment._parse(
            "I must reply VERDICT: YES or VERDICT: NO.\nVERDICT: NO") is None
        assert entailment._parse("VERDICT: YES\nVERDICT: YES") is True

    def test_trailing_prose_does_not_flip_the_verdict(self):
        assert entailment._parse(
            "VERDICT: YES\nThe rows are not exhaustive but they do answer it."
        ) is True
        # "verdict would be NO" has no YES/NO immediately after the label, so
        # it is not a verdict at all.
        assert entailment._parse(
            "VERDICT: YES. Note: the alternative verdict would be NO."
        ) is True

    @pytest.mark.parametrize("refusal", [
        "VERDICT: UNKNOWN", "VERDICT: NONE", "VERDICT: CANNOT DETERMINE",
        "VERDICT: NOT ENOUGH INFORMATION",
    ])
    def test_a_refusal_to_judge_is_not_a_no(self, refusal):
        """`"NO" in tail` read every one of these as a confident no.

        Each is the model declining, and turning a decline into an abstention
        is precisely the parser-invented false abstention this contract bans.
        """
        assert entailment._parse(refusal) is None

    @pytest.mark.parametrize("text", ["", "maybe", "I am not sure", None])
    def test_unparseable_is_none_not_a_guess(self, text):
        """One reply in 66 was unparseable in the measured run.

        Reading that as NO would be a false abstention invented by the parser.
        """
        assert entailment._parse(text) is None


class TestRequestShape:
    def test_thinking_is_disabled_and_the_budget_is_not_120(self, monkeypatch):
        """The trap that voided 32 of 66 verdicts on 2026-09-04.

        Hidden reasoning consumes the completion budget and the model returns
        finish=length with EMPTY content, which a caller reads as a failure to
        answer rather than as a failure to ask properly.
        """
        seen = {}

        class Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps(
                    {"choices": [{"message": {"content": "VERDICT: YES"}}]}
                ).encode()

        def capture(req, timeout=None):
            seen["body"] = json.loads(req.data)
            seen["timeout"] = timeout
            return Resp()

        monkeypatch.setenv("HERMES_ENTAIL_URL", "http://stub/v1/chat/completions")
        monkeypatch.setattr(entailment.urllib.request, "urlopen", capture)
        assert entailment.answers_question("q", ROWS) is True
        body = seen["body"]
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert body["max_tokens"] >= 256
        assert body["temperature"] == 0
        assert body["model"] == entailment.DEFAULT_MODEL

    def test_facts_and_query_are_capped(self, monkeypatch):
        seen = {}

        class Resp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self):
                return json.dumps(
                    {"choices": [{"message": {"content": "VERDICT: NO"}}]}
                ).encode()

        def capture(req, timeout=None):
            seen["body"] = json.loads(req.data)
            return Resp()

        monkeypatch.setenv("HERMES_ENTAIL_URL", "http://stub/v1/chat/completions")
        monkeypatch.setattr(entailment.urllib.request, "urlopen", capture)
        rows = [{"fact_id": i, "content": "x" * 5000} for i in range(20)]
        assert entailment.answers_question("q" * 9000, rows) is False
        prompt = seen["body"]["messages"][0]["content"]
        assert prompt.count("[fid ") == entailment.MAX_FACTS
        assert len(prompt) < entailment.MAX_QUERY_CHARS + \
            entailment.MAX_FACTS * (entailment.MAX_FACT_CHARS + 40) + 400


class TestComposition:
    """`no_confident_match(meta) or no_entailed_answer(...)` is the measured rule."""

    def test_the_floor_still_fires_on_its_own(self, monkeypatch):
        monkeypatch.setattr(entailment, "answers_question", lambda *a, **k: True)
        low = {**META, "top_ce": 0.001}
        assert no_confident_match(low) is not None
        assert (no_confident_match(low) or no_entailed_answer(Q, ROWS, low))

    def test_entailment_catches_what_the_floor_passes(self, monkeypatch):
        """The p47/p52/p54/p56 shape: on topic, high ce, and not an answer."""
        monkeypatch.setattr(entailment, "answers_question", lambda *a, **k: False)
        high = {**META, "top_ce": 0.98}
        assert no_confident_match(high) is None
        v = no_confident_match(high) or no_entailed_answer(Q, ROWS, high)
        assert v and v["reason"] == "entailment"

    def test_a_good_answer_passes_both(self, monkeypatch):
        monkeypatch.setattr(entailment, "answers_question", lambda *a, **k: True)
        high = {**META, "top_ce": 0.98}
        assert (no_confident_match(high) or no_entailed_answer(Q, ROWS, high)) is None


class TestQueryForm:
    """The judge is asked "do these facts ANSWER the question?" — so it may
    only be shown a question.

    A bare noun phrase has no answer, and the model correctly says NO:
    measured 8 of 8 entity names abstaining with good rows and top_ce ~0.99.
    That form is what consolidate-synthesize STEP 1(b) issues all night, and
    it sits entirely outside the calibration set, whose 56 answerable probes
    are all question-mark-terminated and ten words or more.
    """

    @pytest.fixture
    def says_no(self, monkeypatch):
        calls = []
        monkeypatch.setattr(entailment, "answers_question",
                            lambda q, f, **k: (calls.append(q), False)[1])
        return calls

    @pytest.mark.parametrize("query", [
        "LightRAG", "CubeOH", "R9700", "qwen35-4b-util",
        "llama-swap reranker", "memory dense lane abstention gate",
    ])
    def test_bare_entity_queries_are_never_judged(self, says_no, query):
        assert no_entailed_answer(query, ROWS, META) is None
        assert says_no == [], "the model must not even be called"

    @pytest.mark.parametrize("query", [
        "What MMLU and GPQA scores did qwen36-27b-cron get in our battery?",
        "How worn is the NVMe drive that holds the GGUF model files?",
        "does the pdrag reranker fit on the GPU or not, and why",
    ])
    def test_real_questions_are_still_judged(self, says_no, query):
        v = no_entailed_answer(query, ROWS, META)
        assert v and v["reason"] == "entailment"
        assert says_no == [query]

    def test_the_whole_calibration_set_still_qualifies(self):
        """Guard the measured numbers: if this set stops qualifying, the
        9/10-for-1/56 figure no longer describes what ships."""
        import json
        import os

        from plugins.memory.holographic.retrieval import _is_question

        path = os.path.expanduser(
            "~/.hermes/scripts/model-battery/memory-probes.json")
        if not os.path.exists(path):
            pytest.skip("probe set not present")
        probes = json.load(open(path))["probes"]
        rejected = [p["id"] for p in probes if not _is_question(p["question"])]
        assert not rejected, f"calibration probes no longer judged: {rejected}"
