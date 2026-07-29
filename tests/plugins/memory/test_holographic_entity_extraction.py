"""Regression tests for entity extraction in the holographic store.

Two silent defects — they corrupted the ``entities`` table rather than raising,
so every other holographic test passed while the store filled with garbage:

1. The old single-quote rule ``r"'([^']+)'"`` treated the APOSTROPHE in
   possessives and contractions as an opening quote, capturing everything up to
   the next apostrophe anywhere later in the text. On a real corpus this
   produced multi-hundred-character prose blobs, most of them beginning with the
   literal ``"s "``. Boundary-lookaround repairs still leak on ``.'`` and ``)'``
   (``et al.'s``), so the rule was removed rather than patched.

2. ``_resolve_entity`` looked candidates up with ``name LIKE ?``, so ``_`` and
   ``%`` inside a candidate acted as LIKE WILDCARDS: ``"model_spec"`` resolved
   to whichever row the index scan reached first — a *different* entity — and
   ``_compute_hrr_vector`` then encoded that wrong name into the fact vector.
"""

import pytest

from plugins.memory.holographic.store import _ENTITY_MAX_LEN, MemoryStore

# Fragments produced by a quote span that opened on an apostrophe always begin
# with the tail of the contraction/possessive that opened them.
_CONTRACTION_REMNANTS = ("s ", "t ", "ll ", "re ", "ve ", "d ", "m ")


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path / "memory_store.db")
    yield s
    s.close()


class TestApostropheIsNotAQuote:
    @pytest.mark.parametrize(
        "text",
        [
            "Earth's gravity field is measured by GRACE-FO and don't forget LARES-2.",
            "The Zernike polynomials don't commute with Doodson's tidal arguments.",
            "It's the store's fault that we can't dedupe entities properly.",
            "We'll ship it once the cron won't stall on the numpy check.",
            # The three shapes that defeat a boundary-lookaround repair: '.', ')'
            # and ']' are not word characters, so they still open a span.
            "Costin et al.'s theorem matches the students' data.",
            "Vol. 3.'s appendix contradicts the satellites' orbits.",
            "(Earth)'s field and the sensors' noise floor.",
        ],
    )
    def test_no_candidate_is_a_contraction_remnant(self, store, text):
        offenders = [
            name
            for name in store._extract_entities(text)
            if name.lower().startswith(_CONTRACTION_REMNANTS)
        ]
        assert offenders == [], f"apostrophe opened a quote span: {offenders}"

    def test_prose_is_never_captured_as_an_entity(self, store):
        text = (
            "Earth's oblateness variation is climate-driven, so the Brillouin "
            "sphere theorem is now rigorously settled for the exterior domain, "
            "and we don't expect any of it to change the ice-sheet conclusions "
            "that the reprocessing campaign already published."
        )
        for name in store._extract_entities(text):
            assert len(name) <= _ENTITY_MAX_LEN, f"prose blob: {name!r}"

    def test_apostrophe_pollution_never_reaches_the_entities_table(self, store):
        store.add_fact("Earth's gravity field and don't forget the tides.")
        names = [r["name"] for r in store._conn.execute("SELECT name FROM entities")]
        assert not [n for n in names if n.lower().startswith(_CONTRACTION_REMNANTS)]


class TestValuableEntitiesSurvive:
    """Guard against an over-aggressive filter. These must keep extracting.

    The original rules were structurally blind to every one of these: a bare
    acronym, a CamelCase token and a hyphenated name all fail
    ``[A-Z][a-z]+(\\s+[A-Z][a-z]+)+``, so facts full of them yielded nothing.
    """

    @pytest.mark.parametrize(
        "text, expected",
        [
            ('The "GRACE-FO" mission superseded "LARES-2".', ["GRACE-FO", "LARES-2"]),
            ("Propagate with SGP4 before the COSMIC-2 comparison.", ["SGP4", "COSMIC-2"]),
            (
                "WidgetV6 feeds the pipeline via LightRAG on GitHub.",
                ["WidgetV6", "LightRAG", "GitHub"],
            ),
            ("See arXiv:2607.19083 for the derivation.", ["arXiv:2607.19083"]),
            ("Claude Code wrote the Gauss-Newton solver.", ["Claude Code", "Gauss-Newton"]),
        ],
    )
    def test_identifiers_and_phrases_extract(self, store, text, expected):
        names = store._extract_entities(text)
        for want in expected:
            assert want in names, f"{want!r} lost from {names!r}"

    def test_aka_rule_still_fires(self, store):
        assert "Guido" in store._extract_entities("Guido aka BDFL wrote it.")


class TestFragmentsAndNoiseAreRejected:
    @pytest.mark.parametrize(
        "text",
        [
            # _RE_CAPITALIZED shredding a phrase at an interior function word.
            "The Zernike basis is orthogonal.",
            "Uses Telemetry downstream.",
            # Hyphenated adjectives are not identifiers.
            "A real-time, long-running, climate-driven pipeline.",
            # ALL-CAPS section markers, which a bare [A-Z]{2,} rule would make
            # the highest-degree nodes in the whole graph.
            "PAPER CONFIRMED LESSON VERDICT ACTIVITY: nothing here.",
        ],
    )
    def test_no_entities_extracted(self, store, text):
        assert store._extract_entities(text) == []

    def test_quoted_error_string_is_not_an_entity(self, store):
        text = (
            'Job failed: "HTTP 400: Cannot have 2 or more assistant messages '
            'at the end of the list".'
        )
        assert all(len(n) <= _ENTITY_MAX_LEN for n in store._extract_entities(text))


class TestResolveEntityIsNotAPattern:
    def test_underscore_is_not_a_single_char_wildcard(self, store):
        first = store._resolve_entity("model_spec")
        store._resolve_entity("modelXspec")  # decoy matched by the '_' wildcard
        assert store._resolve_entity("model_spec") == first

    def test_percent_is_not_a_multi_char_wildcard(self, store):
        first = store._resolve_entity("100% recall")
        store._resolve_entity("100 percent recall")  # decoy matched by '%'
        assert store._resolve_entity("100% recall") == first

    def test_resolution_stays_case_insensitive(self, store):
        # LIKE gave this for free; a bare '=' would silently fork every entity
        # by capitalisation. The fix must keep COLLATE NOCASE.
        assert store._resolve_entity("LightRAG") == store._resolve_entity("lightrag")

    def test_same_name_never_creates_a_second_row(self, store):
        first = store._resolve_entity("GRACE-FO")
        assert store._resolve_entity("GRACE-FO") == first
        count = store._conn.execute(
            "SELECT COUNT(*) FROM entities WHERE name = 'GRACE-FO'"
        ).fetchone()[0]
        assert count == 1
