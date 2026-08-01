"""``skills_list`` must report which skills are pinned.

The background review fork is instructed that pinned skills are off-limits to
autonomous maintenance, and ``_background_review_write_guard`` enforces that.
But ``skills_list`` returned only name/description/category, so the fork had no
way to tell which of the skills it could see were pinned. It kept selecting the
most relevant skill — which for a cron session is always one of the pinned ones
the job loaded — reading it, composing a patch, and only then being refused.
Wasted turns out of a budgeted cron run, repeatedly on the same skills.

The invariant these tests pin is not just "a flag is present" but that the flag
agrees with the guard: what the agent is shown must match what the guard will
do, or the annotation just moves the surprise somewhere else.
"""

import json

import pytest

from tools import skill_usage
from tools.skills_tool import skills_list


@pytest.fixture
def skills_home(tmp_path, monkeypatch):
    """Point both skills_tool and skill_usage at an isolated skills dir."""
    home = tmp_path / "hermes-home"
    (home / "skills").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("tools.skills_tool.SKILLS_DIR", home / "skills")
    monkeypatch.setattr("tools.skill_usage.SKILLS_DIR", home / "skills", raising=False)
    import tools.skills_tool as st
    st._SKILLS_CACHE.clear()
    yield home / "skills"
    st._SKILLS_CACHE.clear()


def _make_skill(skills_dir, name, description="A test skill."):
    d = skills_dir / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n\nBody.\n"
    )


def _by_name(payload):
    return {s["name"]: s for s in json.loads(payload)["skills"]}


# Names deliberately avoid the substring "pinned": a skill called
# "unpinned-one" makes `"pinned" in error` match the wrong refusal branch.
LOCKED = "locked-skill"
OPEN = "open-skill"

# The distinctive phrase from the pin branch of _background_review_write_guard;
# its other branches (not curator-managed, external dir, protected builtin)
# refuse for different reasons and must not be conflated with a pin.
PIN_REFUSAL = "pinned skills are off-limits"


def test_pinned_skill_is_flagged(skills_home):
    _make_skill(skills_home, LOCKED)
    _make_skill(skills_home, OPEN)
    skill_usage.set_pinned(LOCKED, True)

    skills = _by_name(skills_list())
    assert skills[LOCKED].get("pinned") is True


def test_unpinned_skill_carries_no_flag(skills_home):
    """Absent, not false — a false flag on every row is pure token cost."""
    _make_skill(skills_home, OPEN)
    skill_usage.set_pinned(OPEN, False)

    skills = _by_name(skills_list())
    assert "pinned" not in skills[OPEN]


def test_flag_matches_the_write_guard_decision(skills_home, monkeypatch):
    """The reported flag must predict whether a background write is refused."""
    from tools.skill_manager_tool import _background_review_write_guard

    _make_skill(skills_home, LOCKED)
    _make_skill(skills_home, OPEN)
    skill_usage.set_pinned(LOCKED, True)
    # The created_by=agent marker is what makes a skill curator-managed;
    # without it the guard refuses as user-owned and the pin path is never
    # reached, which would make the agreement assertion below vacuous.
    skill_usage.mark_agent_created(OPEN)
    skill_usage.set_pinned(OPEN, False)

    monkeypatch.setattr("tools.skill_provenance.is_background_review", lambda: True)

    skills = _by_name(skills_list())
    assert set(skills) == {LOCKED, OPEN}
    for name, entry in skills.items():
        refusal = _background_review_write_guard(name, skills_home / name, "patch")
        refused_for_pin = bool(refusal) and PIN_REFUSAL in refusal.get("error", "")
        assert entry.get("pinned", False) == refused_for_pin, (
            f"{name}: skills_list says pinned={entry.get('pinned', False)} but "
            f"the guard {'refuses' if refused_for_pin else 'allows'} a patch"
        )

    # And the unpinned one must genuinely be writable, not merely refused for
    # some other reason — otherwise the agreement above is vacuous.
    assert _background_review_write_guard(OPEN, skills_home / OPEN, "patch") is None


def test_hint_mentions_pinned_when_any_are_pinned(skills_home):
    _make_skill(skills_home, LOCKED)
    skill_usage.set_pinned(LOCKED, True)

    payload = json.loads(skills_list())
    assert "pinned" in payload["hint"]


def test_usage_file_is_read_once_not_once_per_skill(skills_home, monkeypatch):
    """get_record() re-parses .usage.json on every call; over ~100 skills that
    dominated the cost of skills_list. Annotation must load the map once."""
    for i in range(12):
        _make_skill(skills_home, f"skill-{i:02d}")
    skill_usage.set_pinned("skill-00", True)

    calls = []
    real = skill_usage.load_usage
    monkeypatch.setattr(
        skill_usage, "load_usage", lambda: (calls.append(1), real())[1]
    )

    payload = json.loads(skills_list())
    assert payload["count"] == 12
    assert len(calls) == 1, f"read .usage.json {len(calls)}x for 12 skills"


def test_hint_unchanged_when_nothing_is_pinned(skills_home):
    _make_skill(skills_home, OPEN)
    skill_usage.set_pinned(OPEN, False)

    payload = json.loads(skills_list())
    assert "pinned" not in payload["hint"]


def test_unpinning_clears_the_flag(skills_home):
    _make_skill(skills_home, "toggle-me")
    skill_usage.set_pinned("toggle-me", True)
    assert _by_name(skills_list())["toggle-me"].get("pinned") is True

    # Pin state lives in .usage.json, which is not part of the skills-scan
    # cache signature — a stale flag here would re-create the original bug.
    skill_usage.set_pinned("toggle-me", False)
    assert "pinned" not in _by_name(skills_list())["toggle-me"]
