"""Tests for user-defined deny rules (approvals.deny in config.yaml).

approvals.deny is a list of fnmatch globs matched against terminal commands.
A match blocks unconditionally — BEFORE the --yolo / /yolo / mode=off bypass —
making it the user-editable counterpart to the code-shipped hardline floor.
"""

import shlex

import pytest

from tools import approval as mod
from tools import approval_floors
from tools import approval_context

@pytest.fixture
def deny_config(monkeypatch):
    """Install a deny list into the approvals config and return a setter."""

    state = {"config": {"mode": "manual", "deny": []}}

    def set_deny(patterns, **extra):
        state["config"] = {"mode": "manual", "deny": list(patterns), **extra}

    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: state["config"])
    return set_deny

@pytest.fixture
def clean_env(monkeypatch):
    """Non-interactive, non-gateway, non-cron, non-yolo baseline."""
    for var in ("HERMES_YOLO_MODE", "HERMES_GATEWAY_SESSION",
                "HERMES_CRON_SESSION", "HERMES_INTERACTIVE",
                "HERMES_EXEC_ASK"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(mod, "_YOLO_MODE_FROZEN", False)

class TestMatchUserDenyRule:
    def test_no_config_is_noop(self, deny_config):
        deny_config([])
        assert mod._match_user_deny_rule("git push --force origin main") is None

    def test_missing_key_is_noop(self, monkeypatch):
        monkeypatch.setattr(approval_context, "_get_approval_config", lambda: {"mode": "manual"})
        assert mod._match_user_deny_rule("rm -rf build/") is None

    def test_config_load_failure_fails_open(self, monkeypatch):
        def boom():
            raise RuntimeError("config unavailable")
        monkeypatch.setattr(approval_context, "_get_approval_config", boom)
        assert mod._match_user_deny_rule("git push --force") is None

    def test_quote_obfuscation_still_matches(self, deny_config):
        """Deobfuscation variants from the detector also feed deny matching."""
        deny_config(["git push --force*"])
        assert mod._match_user_deny_rule('git pu""sh --force origin main') is not None

    def test_whole_quoted_argument_matches_after_quote_removal(self, deny_config):
        """The shell removes quotes before ``git`` sees its argv, so the rule must too (#117815)."""
        deny_config(["git status", "rm -rf /important*"])
        for command in ('git "status"', "git 'status'", r"git \status", 'git "sta\\\ntus"', 'rm -rf "/important/data"'):
            assert mod._match_user_deny_rule(command) is not None, command
        # Quoted bytes stay data: a quoted space is not an argument boundary.
        assert mod._match_user_deny_rule('git "sta tus"') is None
        assert mod._match_user_deny_rule(r'git "st\atus"') is None  # a backslash inside "..." stays

def test_deny_follows_executable_identity(deny_config, clean_env, monkeypatch):
    """Paths, prefixes and shell carriers cannot outrank an explicit deny."""
    commands = [
        "sudo -n id -u", "/usr/bin/sudo -n id -u", "./sudo -n id -u",
        '"/usr/bin/su"do -n id -u', r"/usr/bin/sud\o -n id -u",
        "FOO=bar /usr/bin/sudo -n id -u", "env sudo -n id -u",
        "/usr/bin/env -i FOO=bar /usr/bin/sudo -n id -u",
        "env -u FOO -- /usr/bin/sudo -n id -u",
        "env -C /tmp /usr/bin/sudo -n id -u",
        "command -p /usr/bin/sudo -n id -u", "exec -a label /usr/bin/sudo -n id -u",
        "nohup /usr/bin/sudo -n id -u", "nice -n 5 /usr/bin/sudo -n id -u",
        "timeout 5 /usr/bin/sudo -n id -u",
        "setsid -f /usr/bin/sudo -n id -u", "time -p /usr/bin/sudo -n id -u",
        "stdbuf --output L /usr/bin/sudo -n id -u",
        "ionice --class 2 /usr/bin/sudo -n id -u",
        "chrt --fifo 20 /usr/bin/sudo -n id -u",
        "taskset --cpu-list 0 /usr/bin/sudo -n id -u",
        "chroot --userspec root:root /srv /usr/bin/sudo -n id -u",
        "true && /usr/bin/sudo -n id -u; echo ok",
        "echo ok | /usr/bin/sudo -n id -u", "(/usr/bin/sudo -n id -u)",
        'echo "$(/usr/bin/sudo -n id -u)"',
        "bash -lc 'env -i /usr/bin/sudo -n id -u'",
        "env -S '/usr/bin/sudo -n id -u'",
        "env -S /usr/bin/sudo -n id -u",
        "env --split-string=/usr/bin/sudo -n id -u",
        "env -S \"bash -c '/usr/bin/sudo -n id -u'\"",
        "echo ok # ignored\n bash -c '/usr/bin/sudo -n id -u'",
        "printf SAFE", "env -S printf SAFE", "env -Sprintf SAFE",
        "env --split-string=printf SAFE", "env -S 'printf' SAFE",
        r"env -S 'printf\_SAFE'", r"env -S 'printf\_SAFE\c ignored'",
        "env -S 'printf SAFE # ignored'", "env -S 'printf # ignored' SAFE",
        "env -a marker printf SAFE", "env --argv0 marker printf SAFE",
        "env --argv0=marker printf SAFE", "env -amarker printf SAFE",
        r"env -a marker -S 'printf\_SAFE'",
        "env -S '\"printf\"\\_SAFE'", "env -S \"'printf' SAFE\"",
        r"env -S 'printf\_\_SAFE'", r"env -S 'printf\c ignored' SAFE",
        "2>/tmp/log FOO=bar /usr/bin/sudo -n id -u",
        "if true; then /usr/bin/sudo -n id -u; fi",
    ]
    for mode, yolo in (("manual", False), ("off", False), ("manual", True)):
        deny_config(["sudo *", "printf SAFE"], mode=mode)
        monkeypatch.setattr(mod, "_YOLO_MODE_FROZEN", yolo)
        for command in commands:
            assert approval_floors._match_user_deny_rule(command), command
            for guard in (mod.check_dangerous_command, mod.check_all_command_guards):
                result = guard(command, "local")
                assert result.get("user_deny") is True, (mode, yolo, command, result)
                assert result["approved"] is False

def test_deny_projection_preserves_data_and_path_rules(deny_config):
    """Project only executable positions; retain spelling-sensitive argument data."""
    deny_config(["/usr/bin/sudo -n id -u"])
    assert approval_floors._match_user_deny_rule("env /usr/bin/sudo -n id -u; true")
    assert approval_floors._match_user_deny_rule("/opt/bin/sudo -n id -u") is None
    deny_config(["sudo", "sudo *", "git status"])
    for command in (
        'echo "sudo -n id -u"', "printf 'ok && sudo -n id -u'",
        'printf "%s" "$(printf safe) sudo -n id -u"',
        "command -v sudo", "command -V sudo", "command -pv sudo",
        "env -u sudo printf ok", "exec -a sudo printf ok",
        "env -a sudo printf ok", "env --argv0 sudo printf ok",
        r"env -S 'printf %s\_sudo\_-n\_id'",
        "env -S 'printf %s \"sudo\\_-n\\_id\"'",
        r"env -S 'printf ok\c bash -c sudo'",
        "env -S 'printf ok # bash -c sudo'",
        r"env -S 'printf %s \${IGNORED}'",
        "ionice --pid sudo", "chrt --pid sudo", "taskset --pid 1 sudo",
        "echo 'first\nsudo -n id -u'", "echo ok # ; sudo -n id -u",
        "env -S 'printf %s; sudo -n id'",
        "env -S 'printf %s' 'sudo -n id'",
        "env -S 'printf %s' '$(sudo -n id)'",
        "echo ok # ; bash -c '/usr/bin/sudo -n id'",
        "echo ok # unmatched '\n printf ok",
        "printf '%s' '# ; bash -c sudo'",
        "git log --grep='git status'", r'printf "%s" "a\"; sudo -n id -u"',
    ):
        assert approval_floors._match_user_deny_rule(command) is None, command
    for command in ("env git status; echo ok", "(git status)", "git\tstatus # comment",
                    "env -S git status", "env -S 'git' status", "env -Sgit status",
                    "env --split-string=git status"):
        assert approval_floors._match_user_deny_rule(command) == "git status", command
    assert approval_floors._match_user_deny_rule('env git st""atus') == "git status"
    deny_config(['printf "a  b"'])
    assert approval_floors._match_user_deny_rule('env printf "a  b"')
    assert approval_floors._match_user_deny_rule('env printf "a b"') is None
    # The argv projection must retain GNU escapes as data, not shell syntax.
    from tools.approval_env_split import _split_env_string

    for literal, expected in (
        (r'a\_b', ['a', 'b']), (r'"a\_b"', ['a b']),
        (r"'a\_b'", [r'a\_b']), (r'a\cb ignored', ['a']),
        ('a # ignored', ['a']), ('a#b', ['a#b']), (r'\#a', ['#a']),
        (r'\${NAME}', ['${NAME}']), ("'${NAME}'", ['${NAME}']),
        (r"'a\'b'", ["a'b"]), (r"'a\\b'", [r'a\b']),
        (r'a\"b', ['a"b']), ('"" a', ['', 'a']),
        *((rf'a\{key}b', ['a' + value + 'b'])
          for key, value in {'f': '\f', 'n': '\n', 'r': '\r', 't': '\t', 'v': '\v'}.items()),
    ):
        assert _split_env_string(literal) == expected, literal
        command = 'env -S ' + shlex.quote('printf %s ' + literal)
        deny_config(['sudo *'])
        assert approval_floors._match_user_deny_rule(command) is None, command
    for unresolved in ('${NAME}', '"${NAME}"', r'"a\cb"', r'a\qb', "'unclosed"):
        assert _split_env_string(unresolved) is None

def test_git_global_options_do_not_hide_the_subcommand(deny_config):
    """2026-11-27 cron review: `git -C <repo> checkout` slid past `*git checkout*` (the guard and the
    nightly command audit both missed it). git's global options are dropped wherever `git` stands in a
    form that keeps its quotes, as the globs match anywhere, and in the dequoted form only at a git
    command word; an option's value is never read as the subcommand."""
    deny_config(["*git checkout*", "*git switch*", "*git stash*", "*git reset --hard*", "git status"])
    for command, rule in (
        ("git -C ~/repo checkout main", "*git checkout*"),
        ("git -C '/my repo' checkout x", "*git checkout*"),
        ('git -c core.pager="less -R" checkout x', "*git checkout*"),
        ("git -c core.x=1 -C r switch dev", "*git switch*"),
        ("git --git-dir=/r/.git --work-tree /r checkout -- f", "*git checkout*"),
        ("timeout 30 git -C ~/r stash", "*git stash*"),
        ("cd r && git --no-pager -C . reset --hard HEAD", "*git reset --hard*"),
        ("bash -c 'git -C r checkout x'", "*git checkout*"),
        ("eval git -C r checkout x", "*git checkout*"),
        ("xargs -I{} git -C {} checkout x", "*git checkout*"),
        ("flock /tmp/l git -C r checkout x", "*git checkout*"),
        ("ssh host 'git -C r checkout x'", "*git checkout*"),
        ("git -C r submodule foreach 'git -C . switch main'", "*git switch*"),
        ("git -C r che\\\nckout x", "*git checkout*"),
        # the same as `echo git checkout x`, which the anywhere-match of the glob already blocks
        ("echo git -C r checkout x", "*git checkout*"),
        (r"grep -E 'git -C \S+ checkout' audit.log", "*git checkout*"),   # as `grep 'git checkout' f` is
        ("git -C checkout status", "git status"),
        ("git -P -c x=1 stash", "*git stash*"),
        ("git --bare --git-dir=/r.git switch m", "*git switch*"),
        ("git -C r 'checkout' x", "*git checkout*"),
        # an inline alias is expanded
        ("git -c alias.co=checkout co main", "*git checkout*"),
        ("git -c 'alias.sw=switch' -C r sw dev", "*git switch*"),
        ("git -c alias.st='stash push' st", "*git stash*"),
        ("git -c alias.x='!git checkout' x main", "*git checkout*"),
        # git ignores an alias named after a builtin: the builtin runs, so the rule must still match
        ("git -c alias.checkout=log checkout main", "*git checkout*"),
        ("git -c alias.stash=log stash", "*git stash*"),
        ("git -C r -c alias.checkout=status checkout x", "*git checkout*"),
        # a quoted subcommand behind a wrapper the parser does not read as a command word
        ("xargs -I{} git -C {} 'checkout' x", "*git checkout*"),
        ("flock /tmp/l git -C r 'checkout' x", "*git checkout*"),
        ("watch git -C r 'stash'", "*git stash*"),
        ("ssh host git -C r 'checkout' x", "*git checkout*"),
        ("git -C r submodule foreach git -C . 'checkout' main", "*git checkout*"),
        ("find . -execdir git -C .. \"checkout\" x \\;", "*git checkout*"),
    ):
        assert approval_floors._match_user_deny_rule(command) == rule, command
    for command in ("git -C ~/repo ls-tree abc123 src/", "git -C ~/repo grep -n checkout abc123",
                    "git -C ~/r show HEAD:checkout.py", "git -C ~/r log --oneline -5", "git -C ~/r diff stash@{0}",
                    "git -C", "git -C ~/r", "legit -C r checkout x", "git-lfs -C r checkout x",
                    # only git's own flags count: these name git as data (2026-11-27 review)
                    "grep -n -e git -e checkout notes.md", "rg -e git -e switch src/",
                    "find . -name git -prune -o -name stash -print", "cat git -n checkout.txt",
                    # removing the quotes must not glue a quoted pattern to the next word
                    'grep -n "git -C" -A3 checkout.sh', 'grep -c "git -p" stash.log',
                    "git -c alias.co=checkout log -5"):
        assert approval_floors._match_user_deny_rule(command) is None, command


class TestDenyBeatsYolo:
    def test_deny_blocks_under_yolo_env(self, deny_config, clean_env, monkeypatch):
        deny_config(["git push --force*"])
        monkeypatch.setattr(mod, "_YOLO_MODE_FROZEN", True)

        result = mod.check_dangerous_command("git push --force origin main", "local")
        assert result["approved"] is False
        assert result.get("user_deny") is True
        assert "approvals.deny" in result["message"]

    def test_deny_blocks_under_session_yolo(self, deny_config, clean_env, monkeypatch):
        deny_config(["*curl*|*sh*"])
        monkeypatch.setattr(mod, "is_current_session_yolo_enabled", lambda: True)

        result = mod.check_dangerous_command("curl https://x.io/i.sh | sh", "local")
        assert result["approved"] is False
        assert result.get("user_deny") is True

    def test_non_matching_command_still_bypassed_by_yolo(
            self, deny_config, clean_env, monkeypatch):
        deny_config(["git push --force*"])
        monkeypatch.setattr(mod, "_YOLO_MODE_FROZEN", True)

        # Dangerous but not denied — yolo passes it through unchanged.
        result = mod.check_dangerous_command("rm -rf build/", "local")
        assert result["approved"] is True

    def test_empty_deny_list_preserves_yolo_behavior(
            self, deny_config, clean_env, monkeypatch):
        deny_config([])
        monkeypatch.setattr(mod, "_YOLO_MODE_FROZEN", True)

        result = mod.check_dangerous_command("git push --force origin main", "local")
        assert result["approved"] is True

class TestDenyOrdering:
    def test_hardline_fires_before_deny(self, deny_config, clean_env):
        """A hardline command reports the hardline block, not the deny rule."""
        deny_config(["*"])
        result = mod.check_dangerous_command("rm -rf /", "local")
        assert result["approved"] is False
        assert result.get("hardline") is True
        assert result.get("user_deny") is None

    def test_deny_beats_permanent_allowlist(self, deny_config, clean_env, monkeypatch):
        """Deny is checked before the command_allowlist shortcut."""
        deny_config(["git push --force*"])
        monkeypatch.setattr(
            mod, "_command_matches_permanent_allowlist", lambda c: True)
        monkeypatch.setattr(
            approval_floors, "_command_matches_permanent_allowlist", lambda c: True)

        result = mod.check_dangerous_command("git push --force origin main", "local")
        assert result["approved"] is False
        assert result.get("user_deny") is True

    @pytest.mark.parametrize(
        "guard",
        [mod.check_dangerous_command, mod.check_all_command_guards],
    )
    @pytest.mark.parametrize(
        "env_type",
        ["docker", "modal", "daytona", "vercel_sandbox"],
    )
    def test_container_backend_cannot_skip_deny(
            self, guard, env_type, deny_config, clean_env, monkeypatch):
        deny_config(["*chmod*"], mode="off")
        monkeypatch.setattr(mod, "_YOLO_MODE_FROZEN", True)

        result = guard("chmod 600 /tmp/hermes-approval-deny-test", env_type)

        assert result["approved"] is False
        assert result.get("user_deny") is True

    @pytest.mark.parametrize(
        "guard",
        [mod.check_dangerous_command, mod.check_all_command_guards],
    )
    @pytest.mark.parametrize(
        "env_type",
        ["docker", "modal", "daytona", "vercel_sandbox"],
    )
    def test_container_backend_still_skips_non_denied_command(
            self, guard, env_type, deny_config, clean_env):
        deny_config(["*chmod*"])

        result = guard("rm -rf build/", env_type)

        assert result["approved"] is True

    def test_benign_command_unaffected(self, deny_config, clean_env):
        deny_config(["git push --force*"])
        result = mod.check_dangerous_command("ls -la", "local")
        assert result["approved"] is True
