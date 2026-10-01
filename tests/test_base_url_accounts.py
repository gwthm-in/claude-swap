"""Tests for base-URL (custom endpoint: relay / gateway) account support.

Covers ``add-token --base-url`` validation and kind selection, the roster's
``baseUrl`` field across both add-token paths, the global switch's
ownership-tracked ``env.ANTHROPIC_BASE_URL`` in Claude Code's settings.json
(set / change / reset to Anthropic's default, user-owned values, symlinks,
rollback), the restart note on an OAuth / API-key change, remove/purge
cleanup, session mode (``cswap run``) env and ``--settings`` pinning, the
"custom endpoint" usage sentinel with no Anthropic calls, and the JSON /
list / export-import surfaces.
"""

from __future__ import annotations

import builtins
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_swap import cli
from claude_swap import session as session_mod
from claude_swap.claude_settings import ANTHROPIC_API_URL, BASE_URL_ENV
from claude_swap.credentials import approved_form
from claude_swap.exceptions import SessionError, SwitchError, ValidationError
from claude_swap.json_output import USAGE_CUSTOM_ENDPOINT, account_row, usage_fields
from claude_swap.models import Platform, base_url_host, normalize_base_url
from claude_swap.paths import (
    get_claude_settings_path,
    get_credentials_path,
    get_global_config_path,
)
from claude_swap.session import SessionManager, session_dir_for
from claude_swap.switcher import ClaudeAccountSwitcher
from claude_swap.transfer import export_accounts, import_accounts

URL = "https://relay.example.com"
OTHER_URL = "https://relay-two.example.com:8443/api"
POOL_KEY = "pool_" + "k3y" * 12
OTHER_POOL_KEY = "pool_" + "n3w" * 12
API_KEY = "sk-ant-api03-" + "a1b2c3d4e5" * 4
SETUP_TOKEN = "sk-ant-oat01-" + "s" * 30


def _switcher() -> ClaudeAccountSwitcher:
    s = ClaudeAccountSwitcher()
    s.platform = Platform.LINUX
    s._setup_directories()
    s._init_sequence_file()
    return s


def _record(s: ClaudeAccountSwitcher, num: str) -> dict:
    return s._get_sequence_data()["accounts"][num]


def _settings() -> dict:
    path = get_claude_settings_path()
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _settings_base_url() -> str | None:
    return (_settings().get("env") or {}).get(BASE_URL_ENV)


def _seed_pair(s: ClaudeAccountSwitcher) -> None:
    """Slot 1: plain OAuth setup-token (active). Slot 2: relay key + URL."""
    s.add_account_from_token(SETUP_TOKEN, slot=1)
    s.add_account_from_token(POOL_KEY, slot=2, base_url=URL)
    s.switch_to("1", json_output=True)


# ---------------------------------------------------------------------------
# URL validation
# ---------------------------------------------------------------------------


class TestNormalizeBaseUrl:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("https://relay.example.com", "https://relay.example.com"),
            ("https://relay.example.com/", "https://relay.example.com"),
            ("  http://10.0.0.5:8080/proxy//  ", "http://10.0.0.5:8080/proxy"),
            ("HTTPS://Relay.example.com", "HTTPS://Relay.example.com"),
        ],
    )
    def test_accepts_and_strips_trailing_slash(self, raw, expected):
        assert normalize_base_url(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "relay.example.com",
            "ftp://relay.example.com",
            "https://",
            "https:///path",
            "https://user:pw@relay.example.com",
            "https://relay.example.com/?x=1",
            "https://relay.example.com/#frag",
            "https://relay .example.com",
            "https://relay.example.com:notaport",
        ],
    )
    def test_rejects(self, raw):
        with pytest.raises(ValueError):
            normalize_base_url(raw)

    def test_host_for_display(self):
        assert base_url_host(OTHER_URL) == "relay-two.example.com:8443"
        assert base_url_host("") == ""

    def test_add_token_rejects_invalid_url(self, temp_home: Path):
        s = _switcher()
        with pytest.raises(ValidationError, match="http:// or https://"):
            s.add_account_from_token(POOL_KEY, base_url="relay.example.com")
        assert s._get_sequence_data()["accounts"] == {}


class TestCli:
    def test_base_url_forwarded(self, temp_home: Path):
        with patch.object(
            sys, "argv",
            ["cswap", "add-token", "pool_x", "--base-url", "https://r.example.com/"],
        ), patch.object(ClaudeAccountSwitcher, "add_account_from_token") as add:
            cli.main()
        add.assert_called_once_with(
            token="pool_x", email=None, slot=None, base_url="https://r.example.com"
        )

    def test_base_url_only_with_add_token(self, capsys):
        with patch.object(sys, "argv", ["cswap", "list", "--base-url", URL]):
            with pytest.raises(SystemExit) as exc:
                cli.main()
        assert exc.value.code == 2
        assert "--base-url can only be used with 'add-token'" in capsys.readouterr().err

    def test_invalid_url_rejected_before_token_prompt(self, capsys):
        with patch.object(
            sys, "argv", ["cswap", "add-token", "--base-url", "ftp://x"]
        ), patch("getpass.getpass", side_effect=AssertionError("prompted")):
            with pytest.raises(SystemExit) as exc:
                cli.main()
        assert exc.value.code == 2
        assert "--base-url" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Kind selection + roster record
# ---------------------------------------------------------------------------


class TestKindSelection:
    def test_non_anthropic_key_with_base_url_is_api_key(self, temp_home: Path):
        s = _switcher()
        s.add_account_from_token(POOL_KEY, base_url=URL + "/")
        rec = _record(s, "1")
        assert rec["kind"] == "api_key"
        assert rec["baseUrl"] == URL
        assert rec["email"] == "api-key-1@token.local"
        assert s._read_account_credentials("1", rec["email"]) == POOL_KEY

    def test_setup_token_with_base_url_stays_setup_token(self, temp_home: Path):
        s = _switcher()
        s.add_account_from_token(SETUP_TOKEN, base_url=URL)
        rec = _record(s, "1")
        assert "kind" not in rec
        assert rec["baseUrl"] == URL
        assert rec["email"] == "setup-token-1@token.local"
        blob = json.loads(s._read_account_credentials("1", rec["email"]))
        assert blob["claudeAiOauth"]["accessToken"] == SETUP_TOKEN

    def test_without_base_url_detection_unchanged(self, temp_home: Path):
        s = _switcher()
        s.add_account_from_token(POOL_KEY)
        rec = _record(s, "1")
        assert "kind" not in rec and "baseUrl" not in rec
        assert rec["email"] == "setup-token-1@token.local"

    def test_stdin_token(self, temp_home: Path, monkeypatch):
        import io

        s = _switcher()
        monkeypatch.setattr(sys, "stdin", io.StringIO(POOL_KEY + "\n"))
        s.add_account_from_token("-", email="relay@local.dev", slot=5, base_url=URL)
        rec = _record(s, "5")
        assert rec["email"] == "relay@local.dev"
        assert rec["kind"] == "api_key" and rec["baseUrl"] == URL

    def test_no_network_during_add(self, temp_home: Path, monkeypatch):
        monkeypatch.setattr(
            "claude_swap.oauth.fetch_oauth_profile",
            lambda token: pytest.fail("profile lookup during add-token"),
        )
        s = _switcher()
        s.add_account_from_token(POOL_KEY, base_url=URL)

    def test_key_never_printed(self, temp_home: Path, capsys):
        s = _switcher()
        s.add_account_from_token(POOL_KEY, base_url=URL)
        out = capsys.readouterr().out
        assert POOL_KEY not in out
        assert "relay.example.com" in out


class TestRefreshInPlace:
    def test_updates_base_url(self, temp_home: Path):
        s = _switcher()
        s.add_account_from_token(POOL_KEY, email="relay@local.dev", base_url=URL)
        s.add_account_from_token(OTHER_POOL_KEY, email="relay@local.dev", base_url=OTHER_URL)
        data = s._get_sequence_data()
        assert list(data["accounts"]) == ["1"]
        assert data["accounts"]["1"]["baseUrl"] == OTHER_URL
        assert s._read_account_credentials("1", "relay@local.dev") == OTHER_POOL_KEY

    def test_omitted_base_url_keeps_existing_and_kind(self, temp_home: Path):
        s = _switcher()
        s.add_account_from_token(POOL_KEY, email="relay@local.dev", base_url=URL)
        s.add_account_from_token(OTHER_POOL_KEY, email="relay@local.dev")
        rec = _record(s, "1")
        assert rec["baseUrl"] == URL and rec["kind"] == "api_key"
        assert s._read_account_credentials("1", "relay@local.dev") == OTHER_POOL_KEY

    def test_empty_base_url_clears(self, temp_home: Path):
        s = _switcher()
        s.add_account_from_token(API_KEY, email="k@local.dev", base_url=URL)
        s.add_account_from_token(API_KEY, email="k@local.dev", base_url="")
        rec = _record(s, "1")
        assert "baseUrl" not in rec
        assert rec["kind"] == "api_key"

    def test_default_label_slot_readd_keeps_endpoint(self, temp_home: Path):
        s = _switcher()
        s.add_account_from_token(POOL_KEY, slot=4, base_url=URL)
        s.add_account_from_token(OTHER_POOL_KEY, slot=4, assume_yes=True)
        rec = _record(s, "4")
        assert rec["email"] == "api-key-4@token.local"
        assert rec["baseUrl"] == URL and rec["kind"] == "api_key"


class TestSlotRebuild:
    def test_same_slot_rebuild_preserves_alias_and_base_url(self, temp_home: Path):
        s = _switcher()
        s.add_account_from_token(POOL_KEY, email="relay@local.dev", slot=3, base_url=URL)
        s.set_alias("3", "relay")
        s.add_account_from_token(OTHER_POOL_KEY, email="relay@local.dev", slot=3)
        rec = _record(s, "3")
        assert rec["alias"] == "relay"
        assert rec["baseUrl"] == URL
        assert rec["kind"] == "api_key"

    def test_move_to_new_slot_preserves_alias_and_base_url(self, temp_home: Path):
        s = _switcher()
        s.add_account_from_token(POOL_KEY, email="relay@local.dev", slot=3, base_url=URL)
        s.set_alias("3", "relay")
        s.add_account_from_token(POOL_KEY, email="relay@local.dev", slot=6)
        data = s._get_sequence_data()
        assert "3" not in data["accounts"]
        assert data["accounts"]["6"]["alias"] == "relay"
        assert data["accounts"]["6"]["baseUrl"] == URL

    def test_rebuild_can_override_base_url(self, temp_home: Path):
        s = _switcher()
        s.add_account_from_token(POOL_KEY, email="relay@local.dev", slot=3, base_url=URL)
        s.add_account_from_token(POOL_KEY, email="relay@local.dev", slot=3, base_url=OTHER_URL)
        assert _record(s, "3")["baseUrl"] == OTHER_URL


# ---------------------------------------------------------------------------
# Global switch: settings.json env.ANTHROPIC_BASE_URL with ownership
# ---------------------------------------------------------------------------


class TestGlobalSwitch:
    def test_switch_sets_then_resets_to_default(self, temp_home: Path):
        s = _switcher()
        _seed_pair(s)
        assert _settings_base_url() is None

        result = s.switch_to("2", json_output=True)
        assert result["switched"] is True
        assert result["baseUrlChanged"] is True
        assert _settings_base_url() == URL
        assert s._get_sequence_data()["managedBaseUrl"] == URL
        # The relay key is active on Claude Code's API-key axis.
        cfg = json.loads(get_global_config_path().read_text(encoding="utf-8"))
        assert cfg["primaryApiKey"] == POOL_KEY
        assert approved_form(POOL_KEY) in cfg["customApiKeyResponses"]["approved"]
        assert not get_credentials_path().exists()

        # Not removed: a running session keeps the last URL it saw when the
        # key disappears, so cswap writes (and owns) Anthropic's default.
        result = s.switch_to("1", json_output=True)
        assert result["baseUrlChanged"] is True
        assert _settings() == {"env": {BASE_URL_ENV: ANTHROPIC_API_URL}}
        assert s._get_sequence_data()["managedBaseUrl"] == ANTHROPIC_API_URL
        cfg = json.loads(get_global_config_path().read_text(encoding="utf-8"))
        assert "primaryApiKey" not in cfg
        live = json.loads(get_credentials_path().read_text(encoding="utf-8"))
        assert live["claudeAiOauth"]["accessToken"] == SETUP_TOKEN
        # The relay slot's backup survived the round trip.
        assert s._read_account_credentials("2", "api-key-2@token.local") == POOL_KEY

    def test_managed_default_replaced_by_later_base_url_switch(
        self, temp_home: Path
    ):
        s = _switcher()
        _seed_pair(s)
        s.switch_to("2", json_output=True)
        s.switch_to("1", json_output=True)
        assert _settings_base_url() == ANTHROPIC_API_URL
        result = s.switch_to("2", json_output=True)
        assert result["baseUrlChanged"] is True
        assert _settings_base_url() == URL
        assert s._get_sequence_data()["managedBaseUrl"] == URL

    def test_normal_to_normal_keeps_managed_default(self, temp_home: Path):
        s = _switcher()
        _seed_pair(s)
        s.add_account_from_token("sk-ant-oat01-other", slot=3)
        s.switch_to("2", json_output=True)
        s.switch_to("1", json_output=True)
        before = get_claude_settings_path().read_text(encoding="utf-8")
        result = s.switch_to("3", json_output=True)
        assert "baseUrlChanged" not in result
        assert result["warnings"] == []
        assert get_claude_settings_path().read_text(encoding="utf-8") == before
        assert s._get_sequence_data()["managedBaseUrl"] == ANTHROPIC_API_URL

    def test_no_write_when_never_set(self, temp_home: Path):
        s = _switcher()
        _seed_pair(s)
        s.add_account_from_token("sk-ant-oat01-other", slot=3)
        path = get_claude_settings_path()
        path.write_text(json.dumps({"model": "opus"}), encoding="utf-8")
        before = path.read_text(encoding="utf-8")
        result = s.switch_to("3", json_output=True)
        assert "baseUrlChanged" not in result
        assert path.read_text(encoding="utf-8") == before
        assert "managedBaseUrl" not in s._get_sequence_data()

    def test_user_removed_managed_value_not_rewritten(self, temp_home: Path):
        s = _switcher()
        _seed_pair(s)
        s.switch_to("2", json_output=True)
        path = get_claude_settings_path()
        path.write_text(json.dumps({"model": "opus"}), encoding="utf-8")
        before = path.read_text(encoding="utf-8")
        result = s.switch_to("1", json_output=True)
        assert "baseUrlChanged" not in result
        assert path.read_text(encoding="utf-8") == before
        assert "managedBaseUrl" not in s._get_sequence_data()

    def test_switch_between_relays_changes_value(self, temp_home: Path):
        s = _switcher()
        _seed_pair(s)
        s.add_account_from_token(OTHER_POOL_KEY, slot=3, base_url=OTHER_URL)
        s.switch_to("2", json_output=True)
        result = s.switch_to("3", json_output=True)
        assert result["baseUrlChanged"] is True
        assert "restartRequired" not in result
        assert _settings_base_url() == OTHER_URL
        assert s._get_sequence_data()["managedBaseUrl"] == OTHER_URL

    def test_plain_switch_has_no_additive_fields(self, temp_home: Path):
        s = _switcher()
        s.add_account_from_token(SETUP_TOKEN, slot=1)
        s.add_account_from_token("sk-ant-oat01-other", slot=2)
        s.switch_to("1", json_output=True)
        result = s.switch_to("2", json_output=True)
        assert "baseUrlChanged" not in result and "restartRequired" not in result
        assert not get_claude_settings_path().exists()

    def test_other_settings_preserved(self, temp_home: Path):
        s = _switcher()
        _seed_pair(s)
        path = get_claude_settings_path()
        path.write_text(
            json.dumps({"model": "opus", "env": {"FOO": "1"}}, indent=2) + "\n",
            encoding="utf-8",
        )
        s.switch_to("2", json_output=True)
        assert _settings() == {"model": "opus", "env": {"FOO": "1", BASE_URL_ENV: URL}}
        s.switch_to("1", json_output=True)
        assert _settings() == {
            "model": "opus", "env": {"FOO": "1", BASE_URL_ENV: ANTHROPIC_API_URL},
        }

    def test_user_value_refuses_base_url_switch(self, temp_home: Path):
        s = _switcher()
        _seed_pair(s)
        path = get_claude_settings_path()
        path.write_text(
            json.dumps({"env": {BASE_URL_ENV: "https://mine.example.com"}}),
            encoding="utf-8",
        )
        before = path.read_text(encoding="utf-8")
        with pytest.raises(SwitchError, match="cswap did not write it"):
            s.switch_to("2", json_output=True)
        assert path.read_text(encoding="utf-8") == before
        assert s._get_sequence_data()["activeAccountNumber"] == 1
        assert "managedBaseUrl" not in s._get_sequence_data()
        live = json.loads(get_credentials_path().read_text(encoding="utf-8"))
        assert live["claudeAiOauth"]["accessToken"] == SETUP_TOKEN

    def test_user_value_left_alone_on_normal_switch(self, temp_home: Path):
        s = _switcher()
        s.add_account_from_token(SETUP_TOKEN, slot=1)
        s.add_account_from_token("sk-ant-oat01-other", slot=2)
        s.switch_to("1", json_output=True)
        path = get_claude_settings_path()
        path.write_text(
            json.dumps({"env": {BASE_URL_ENV: "https://mine.example.com"}}),
            encoding="utf-8",
        )
        before = path.read_text(encoding="utf-8")
        result = s.switch_to("2", json_output=True)
        assert path.read_text(encoding="utf-8") == before
        assert any("not set by cswap" in w for w in result["warnings"])
        assert "baseUrlChanged" not in result

    @pytest.mark.parametrize(
        "default", [ANTHROPIC_API_URL, ANTHROPIC_API_URL + "/"]
    )
    def test_user_set_default_left_alone_without_warning(
        self, temp_home: Path, default
    ):
        s = _switcher()
        s.add_account_from_token(SETUP_TOKEN, slot=1)
        s.add_account_from_token("sk-ant-oat01-other", slot=2)
        s.switch_to("1", json_output=True)
        path = get_claude_settings_path()
        path.write_text(
            json.dumps({"env": {BASE_URL_ENV: default}}), encoding="utf-8"
        )
        before = path.read_text(encoding="utf-8")
        result = s.switch_to("2", json_output=True)
        assert path.read_text(encoding="utf-8") == before
        assert result["warnings"] == []
        assert "managedBaseUrl" not in s._get_sequence_data()

    @pytest.mark.parametrize(
        "default",
        [ANTHROPIC_API_URL, ANTHROPIC_API_URL + "/", "HTTPS://API.Anthropic.com"],
    )
    def test_unmarked_default_is_taken_over(self, temp_home: Path, default):
        s = _switcher()
        _seed_pair(s)
        get_claude_settings_path().write_text(
            json.dumps({"model": "opus", "env": {BASE_URL_ENV: default}}),
            encoding="utf-8",
        )
        result = s.switch_to("2", json_output=True)
        assert result["baseUrlChanged"] is True
        assert _settings() == {"model": "opus", "env": {BASE_URL_ENV: URL}}
        assert s._get_sequence_data()["managedBaseUrl"] == URL
        s.switch_to("1", json_output=True)
        assert _settings_base_url() == ANTHROPIC_API_URL
        assert s._get_sequence_data()["managedBaseUrl"] == ANTHROPIC_API_URL

    @pytest.mark.parametrize(
        "foreign",
        ["https://api.anthropic.com.evil.example", "https://api.anthropic.com/v2"],
    )
    def test_near_default_value_still_refused(self, temp_home: Path, foreign):
        s = _switcher()
        _seed_pair(s)
        path = get_claude_settings_path()
        path.write_text(json.dumps({"env": {BASE_URL_ENV: foreign}}), encoding="utf-8")
        before = path.read_text(encoding="utf-8")
        with pytest.raises(SwitchError, match="cswap did not write it"):
            s.switch_to("2", json_output=True)
        assert path.read_text(encoding="utf-8") == before
        assert "managedBaseUrl" not in s._get_sequence_data()

    def test_user_replaced_managed_value_is_not_removed(self, temp_home: Path):
        s = _switcher()
        _seed_pair(s)
        s.switch_to("2", json_output=True)
        path = get_claude_settings_path()
        path.write_text(
            json.dumps({"env": {BASE_URL_ENV: "https://mine.example.com"}}),
            encoding="utf-8",
        )
        s.switch_to("1", json_output=True)
        assert _settings_base_url() == "https://mine.example.com"
        assert "managedBaseUrl" not in s._get_sequence_data()

    def test_unreadable_settings_refuses_base_url_switch(self, temp_home: Path):
        s = _switcher()
        _seed_pair(s)
        path = get_claude_settings_path()
        path.write_text('{"env": {', encoding="utf-8")
        with pytest.raises(SwitchError, match="cannot be used"):
            s.switch_to("2", json_output=True)
        assert path.read_text(encoding="utf-8") == '{"env": {'
        assert s._get_sequence_data()["activeAccountNumber"] == 1

    def test_symlinked_settings_written_through(self, temp_home: Path):
        if sys.platform == "win32":
            pytest.skip("symlinks need privileges on Windows")
        s = _switcher()
        _seed_pair(s)
        dotfiles = temp_home / "dotfiles"
        dotfiles.mkdir()
        target = dotfiles / "settings.json"
        target.write_text(json.dumps({"theme": "dark"}), encoding="utf-8")
        os.chmod(target, 0o644)
        link = get_claude_settings_path()
        link.symlink_to(target)
        claude_dir_mode = os.stat(link.parent).st_mode

        s.switch_to("2", json_output=True)
        assert link.is_symlink() and link.resolve() == target.resolve()
        assert json.loads(target.read_text(encoding="utf-8")) == {
            "theme": "dark", "env": {BASE_URL_ENV: URL},
        }
        assert os.stat(target).st_mode & 0o777 == 0o644
        assert os.stat(link.parent).st_mode == claude_dir_mode

        s.switch_to("1", json_output=True)
        assert link.is_symlink()
        assert json.loads(target.read_text(encoding="utf-8")) == {
            "theme": "dark", "env": {BASE_URL_ENV: ANTHROPIC_API_URL},
        }


class TestRestartNote:
    """URL changes apply live; only an OAuth <-> API-key change may not."""

    def test_oauth_to_api_key_and_back_flags_restart(self, temp_home: Path):
        s = _switcher()
        _seed_pair(s)
        assert s.switch_to("2", json_output=True)["restartRequired"] is True
        assert s.switch_to("1", json_output=True)["restartRequired"] is True

    def test_plain_api_key_without_base_url_flags_restart(self, temp_home: Path):
        s = _switcher()
        s.add_account_from_token(SETUP_TOKEN, slot=1)
        s.add_account_from_token(API_KEY, slot=2)
        s.switch_to("1", json_output=True)
        result = s.switch_to("2", json_output=True)
        assert result["restartRequired"] is True
        assert "baseUrlChanged" not in result

    def test_url_change_without_kind_change_no_restart(self, temp_home: Path):
        s = _switcher()
        s.add_account_from_token(SETUP_TOKEN, slot=1)
        s.add_account_from_token(SETUP_TOKEN + "x", slot=2, base_url=URL)
        s.switch_to("1", json_output=True)
        result = s.switch_to("2", json_output=True)
        assert result["baseUrlChanged"] is True
        assert "restartRequired" not in result
        result = s.switch_to("1", json_output=True)
        assert result["baseUrlChanged"] is True
        assert "restartRequired" not in result

    def test_direct_activation_flags_kind_change(self, temp_home: Path):
        s = _switcher()
        _seed_pair(s)
        result = s.switch_to("2", json_output=True, force=True)
        assert result["restartRequired"] is True

    def test_human_kind_change_warns_restart(self, temp_home: Path, capsys):
        s = _switcher()
        _seed_pair(s)
        capsys.readouterr()
        s.switch_to("2")
        out = capsys.readouterr()
        text = out.out + out.err
        assert "endpoint changed (now relay.example.com)" in text
        assert "next request" in text
        assert "may need a restart" in text
        assert "until restarted" not in text
        assert "no restart needed" not in text

    def test_human_url_change_only_no_restart_warning(
        self, temp_home: Path, capsys
    ):
        s = _switcher()
        s.add_account_from_token(SETUP_TOKEN, slot=1)
        s.add_account_from_token(SETUP_TOKEN + "x", slot=2, base_url=URL)
        s.switch_to("2", json_output=True)
        capsys.readouterr()
        s.switch_to("1")
        out = capsys.readouterr()
        text = out.out + out.err
        assert "endpoint changed (back to api.anthropic.com)" in text
        assert "may need a restart" not in text


class TestSwitchRollback:
    def test_normal_branch_restores_settings(self, temp_home: Path, monkeypatch):
        s = _switcher()
        _seed_pair(s)
        path = get_claude_settings_path()
        path.write_text(json.dumps({"model": "opus"}), encoding="utf-8")

        def boom(data, value):
            raise OSError("disk full")

        monkeypatch.setattr(s, "_set_managed_base_url_marker", boom)
        with pytest.raises(SwitchError, match="rolled back"):
            s.switch_to("2", json_output=True)
        # The key was absent, but removing it again would leave running
        # sessions on the failed attempt's URL: the default is written.
        assert _settings() == {
            "model": "opus", "env": {BASE_URL_ENV: ANTHROPIC_API_URL},
        }
        assert s._get_sequence_data()["managedBaseUrl"] == ANTHROPIC_API_URL
        live = json.loads(get_credentials_path().read_text(encoding="utf-8"))
        assert live["claudeAiOauth"]["accessToken"] == SETUP_TOKEN
        assert s._get_sequence_data()["activeAccountNumber"] == 1

    def test_rollback_restores_managed_default_exactly(
        self, temp_home: Path, monkeypatch
    ):
        s = _switcher()
        _seed_pair(s)
        s.switch_to("2", json_output=True)
        s.switch_to("1", json_output=True)

        def boom(data, value):
            raise OSError("disk full")

        monkeypatch.setattr(s, "_set_managed_base_url_marker", boom)
        with pytest.raises(SwitchError, match="rolled back"):
            s.switch_to("2", json_output=True)
        assert _settings_base_url() == ANTHROPIC_API_URL
        assert s._get_sequence_data()["managedBaseUrl"] == ANTHROPIC_API_URL
        assert s._get_sequence_data()["activeAccountNumber"] == 1

    def test_rollback_restores_relay_key_on_api_key_axis(
        self, temp_home: Path, monkeypatch
    ):
        s = _switcher()
        _seed_pair(s)
        s.switch_to("2", json_output=True)

        def boom(data, value):
            raise OSError("disk full")

        monkeypatch.setattr(s, "_set_managed_base_url_marker", boom)
        with pytest.raises(SwitchError, match="rolled back"):
            s.switch_to("1", json_output=True)
        assert _settings_base_url() == URL
        assert s._get_sequence_data()["managedBaseUrl"] == URL
        cfg = json.loads(get_global_config_path().read_text(encoding="utf-8"))
        assert cfg["primaryApiKey"] == POOL_KEY
        assert not get_credentials_path().exists()
        assert s._get_sequence_data()["activeAccountNumber"] == 2

    def test_direct_branch_restores_settings(self, temp_home: Path, monkeypatch):
        s = _switcher()
        s.add_account_from_token(POOL_KEY, slot=2, base_url=URL)

        def boom(data, value):
            raise OSError("disk full")

        monkeypatch.setattr(s, "_set_managed_base_url_marker", boom)
        with pytest.raises(OSError):
            s.switch_to("2", json_output=True)
        assert _settings_base_url() == ANTHROPIC_API_URL
        assert s._get_sequence_data()["managedBaseUrl"] == ANTHROPIC_API_URL

    def test_direct_branch_restores_previous_value_exactly(
        self, temp_home: Path, monkeypatch
    ):
        s = _switcher()
        s.add_account_from_token(POOL_KEY, slot=2, base_url=URL)
        get_claude_settings_path().write_text(
            json.dumps({"env": {BASE_URL_ENV: OTHER_URL}}), encoding="utf-8"
        )
        data = s._get_sequence_data()
        data["managedBaseUrl"] = OTHER_URL
        s._write_json(s.sequence_file, data)

        def boom(data, value):
            raise OSError("disk full")

        monkeypatch.setattr(s, "_set_managed_base_url_marker", boom)
        with pytest.raises(OSError):
            s.switch_to("2", json_output=True)
        assert _settings_base_url() == OTHER_URL
        assert s._get_sequence_data()["managedBaseUrl"] == OTHER_URL


def _log_in_elsewhere(s: ClaudeAccountSwitcher) -> None:
    """The user logs Claude Code in to something else: the relay key is no
    longer the live credential (OAuth written, managed key cleared)."""
    s._write_credentials(json.dumps(
        {"claudeAiOauth": {"accessToken": "sk-ant-oat01-elsewhere"}}
    ))


class TestRemoveAndPurge:
    def test_remove_while_relay_key_is_live_keeps_url_and_marker(
        self, temp_home: Path, capsys
    ):
        s = _switcher()
        _seed_pair(s)
        s.switch_to("2", json_output=True)
        capsys.readouterr()
        s.remove_account("2", assume_yes=True)
        assert _settings_base_url() == URL
        assert s._get_sequence_data()["managedBaseUrl"] == URL
        out = capsys.readouterr().out
        assert "still Claude Code's live login" in out
        assert "cswap switch" in out
        assert POOL_KEY not in out

        # A later switch to an ordinary account still resets it.
        s.switch_to("1", json_output=True)
        assert _settings_base_url() == ANTHROPIC_API_URL
        assert s._get_sequence_data()["managedBaseUrl"] == ANTHROPIC_API_URL

    def test_remove_after_login_elsewhere_writes_default(
        self, temp_home: Path, capsys
    ):
        s = _switcher()
        _seed_pair(s)
        s.switch_to("2", json_output=True)
        _log_in_elsewhere(s)
        capsys.readouterr()
        s.remove_account("2", assume_yes=True)
        assert _settings() == {"env": {BASE_URL_ENV: ANTHROPIC_API_URL}}
        assert s._get_sequence_data()["managedBaseUrl"] == ANTHROPIC_API_URL
        out = capsys.readouterr().out
        assert "Set ANTHROPIC_BASE_URL" in out
        assert "restart" not in out.lower()

    def test_remove_with_unreadable_live_store_keeps_url(
        self, temp_home: Path, monkeypatch
    ):
        s = _switcher()
        _seed_pair(s)
        s.switch_to("2", json_output=True)
        _log_in_elsewhere(s)
        monkeypatch.setattr(s, "_read_credentials", lambda: None)
        s.remove_account("2", assume_yes=True)
        assert _settings_base_url() == URL
        assert s._get_sequence_data()["managedBaseUrl"] == URL

    def test_remove_leaves_user_value(self, temp_home: Path):
        s = _switcher()
        _seed_pair(s)
        s.switch_to("2", json_output=True)
        _log_in_elsewhere(s)
        get_claude_settings_path().write_text(
            json.dumps({"env": {BASE_URL_ENV: "https://mine.example.com"}}),
            encoding="utf-8",
        )
        s.remove_account("2", assume_yes=True)
        assert _settings_base_url() == "https://mine.example.com"
        assert "managedBaseUrl" not in s._get_sequence_data()

    def test_remove_inactive_slot_keeps_value(self, temp_home: Path):
        s = _switcher()
        _seed_pair(s)
        s.add_account_from_token(OTHER_POOL_KEY, slot=3, base_url=OTHER_URL)
        s.switch_to("2", json_output=True)
        s.remove_account("3", assume_yes=True)
        assert _settings_base_url() == URL
        assert s._get_sequence_data()["managedBaseUrl"] == URL

    def test_purge_while_relay_key_is_live_keeps_url(
        self, temp_home: Path, monkeypatch, capsys
    ):
        s = _switcher()
        _seed_pair(s)
        s.switch_to("2", json_output=True)
        get_claude_settings_path().write_text(
            json.dumps({"model": "opus", "env": {BASE_URL_ENV: URL}}), encoding="utf-8"
        )
        monkeypatch.setattr(builtins, "input", lambda prompt="": "y")
        capsys.readouterr()
        s.purge()
        assert _settings() == {"model": "opus", "env": {BASE_URL_ENV: URL}}
        assert not s.backup_dir.exists()
        out = capsys.readouterr().out
        assert "still Claude Code's live login" in out
        assert f"to {ANTHROPIC_API_URL} by hand" in out
        assert POOL_KEY not in out

    def test_purge_after_login_elsewhere_leaves_default(
        self, temp_home: Path, monkeypatch, capsys
    ):
        s = _switcher()
        _seed_pair(s)
        s.switch_to("2", json_output=True)
        _log_in_elsewhere(s)
        get_claude_settings_path().write_text(
            json.dumps({"model": "opus", "env": {BASE_URL_ENV: URL}}), encoding="utf-8"
        )
        monkeypatch.setattr(builtins, "input", lambda prompt="": "y")
        capsys.readouterr()
        s.purge()
        assert _settings() == {
            "model": "opus", "env": {BASE_URL_ENV: ANTHROPIC_API_URL},
        }
        assert not s.backup_dir.exists()
        out = capsys.readouterr().out
        assert "harmless" in out
        assert "by hand" not in out

    def test_purge_then_add_relay_and_switch(
        self, temp_home: Path, monkeypatch
    ):
        s = _switcher()
        _seed_pair(s)
        s.switch_to("2", json_output=True)
        _log_in_elsewhere(s)
        monkeypatch.setattr(builtins, "input", lambda prompt="": "y")
        s.purge()
        assert _settings_base_url() == ANTHROPIC_API_URL

        s = _switcher()
        _seed_pair(s)
        assert "managedBaseUrl" not in s._get_sequence_data()
        result = s.switch_to("2", json_output=True)
        assert result["switched"] is True
        assert _settings_base_url() == URL
        assert s._get_sequence_data()["managedBaseUrl"] == URL

    def test_purge_with_managed_default_leaves_it(
        self, temp_home: Path, monkeypatch, capsys
    ):
        s = _switcher()
        _seed_pair(s)
        s.switch_to("2", json_output=True)
        s.switch_to("1", json_output=True)
        before = get_claude_settings_path().read_text(encoding="utf-8")
        monkeypatch.setattr(builtins, "input", lambda prompt="": "y")
        capsys.readouterr()
        s.purge()
        assert get_claude_settings_path().read_text(encoding="utf-8") == before
        out = capsys.readouterr().out
        assert f"left at {ANTHROPIC_API_URL}" in out
        assert "harmless" in out


# ---------------------------------------------------------------------------
# Session mode
# ---------------------------------------------------------------------------


class _ExecCalled(Exception):
    def __init__(self, argv, env):
        self.argv, self.env = argv, env


@pytest.fixture
def capture_exec(monkeypatch):
    def fake_exec(self, claude_bin, claude_args, env):
        raise _ExecCalled([claude_bin, *claude_args], env)

    monkeypatch.setattr(session_mod.SessionManager, "_exec", fake_exec)
    monkeypatch.setattr(session_mod.shutil, "which", lambda name: f"/fake/bin/{name}")
    monkeypatch.setattr(
        session_mod.subprocess,
        "run",
        lambda *a, **k: pytest.fail("claude was probed for a key session"),
    )


def _settings_arg(argv: list[str]) -> dict:
    assert argv.count("--settings") == 1, argv
    return json.loads(argv[argv.index("--settings") + 1])


class TestKeySession:
    def test_env_and_settings_pin(self, temp_home: Path, capture_exec, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-from-shell")
        monkeypatch.setenv(BASE_URL_ENV, "https://shell.example.com")
        s = _switcher()
        s.add_account_from_token(POOL_KEY, slot=2, base_url=URL)
        with pytest.raises(_ExecCalled) as exc:
            SessionManager(s).run("2", ["--resume"])
        argv, env = exc.value.argv, exc.value.env

        assert env["ANTHROPIC_API_KEY"] == POOL_KEY
        assert env[BASE_URL_ENV] == URL
        assert all(POOL_KEY not in a for a in argv)
        assert _settings_arg(argv) == {"env": {BASE_URL_ENV: URL}}
        assert argv[-1] == "--resume"

        session_dir = session_dir_for(s.backup_dir, "2", "api-key-2@token.local")
        assert env["CLAUDE_CONFIG_DIR"] == str(session_dir)
        cfg = json.loads((session_dir / ".claude.json").read_text(encoding="utf-8"))
        assert approved_form(POOL_KEY) in cfg["customApiKeyResponses"]["approved"]
        assert cfg["oauthAccount"]["emailAddress"] == "api-key-2@token.local"
        assert cfg["hasCompletedOnboarding"] is True
        assert "primaryApiKey" not in cfg
        assert not (session_dir / ".credentials.json").exists()

    def test_reseed_is_idempotent(self, temp_home: Path, capture_exec):
        s = _switcher()
        s.add_account_from_token(POOL_KEY, slot=2, base_url=URL)
        mgr = SessionManager(s)
        with pytest.raises(_ExecCalled):
            mgr.run("2", [])
        session_dir = session_dir_for(s.backup_dir, "2", "api-key-2@token.local")
        cfg_path = session_dir / ".claude.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["projects"] = {"/work": {}}
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        before = cfg_path.read_text(encoding="utf-8")
        with pytest.raises(_ExecCalled):
            mgr.run("2", [])
        assert cfg_path.read_text(encoding="utf-8") == before

    def test_setup_session_accepts_key_account(self, temp_home: Path, capture_exec):
        s = _switcher()
        s.add_account_from_token(POOL_KEY, slot=2, base_url=URL)
        session_dir, num, email = SessionManager(s).setup_session("2", share=True)
        assert (num, email) == ("2", "api-key-2@token.local")
        assert (session_dir / ".claude.json").exists()

    def test_api_key_without_base_url_still_refused(self, temp_home: Path, capture_exec):
        s = _switcher()
        s.add_account_from_token(API_KEY, slot=3)
        with pytest.raises(SessionError, match="does not support API-key accounts"):
            SessionManager(s).run("3", [])

    def test_user_inline_settings_are_merged(self, temp_home: Path, capture_exec):
        s = _switcher()
        s.add_account_from_token(POOL_KEY, slot=2, base_url=URL)
        with pytest.raises(_ExecCalled) as exc:
            SessionManager(s).run(
                "2", ["--settings", '{"model": "opus", "env": {"FOO": "1"}}']
            )
        assert _settings_arg(exc.value.argv) == {
            "model": "opus", "env": {"FOO": "1", BASE_URL_ENV: URL},
        }

    def test_user_inline_equals_form_is_merged(self, temp_home: Path, capture_exec):
        s = _switcher()
        s.add_account_from_token(POOL_KEY, slot=2, base_url=URL)
        with pytest.raises(_ExecCalled) as exc:
            SessionManager(s).run("2", ['--settings={"model": "opus"}'])
        argv = exc.value.argv
        assert "--settings" not in argv
        merged = [a for a in argv if a.startswith("--settings=")]
        assert len(merged) == 1
        assert json.loads(merged[0][len("--settings="):]) == {
            "model": "opus", "env": {BASE_URL_ENV: URL},
        }

    def test_user_settings_file_passed_through(
        self, temp_home: Path, capture_exec, capsys
    ):
        s = _switcher()
        s.add_account_from_token(POOL_KEY, slot=2, base_url=URL)
        with pytest.raises(_ExecCalled) as exc:
            SessionManager(s).run("2", ["--settings", "/tmp/mine.json"])
        argv = exc.value.argv
        assert argv[1:] == ["--settings", "/tmp/mine.json"]
        assert exc.value.env[BASE_URL_ENV] == URL
        assert "left it as is" in capsys.readouterr().out

    def test_user_settings_file_refused_when_shared_points_elsewhere(
        self, temp_home: Path, capture_exec
    ):
        s = _switcher()
        s.add_account_from_token(POOL_KEY, slot=2, base_url=URL)
        get_claude_settings_path().write_text(
            json.dumps({"env": {BASE_URL_ENV: "https://mine.example.com"}}),
            encoding="utf-8",
        )
        with pytest.raises(SessionError, match="would be sent there"):
            SessionManager(s).run("2", ["--settings", "/tmp/mine.json"])

    def test_fast_path_pins_endpoint_without_touching_env(
        self, temp_home: Path, capture_exec, monkeypatch
    ):
        s = _switcher()
        s.add_account_from_token(POOL_KEY, slot=2, base_url=URL)
        monkeypatch.setattr(
            s, "_get_current_account", lambda: ("api-key-2@token.local", "")
        )
        with pytest.raises(_ExecCalled) as exc:
            SessionManager(s).run("2", [])
        assert _settings_arg(exc.value.argv) == {"env": {BASE_URL_ENV: URL}}
        assert "CLAUDE_CONFIG_DIR" not in exc.value.env
        assert exc.value.env.get("ANTHROPIC_API_KEY") != POOL_KEY


class TestNormalAccountSession:
    @pytest.fixture
    def seeded(self, temp_home: Path, monkeypatch):
        s = _switcher()
        s.add_account_from_token(SETUP_TOKEN, slot=1)
        s.add_account_from_token(POOL_KEY, slot=2, base_url=URL)
        session_dir = session_dir_for(s.backup_dir, "1", "setup-token-1@token.local")
        session_dir.mkdir(parents=True)
        monkeypatch.setattr(
            SessionManager,
            "setup_session",
            lambda self, ident, share, share_history=False: (
                session_dir, "1", "setup-token-1@token.local",
            ),
        )
        return s

    def _manage_global(self, s: ClaudeAccountSwitcher, value: str) -> None:
        get_claude_settings_path().write_text(
            json.dumps({"env": {BASE_URL_ENV: value}}), encoding="utf-8"
        )
        data = s._get_sequence_data()
        data["managedBaseUrl"] = value
        s._write_json(s.sequence_file, data)

    def test_managed_global_is_overridden(
        self, seeded, capture_exec, monkeypatch
    ):
        monkeypatch.setenv(BASE_URL_ENV, URL)
        self._manage_global(seeded, URL)
        with pytest.raises(_ExecCalled) as exc:
            SessionManager(seeded).run("1", [])
        assert _settings_arg(exc.value.argv) == {
            "env": {BASE_URL_ENV: ANTHROPIC_API_URL},
        }
        assert BASE_URL_ENV not in exc.value.env
        assert "ANTHROPIC_API_KEY" not in exc.value.env

    def test_user_global_is_left_alone(self, seeded, capture_exec):
        get_claude_settings_path().write_text(
            json.dumps({"env": {BASE_URL_ENV: "https://mine.example.com"}}),
            encoding="utf-8",
        )
        with pytest.raises(_ExecCalled) as exc:
            SessionManager(seeded).run("1", ["--resume"])
        assert exc.value.argv[1:] == ["--resume"]

    def test_no_global_no_pin(self, seeded, capture_exec):
        with pytest.raises(_ExecCalled) as exc:
            SessionManager(seeded).run("1", [])
        assert exc.value.argv[1:] == []

    def test_no_share_skips_pin(self, seeded, capture_exec):
        self._manage_global(seeded, URL)
        with pytest.raises(_ExecCalled) as exc:
            SessionManager(seeded).run("1", [], share=False)
        assert exc.value.argv[1:] == []

    def test_managed_global_with_user_settings_file_refused(
        self, seeded, capture_exec
    ):
        self._manage_global(seeded, URL)
        with pytest.raises(SessionError, match="would be sent there"):
            SessionManager(seeded).run("1", ["--settings", "/tmp/mine.json"])


# ---------------------------------------------------------------------------
# Usage: custom-endpoint sentinel, never fetched, no profile lookups
# ---------------------------------------------------------------------------


class TestUsageNeverFetched:
    def test_usage_fields_maps_custom_endpoint(self):
        assert usage_fields(USAGE_CUSTOM_ENDPOINT) == ("custom_endpoint", None)

    @pytest.mark.parametrize("token", [POOL_KEY, SETUP_TOKEN])
    def test_collect_short_circuits(self, temp_home: Path, monkeypatch, token):
        s = _switcher()
        s.add_account_from_token(token, slot=2, base_url=URL)
        email = _record(s, "2")["email"]
        creds = s._read_account_credentials("2", email)
        monkeypatch.setattr(
            s, "_fetch_account_usage",
            lambda *a, **k: pytest.fail("usage fetched for a custom endpoint"),
        )
        entries = s._collect_usage_entries([(2, email, "", "", False, creds, "")])
        assert entries["2"].sentinel == USAGE_CUSTOM_ENDPOINT

    def test_active_setup_token_relay_never_fetched(
        self, temp_home: Path, monkeypatch
    ):
        s = _switcher()
        s.add_account_from_token(SETUP_TOKEN, slot=2, base_url=URL)
        s.switch_to("2", json_output=True)
        monkeypatch.setattr(
            s, "_fetch_active_usage",
            lambda *a, **k: pytest.fail("usage fetched for a custom endpoint"),
        )
        entry = s._active_account_usage("2", "setup-token-2@token.local", "")
        assert entry.sentinel == USAGE_CUSTOM_ENDPOINT

    def test_prefetch_skips_profile_lookup(self, temp_home: Path, monkeypatch):
        s = _switcher()
        s.add_account_from_token(SETUP_TOKEN, slot=2, base_url=URL)
        s.switch_to("2", json_output=True)
        # Live bytes diverge from the backup: a normal slot would be resolved
        # against Anthropic's profile endpoint here.
        get_credentials_path().write_text(
            json.dumps({"claudeAiOauth": {"accessToken": "sk-ant-oat01-rotated"}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "claude_swap.oauth.fetch_oauth_profile",
            lambda token: pytest.fail("profile lookup for a custom endpoint"),
        )
        assert s._prefetch_live_identity()["resolved"] is None

    def test_capture_guard_skips_profile_lookup(self, temp_home: Path, monkeypatch):
        s = _switcher()
        s.add_account_from_token(SETUP_TOKEN, slot=2, base_url=URL)
        monkeypatch.setattr(
            "claude_swap.oauth.fetch_oauth_profile",
            lambda token: pytest.fail("profile lookup for a custom endpoint"),
        )
        creds = s._read_account_credentials("2", "setup-token-2@token.local")
        assert s._reject_foreign_credential_capture(
            creds, "setup-token-2@token.local", "", ""
        ) == creds

    def test_resync_probe_skipped(self, temp_home: Path, monkeypatch):
        s = _switcher()
        s.add_account_from_token(SETUP_TOKEN, slot=2, base_url=URL)
        monkeypatch.setattr(
            "claude_swap.oauth.fetch_oauth_profile",
            lambda token: pytest.fail("profile lookup for a custom endpoint"),
        )
        rotated = json.dumps({"claudeAiOauth": {
            "accessToken": "sk-ant-oat01-a", "refreshToken": "r2",
        }})
        s._resync_rotated_backup("2", "setup-token-2@token.local", "", rotated)
        assert s._read_account_credentials(
            "2", "setup-token-2@token.local"
        ) != rotated

    def test_quotaless_for_autoswitch(self, temp_home: Path):
        s = _switcher()
        s.add_account_from_token(SETUP_TOKEN, slot=1)
        s.add_account_from_token(SETUP_TOKEN + "x", slot=2, base_url=URL)
        s.add_account_from_token(POOL_KEY, slot=3, base_url=URL)
        assert s.account_is_quotaless("1") is False
        assert s.account_is_quotaless("2") is True
        assert s.account_is_quotaless("3") is True

    def test_live_relay_key_not_captured_by_add(self, temp_home: Path):
        s = _switcher()
        _seed_pair(s)
        s.switch_to("2", json_output=True)
        with pytest.raises(ValidationError, match="Active login is an API-key account"):
            s.add_account()


# ---------------------------------------------------------------------------
# Surfaces: list, JSON, status, snapshot, export/import
# ---------------------------------------------------------------------------


class TestSurfaces:
    def test_account_row_additive_base_url(self):
        row = account_row(1, "a@x.com", "", "", False, None, base_url=URL)
        assert row["baseUrl"] == URL
        assert "baseUrl" not in account_row(1, "a@x.com", "", "", False, None)

    def test_list_json(self, temp_home: Path):
        s = _switcher()
        _seed_pair(s)
        payload = s.list_accounts(json_output=True)
        rows = {r["number"]: r for r in payload["accounts"]}
        assert rows[2]["baseUrl"] == URL
        assert rows[2]["usageStatus"] == "custom_endpoint"
        assert "baseUrl" not in rows[1]

    def test_list_human(self, temp_home: Path, capsys):
        s = _switcher()
        _seed_pair(s)
        capsys.readouterr()
        s.list_accounts()
        out = capsys.readouterr().out
        assert "→ relay.example.com" in out
        assert "metered · relay.example.com" in out
        assert POOL_KEY not in out

    def test_status_json(self, temp_home: Path):
        s = _switcher()
        _seed_pair(s)
        s.switch_to("2", json_output=True)
        active = s.status(json_output=True)["active"]
        assert active["baseUrl"] == URL
        assert active["usageStatus"] == "custom_endpoint"

    def test_snapshot(self, temp_home: Path):
        s = _switcher()
        _seed_pair(s)
        snap = {a.number: a for a in s.accounts_snapshot().accounts}
        assert snap["2"].base_url == URL
        assert snap["2"].usage.sentinel == USAGE_CUSTOM_ENDPOINT
        assert snap["1"].base_url == ""

    def test_export_import_round_trip(self, tmp_path: Path):
        from tests.test_api_key_accounts import _patched_home

        src_home = tmp_path / "src"
        (src_home / ".claude").mkdir(parents=True)
        with _patched_home(src_home):
            src = _switcher()
            src.add_account_from_token(POOL_KEY, slot=1, base_url=URL)
            src.add_account_from_token(SETUP_TOKEN, slot=2, base_url=OTHER_URL)
            out = tmp_path / "b.cswap"
            export_accounts(src, str(out))
            payload = json.loads(out.read_text(encoding="utf-8"))
            by_num = {a["number"]: a for a in payload["accounts"]}
            assert by_num[1]["credentials"] == POOL_KEY
            assert by_num[1]["kind"] == "api_key"
            assert by_num[1]["baseUrl"] == URL
            assert by_num[2]["baseUrl"] == OTHER_URL

        dst_home = tmp_path / "dst"
        (dst_home / ".claude").mkdir(parents=True)
        with _patched_home(dst_home):
            dst = _switcher()
            import_accounts(dst, str(out))
            assert _record(dst, "1")["baseUrl"] == URL
            assert _record(dst, "1")["kind"] == "api_key"
            assert _record(dst, "2")["baseUrl"] == OTHER_URL
            assert dst._read_account_credentials("1", "api-key-1@token.local") == POOL_KEY

    def test_import_rejects_invalid_base_url(self, tmp_path: Path):
        from claude_swap.exceptions import TransferError
        from tests.test_api_key_accounts import _patched_home

        src_home = tmp_path / "src"
        (src_home / ".claude").mkdir(parents=True)
        with _patched_home(src_home):
            src = _switcher()
            src.add_account_from_token(POOL_KEY, slot=1, base_url=URL)
            out = tmp_path / "b.cswap"
            export_accounts(src, str(out))
        payload = json.loads(out.read_text(encoding="utf-8"))
        payload["accounts"][0]["baseUrl"] = "ftp://nope"
        out.write_text(json.dumps(payload), encoding="utf-8")

        dst_home = tmp_path / "dst"
        (dst_home / ".claude").mkdir(parents=True)
        with _patched_home(dst_home):
            dst = _switcher()
            with pytest.raises(TransferError, match="invalid baseUrl"):
                import_accounts(dst, str(out))

    def test_import_rejects_raw_relay_key_without_base_url(self, tmp_path: Path):
        from claude_swap.exceptions import TransferError
        from tests.test_api_key_accounts import _patched_home

        src_home = tmp_path / "src"
        (src_home / ".claude").mkdir(parents=True)
        with _patched_home(src_home):
            src = _switcher()
            src.add_account_from_token(POOL_KEY, slot=1, base_url=URL)
            out = tmp_path / "b.cswap"
            export_accounts(src, str(out))
        payload = json.loads(out.read_text(encoding="utf-8"))
        del payload["accounts"][0]["baseUrl"]
        out.write_text(json.dumps(payload), encoding="utf-8")

        dst_home = tmp_path / "dst"
        (dst_home / ".claude").mkdir(parents=True)
        with _patched_home(dst_home):
            dst = _switcher()
            with pytest.raises(TransferError, match="sk-ant-api"):
                import_accounts(dst, str(out))
