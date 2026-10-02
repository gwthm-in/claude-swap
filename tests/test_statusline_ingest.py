"""Tests for ``cswap ingest-statusline`` (claude_swap.statusline_ingest)."""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_swap import cli
from claude_swap.session import session_dir_for
from claude_swap.statusline_ingest import FEED_HOLD_S, ingest_statusline
from claude_swap.switcher import usage_stale_note
from claude_swap.usage_store import FetchRecord
from tests.test_transfer import _linux_switcher, _seed_account

DAY = 86400.0
ALICE = ("alice@example.com", "")
BOB = ("bob@example.com", "org-b")


def _iso(ts: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _payload(
    five: float | None = 80, seven: float | None = 74,
    five_reset: float | None = None, seven_reset: float | None = None,
) -> str:
    now = time.time()
    rate_limits: dict = {}
    if five is not None:
        rate_limits["five_hour"] = {
            "used_percentage": five,
            "resets_at": int(five_reset if five_reset is not None else now + 3 * 3600),
        }
    if seven is not None:
        rate_limits["seven_day"] = {
            "used_percentage": seven,
            "resets_at": int(seven_reset if seven_reset is not None else now + 3 * DAY),
        }
    return json.dumps({"session_id": "s", "model": {}, "rate_limits": rate_limits})


def _settle(s) -> None:
    """Age sequence.json's lastUpdated past the post-switch window."""
    data = s._get_sequence_data()
    data["lastUpdated"] = "2020-01-01T00:00:00Z"
    s._write_json(s.sequence_file, data)


def _login_as(home: Path, identity: tuple[str, str], base: Path | None = None) -> None:
    (base or home).mkdir(parents=True, exist_ok=True)
    ((base or home) / ".claude.json").write_text(json.dumps({
        "oauthAccount": {
            "emailAddress": identity[0],
            "organizationUuid": identity[1],
            "accountUuid": "acct",
        }
    }))


@pytest.fixture
def two_accounts(temp_home: Path):
    s = _linux_switcher(temp_home)
    _seed_account(s, 1, ALICE[0], ALICE[1])
    _seed_account(s, 2, BOB[0], BOB[1])
    _settle(s)
    return s


def _entry(s, num: str, identity: tuple[str, str]):
    return s._usage_store.entries({num: identity})[num]


class TestAttribution:
    def test_default_login_maps_the_current_identity_to_its_slot(
        self, two_accounts, temp_home: Path
    ):
        s = two_accounts
        _login_as(temp_home, BOB)

        assert ingest_statusline(s, _payload(five=80, seven=74)) == "2"

        entry = _entry(s, "2", BOB)
        assert entry.last_good["five_hour"]["pct"] == 80.0
        assert entry.last_good["seven_day"]["pct"] == 74.0
        assert entry.last_good["seven_day"]["resets_at"].endswith("+00:00")
        assert "countdown" in entry.last_good["seven_day"]
        assert entry.age_s == pytest.approx(0.0, abs=5)
        assert _entry(s, "1", ALICE).last_good is None

    def test_an_adopted_reading_holds_cswaps_own_polling(
        self, two_accounts, temp_home: Path
    ):
        s = two_accounts
        _login_as(temp_home, BOB)

        assert ingest_statusline(s, _payload(five=80, seven=74)) == "2"

        entry = _entry(s, "2", BOB)
        assert entry.held_until is not None
        assert entry.held(time.time())
        assert entry.held_until <= time.time() + FEED_HOLD_S + 5

    def test_feeding_a_429_blocked_slot_keeps_its_hold_lapse_off_the_endpoint(
        self, two_accounts, temp_home: Path
    ):
        s = two_accounts
        _login_as(temp_home, BOB)
        s._usage_store.record(
            {"2": FetchRecord(error="http-429", retry_after_s=60.0)},
            {"2": BOB},
        )

        assert ingest_statusline(s, _payload(five=80, seven=74)) == "2"

        entry = _entry(s, "2", BOB)
        assert entry.held_until is not None
        assert entry.backoff_until > entry.held_until + 3600

    def test_session_profile_maps_to_its_own_slot(
        self, two_accounts, temp_home: Path, monkeypatch
    ):
        s = two_accounts
        _login_as(temp_home, ALICE)
        profile = session_dir_for(s.backup_dir, "2", BOB[0])
        _login_as(temp_home, BOB, base=profile)
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(profile))
        # A switch a moment ago concerns the default login, not this profile.
        data = s._get_sequence_data()
        data["lastUpdated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        s._write_json(s.sequence_file, data)

        assert ingest_statusline(s, _payload(five=12)) == "2"
        assert _entry(s, "2", BOB).last_good["five_hour"]["pct"] == 12.0
        assert _entry(s, "1", ALICE).last_good is None

    def test_a_session_profile_logged_in_as_another_account_is_skipped(
        self, two_accounts, temp_home: Path, monkeypatch
    ):
        s = two_accounts
        _login_as(temp_home, BOB)
        profile = session_dir_for(s.backup_dir, "2", BOB[0])
        _login_as(temp_home, ALICE, base=profile)
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(profile))

        assert ingest_statusline(s, _payload()) is None
        assert _entry(s, "1", ALICE).last_good is None
        assert _entry(s, "2", BOB).last_good is None

    def test_an_unknown_login_is_skipped(self, two_accounts, temp_home: Path):
        _login_as(temp_home, ("stranger@example.com", ""))
        assert ingest_statusline(two_accounts, _payload()) is None

    def test_a_quotaless_slot_is_skipped(self, two_accounts, temp_home: Path):
        s = two_accounts
        data = s._get_sequence_data()
        data["accounts"]["2"]["baseUrl"] = "https://relay.example.com"
        s._write_json(s.sequence_file, data)
        _login_as(temp_home, BOB)

        assert ingest_statusline(s, _payload()) is None
        assert _entry(s, "2", BOB).last_good is None


class TestMisattributionGuard:
    def test_a_reading_right_after_a_switch_is_rejected(
        self, two_accounts, temp_home: Path
    ):
        s = two_accounts
        _login_as(temp_home, BOB)
        data = s._get_sequence_data()
        data["lastUpdated"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 30)
        )
        s._write_json(s.sequence_file, data)

        assert ingest_statusline(s, _payload()) is None
        assert _entry(s, "2", BOB).last_good is None

    def test_an_auto_switch_stamp_counts_too(self, two_accounts, temp_home: Path):
        s = two_accounts
        _login_as(temp_home, BOB)
        (s.backup_dir / "autoswitch_state.json").write_text(
            json.dumps({"lastSwitchAt": time.time() - 30})
        )
        assert ingest_statusline(s, _payload()) is None

        (s.backup_dir / "autoswitch_state.json").write_text(
            json.dumps({"lastSwitchAt": time.time() - 120})
        )
        assert ingest_statusline(s, _payload()) == "2"

    def _store_reading(self, s, seven_reset: float) -> None:
        usage = {"seven_day": {"pct": 10.0, "resets_at": _iso(seven_reset)}}
        s._usage_store.record({"2": FetchRecord(usage=usage)}, {"2": BOB})

    def test_a_different_weekly_reset_is_rejected(self, two_accounts, temp_home: Path):
        s = two_accounts
        _login_as(temp_home, BOB)
        now = time.time()
        self._store_reading(s, now + 3 * DAY)

        assert ingest_statusline(s, _payload(seven_reset=now + 5 * DAY)) is None
        assert _entry(s, "2", BOB).last_good["seven_day"]["pct"] == 10.0

    def test_a_matching_weekly_reset_is_accepted(self, two_accounts, temp_home: Path):
        s = two_accounts
        _login_as(temp_home, BOB)
        now = time.time()
        self._store_reading(s, now + 3 * DAY)

        assert ingest_statusline(s, _payload(seven_reset=now + 3 * DAY + 600)) == "2"

    def test_a_passed_reset_may_roll_forward_a_week(self, two_accounts, temp_home: Path):
        s = two_accounts
        _login_as(temp_home, BOB)
        now = time.time()
        self._store_reading(s, now - DAY)

        assert ingest_statusline(s, _payload(seven_reset=now + 6 * DAY)) == "2"
        assert _entry(s, "2", BOB).last_good["seven_day"]["pct"] == 74.0

    def test_a_future_reset_may_not_roll_forward(self, two_accounts, temp_home: Path):
        s = two_accounts
        _login_as(temp_home, BOB)
        now = time.time()
        self._store_reading(s, now + DAY)

        assert ingest_statusline(s, _payload(seven_reset=now + 8 * DAY)) is None


class TestStoredReading:
    def test_scoped_and_spend_windows_are_carried_over(
        self, two_accounts, temp_home: Path
    ):
        s = two_accounts
        _login_as(temp_home, BOB)
        now = time.time()
        seven_reset = now + 3 * DAY
        previous = {
            "five_hour": {"pct": 5.0, "resets_at": _iso(now + 3600)},
            "seven_day": {"pct": 10.0, "resets_at": _iso(seven_reset)},
            "spend": {
                "used": 1.0, "limit": 10.0, "pct": 10.0, "currency": "USD",
                "resets_at": _iso(now + 20 * DAY),
            },
            "scoped": [
                {"name": "Fable", "pct": 33.0, "resets_at": _iso(now + 2 * DAY)},
                {"name": "Old", "pct": 99.0, "resets_at": _iso(now - 60)},
            ],
        }
        s._usage_store.record({"2": FetchRecord(usage=previous)}, {"2": BOB})

        assert ingest_statusline(
            s, _payload(five=None, seven=74, seven_reset=seven_reset)
        ) == "2"

        usage = _entry(s, "2", BOB).last_good
        assert usage["seven_day"]["pct"] == 74.0
        assert usage["five_hour"] == previous["five_hour"]
        assert usage["spend"] == previous["spend"]
        assert usage["scoped"] == [previous["scoped"][0]]

    def test_a_fresh_reading_clears_the_stale_state(
        self, two_accounts, temp_home: Path
    ):
        """A rate-limited active account with an aged-out reading reads
        unknown and stale; the statusline reading makes it decision-grade."""
        s = two_accounts
        _login_as(temp_home, BOB)
        store = s._usage_store
        real_clock = store.clock
        store.clock = lambda: real_clock() - 3 * 3600
        store.record(
            {"2": FetchRecord(usage={"five_hour": {"pct": 50.0}})}, {"2": BOB}
        )
        store.clock = real_clock
        store.record(
            {"2": FetchRecord(error="http-429", retry_after_s=3600.0)}, {"2": BOB}
        )
        before = _entry(s, "2", BOB)
        assert before.decision_value() is None
        assert usage_stale_note(before, time.time()) is not None

        assert ingest_statusline(s, _payload(five=65)) == "2"

        after = _entry(s, "2", BOB)
        assert after.decision_value()["five_hour"]["pct"] == 65.0
        assert usage_stale_note(after, time.time()) is None

    def test_a_newer_stored_reading_is_kept(self, two_accounts, temp_home: Path):
        s = two_accounts
        _login_as(temp_home, BOB)
        store = s._usage_store
        real_clock = store.clock
        store.clock = lambda: real_clock() + 60
        store.record(
            {"2": FetchRecord(usage={"five_hour": {"pct": 50.0}})}, {"2": BOB}
        )
        store.clock = real_clock

        assert ingest_statusline(s, _payload(five=80)) is None
        assert _entry(s, "2", BOB).last_good["five_hour"]["pct"] == 50.0


class TestCli:
    def _run(self, text: str) -> None:
        with patch.object(sys, "argv", ["cswap", "ingest-statusline"]), \
             patch.object(sys, "stdin", io.StringIO(text)), \
             patch("claude_swap.update_check.check_for_update") as update_check:
            cli.main()
        update_check.assert_not_called()

    @pytest.mark.parametrize("text", [
        "",
        "not json",
        "[]",
        "{}",
        '{"rate_limits": null}',
        '{"rate_limits": {}}',
        '{"rate_limits": {"five_hour": {"used_percentage": "x"}}}',
        '{"rate_limits": {"seven_day": {"used_percentage": -1}}}',
    ])
    def test_unusable_input_exits_quietly(self, two_accounts, temp_home, text, capsys):
        _login_as(temp_home, BOB)
        self._run(text)
        out = capsys.readouterr()
        assert out.out == ""
        assert out.err == ""
        assert _entry(two_accounts, "2", BOB).last_good is None

    def test_no_accounts_exits_quietly(self, temp_home, capsys):
        self._run(_payload())
        assert capsys.readouterr().out == ""

    def test_a_failure_inside_is_swallowed(self, two_accounts, temp_home, capsys):
        _login_as(temp_home, BOB)
        with patch(
            "claude_swap.statusline_ingest.ingest_statusline",
            side_effect=RuntimeError("boom"),
        ):
            self._run(_payload())
        assert capsys.readouterr().out == ""

    def test_a_usable_payload_is_adopted_silently(
        self, two_accounts, temp_home, capsys
    ):
        _login_as(temp_home, BOB)
        self._run(_payload(five=80))
        assert capsys.readouterr().out == ""
        assert _entry(two_accounts, "2", BOB).last_good["five_hour"]["pct"] == 80.0
