"""Tests for pool usage: a base-URL API-key account asking its pool's own
``/api/oauth/usage`` (never Anthropic's), the "metered" fallback and its
cached unsupported verdict, the https/loopback rule, auto-switch candidacy
both ways, and the key never reaching a log line.

No network: ``claude_swap.oauth._open_pool_usage`` is replaced by a fake
opener in every test (conftest's default answers 404).
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_swap import oauth
from claude_swap.autoswitch import NoSwitchEvent, TickOutcome
from claude_swap.json_output import USAGE_API_KEY, USAGE_CUSTOM_ENDPOINT
from claude_swap.models import Platform
from claude_swap.statusline_ingest import ingest_statusline
from claude_swap.switcher import ClaudeAccountSwitcher
from tests.test_autoswitch import EngineHarness, _usage

URL = "https://pool.example.com:8448"
POOL_KEY = "pool_" + "s3cr3t" * 8
API_KEY = "sk-ant-api03-" + "a1b2c3d4e5" * 4


def _iso_in(seconds: float) -> str:
    return datetime.fromtimestamp(time.time() + seconds, tz=timezone.utc).isoformat()


def _pool_body(five: float | None = 30, seven: float | None = 12) -> dict:
    return {
        "five_hour": {"utilization": five, "resets_at": _iso_in(3 * 3600)},
        "seven_day": {"utilization": seven, "resets_at": _iso_in(3 * 86400)},
        "pool": {"eligible": 3, "total": 5, "observedAt": _iso_in(0)},
    }


class FakeOpener:
    """Stands in for ``oauth._open_pool_usage``: records every request."""

    def __init__(self, body: object = None, *, status: int | None = None,
                 exc: Exception | None = None):
        self.body = body
        self.status = status
        self.exc = exc
        self.requests: list[urllib.request.Request] = []

    def __call__(self, req: urllib.request.Request) -> bytes:
        self.requests.append(req)
        if self.exc is not None:
            raise self.exc
        if self.status is not None:
            raise urllib.error.HTTPError(req.full_url, self.status, "x", None, None)
        return json.dumps(self.body).encode()


@pytest.fixture
def opener(monkeypatch) -> FakeOpener:
    fake = FakeOpener(_pool_body())
    monkeypatch.setattr("claude_swap.oauth._open_pool_usage", fake)
    return fake


def _switcher() -> ClaudeAccountSwitcher:
    s = ClaudeAccountSwitcher()
    s.platform = Platform.LINUX
    s._setup_directories()
    s._init_sequence_file()
    return s


def _pool_switcher(base_url: str = URL) -> ClaudeAccountSwitcher:
    s = _switcher()
    s.add_account_from_token(POOL_KEY, slot=2, base_url=base_url)
    return s


def _collect(s: ClaudeAccountSwitcher):
    return s._collect_usage_entries(s._build_accounts_info())


# ---------------------------------------------------------------------------
# oauth: URL rule, request, parse, classification
# ---------------------------------------------------------------------------


class TestPoolUsageUrl:
    @pytest.mark.parametrize(
        ("base", "expected"),
        [
            (URL, URL + "/api/oauth/usage"),
            ("https://relay.example.com/api", "https://relay.example.com/api/api/oauth/usage"),
            ("http://localhost:8080", "http://localhost:8080/api/oauth/usage"),
            ("http://127.0.0.1:9000", "http://127.0.0.1:9000/api/oauth/usage"),
            ("http://[::1]:9000", "http://[::1]:9000/api/oauth/usage"),
        ],
    )
    def test_allowed(self, base, expected):
        assert oauth.pool_usage_url(base) == expected

    @pytest.mark.parametrize(
        "base",
        [
            "http://pool.example.com",
            "http://10.0.0.5:8080",
            "http://justin-paseo.tail0f8119.ts.net:8448",
            "ftp://pool.example.com",
            "",
        ],
    )
    def test_refused(self, base):
        assert oauth.pool_usage_url(base) is None

    def test_request_refuses_plain_http_without_opening(self, opener):
        with pytest.raises(ValueError):
            oauth.request_pool_usage_data("http://pool.example.com", POOL_KEY)
        assert opener.requests == []


class TestRequest:
    def test_key_goes_only_to_the_base_url(self, opener):
        oauth.request_pool_usage_data(URL, POOL_KEY)
        [req] = opener.requests
        assert req.full_url == URL + "/api/oauth/usage"
        assert req.get_header("X-api-key") == POOL_KEY
        assert req.get_header("Authorization") is None

    def test_redirects_are_refused(self):
        handler = oauth._RefuseRedirect()
        req = urllib.request.Request(URL + "/api/oauth/usage")
        assert handler.redirect_request(
            req, None, 302, "Found", {}, "https://elsewhere.example.com/"
        ) is None


class TestFetchPoolUsage:
    def test_supported_response_parsed(self, opener):
        outcome = oauth.fetch_pool_usage(URL, POOL_KEY)
        assert outcome.error is None
        assert outcome.usage["five_hour"]["pct"] == 30
        assert outcome.usage["seven_day"]["pct"] == 12
        assert "resets_at" in outcome.usage["five_hour"]
        assert "pool" not in outcome.usage

    def test_null_utilization_window_dropped(self, opener):
        opener.body = _pool_body(five=None, seven=40)
        usage = oauth.fetch_pool_usage(URL, POOL_KEY).usage
        assert "five_hour" not in usage
        assert usage["seven_day"]["pct"] == 40

    def test_all_null_is_success_without_data(self, opener):
        opener.body = _pool_body(five=None, seven=None)
        outcome = oauth.fetch_pool_usage(URL, POOL_KEY)
        assert outcome.error is None and outcome.usage is None

    def test_response_without_pool_object_is_unsupported(self, opener):
        body = _pool_body()
        del body["pool"]
        opener.body = body
        outcome = oauth.fetch_pool_usage(URL, POOL_KEY)
        assert outcome.error == oauth.POOL_USAGE_UNSUPPORTED

    @pytest.mark.parametrize("status", [401, 404, 405, 302])
    def test_unsupported_statuses(self, opener, status):
        opener.status = status
        outcome = oauth.fetch_pool_usage(URL, POOL_KEY)
        assert outcome.error == oauth.POOL_USAGE_UNSUPPORTED
        assert outcome.retry_after_s == oauth.POOL_UNSUPPORTED_RECHECK_S

    def test_server_error_keeps_its_kind(self, opener):
        opener.status = 503
        assert oauth.fetch_pool_usage(URL, POOL_KEY).error == "http-503"

    def test_network_error(self, opener):
        opener.exc = urllib.error.URLError("unreachable")
        assert oauth.fetch_pool_usage(URL, POOL_KEY).error == "network"


# ---------------------------------------------------------------------------
# Collector: supported, unsupported + cache, https enforcement
# ---------------------------------------------------------------------------


class TestCollect:
    def test_supported_endpoint_reports(self, temp_home: Path, opener):
        s = _pool_switcher()
        entry = _collect(s)["2"]
        assert entry.sentinel is None
        assert entry.last_good["five_hour"]["pct"] == 30
        assert s.account_is_quotaless("2") is False
        assert len(opener.requests) == 1

    def test_unsupported_reads_metered_and_is_cached(
        self, temp_home: Path, opener
    ):
        opener.status = 404
        s = _pool_switcher()
        entry = _collect(s)["2"]
        assert entry.sentinel == USAGE_CUSTOM_ENDPOINT
        assert entry.last_error == oauth.POOL_USAGE_UNSUPPORTED
        assert entry.backoff_until - s._usage_store.clock() == pytest.approx(
            oauth.POOL_UNSUPPORTED_RECHECK_S, abs=5
        )
        assert s.account_is_quotaless("2") is True
        # Another pass (and another surface) inside the hour does not ask.
        again = _collect(_switcher())["2"]
        assert again.sentinel == USAGE_CUSTOM_ENDPOINT
        assert len(opener.requests) == 1

    def test_unsupported_rechecked_after_the_hour(self, temp_home: Path, opener):
        opener.status = 404
        s = _pool_switcher()
        start = time.time()
        _collect(s)
        opener.status = None
        s._usage_store.clock = lambda: start + oauth.POOL_UNSUPPORTED_RECHECK_S + 60
        entry = _collect(s)["2"]
        assert len(opener.requests) == 2
        assert entry.sentinel is None
        assert s.account_is_quotaless("2") is False

    def test_network_error_with_no_reading_reads_metered(
        self, temp_home: Path, opener
    ):
        opener.exc = urllib.error.URLError("unreachable")
        s = _pool_switcher()
        entry = _collect(s)["2"]
        assert entry.sentinel == USAGE_CUSTOM_ENDPOINT
        assert entry.last_error == "network"
        assert s.account_is_quotaless("2") is True

    def test_plain_http_pool_never_asked(self, temp_home: Path, opener):
        s = _pool_switcher("http://pool.example.com:8448")
        entry = _collect(s)["2"]
        assert entry.sentinel == USAGE_CUSTOM_ENDPOINT
        assert opener.requests == []
        assert s.account_is_quotaless("2") is True

    def test_loopback_http_pool_asked(self, temp_home: Path, opener):
        s = _pool_switcher("http://127.0.0.1:8448")
        assert _collect(s)["2"].sentinel is None
        assert opener.requests[0].full_url == "http://127.0.0.1:8448/api/oauth/usage"

    def test_active_pool_uses_the_stored_key_at_its_own_url(
        self, temp_home: Path, opener
    ):
        s = _pool_switcher()
        s.switch_to("2", json_output=True)
        entry = _collect(s)["2"]
        assert entry.sentinel is None
        assert all(r.full_url.startswith(URL + "/") for r in opener.requests)
        assert all(r.get_header("X-api-key") == POOL_KEY for r in opener.requests)

    def test_pure_api_key_account_stays_quotaless(self, temp_home: Path, opener):
        s = _switcher()
        s.add_account_from_token(API_KEY, slot=3)
        assert _collect(s)["3"].sentinel == USAGE_API_KEY
        assert s.account_is_quotaless("3") is True
        assert opener.requests == []

    def test_key_never_logged(self, temp_home: Path, opener, caplog):
        caplog.set_level(logging.DEBUG, logger="claude-swap")
        s = _pool_switcher()
        for setup in (
            lambda: setattr(opener, "status", None),
            lambda: setattr(opener, "status", 404),
            lambda: setattr(opener, "status", 500),
            lambda: setattr(opener, "exc", urllib.error.URLError("down")),
        ):
            setup()
            oauth.fetch_pool_usage(URL, POOL_KEY, context="for account 2")
        _collect(s)
        assert caplog.records
        assert POOL_KEY not in caplog.text
        log_dir = Path(s.backup_dir)
        for path in log_dir.rglob("*.log"):
            assert POOL_KEY not in path.read_text(encoding="utf-8", errors="ignore")


class TestDisplay:
    def test_list_shows_windows_for_a_reporting_pool(
        self, temp_home: Path, opener, capsys
    ):
        s = _pool_switcher()
        capsys.readouterr()
        s.list_accounts()
        out = capsys.readouterr().out
        assert "5h:" in out and "7d:" in out
        assert "metered" not in out
        assert POOL_KEY not in out

    def test_list_keeps_metered_when_unsupported(
        self, temp_home: Path, opener, capsys
    ):
        opener.status = 404
        s = _pool_switcher()
        capsys.readouterr()
        s.list_accounts()
        out = capsys.readouterr().out
        assert "metered · pool.example.com:8448" in out

    def test_json_row(self, temp_home: Path, opener):
        s = _pool_switcher()
        rows = {r["number"]: r for r in s.list_accounts(json_output=True)["accounts"]}
        assert rows[2]["usageStatus"] != "custom_endpoint"
        assert rows[2]["baseUrl"] == URL


# ---------------------------------------------------------------------------
# Auto-switch: a reporting pool is a normal candidate; otherwise last resort
# ---------------------------------------------------------------------------


class TestAutoSwitch:
    @pytest.fixture(autouse=True)
    def _no_profile_probe(self):
        with patch("claude_swap.oauth.fetch_oauth_profile", return_value=None):
            yield

    def _harness(self, temp_home, monkeypatch, **settings):
        monkeypatch.setattr("claude_swap.switcher._FETCH_STAGGER_S", 0)
        h = EngineHarness(temp_home, **settings)
        h.seed(1, "a@example.com")
        h.switcher.add_account_from_token(POOL_KEY, slot=2, base_url=URL)
        h.seed(3, "c@example.com")
        h.make_live("a@example.com", 1)
        monkeypatch.setattr(h.switcher, "_live_session_pids", lambda *a: [])
        # Real clock for the store: the pool's resets_at are wall-clock.
        h.switcher._usage_store.clock = time.time
        h.clock.now = time.time()
        return h

    @staticmethod
    def _tick(h, oauth_usage: dict, errors: dict | None = None):
        def fake(num, email, creds, is_active=False, **kwargs):
            if errors and num in errors:
                return oauth.UsageOutcome(None, error=errors[num])
            value = oauth_usage.get(num)
            return oauth.UsageOutcome(dict(value) if value else None)

        with patch(
            "claude_swap.oauth.try_fetch_usage_for_account", side_effect=fake
        ):
            return h.engine.tick()

    def test_switches_onto_a_reporting_pool_with_headroom(
        self, temp_home, monkeypatch, opener
    ):
        opener.body = _pool_body(five=10, seven=5)
        h = self._harness(temp_home, monkeypatch)
        outcome = self._tick(h, {"1": _usage(95), "3": _usage(100)})
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 2

    def test_best_oauth_candidate_beats_a_busier_pool(
        self, temp_home, monkeypatch, opener
    ):
        # The best OAuth candidate still wins over a busier pool.
        opener.body = _pool_body(five=60, seven=5)
        h = self._harness(temp_home, monkeypatch)
        outcome = self._tick(h, {"1": _usage(95), "3": _usage(10)})
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 3

    def test_never_fails_over_onto_a_pool_at_threshold(
        self, temp_home, monkeypatch, opener
    ):
        opener.body = _pool_body(five=92, seven=5)
        h = self._harness(
            temp_home, monkeypatch, include_api_key_accounts=True, unhealthy_ticks=1
        )
        outcome = self._tick(h, {"3": _usage(100)}, errors={"1": "network"})
        assert outcome is not TickOutcome.SWITCHED
        assert h.active_number() == 1

    def test_fails_over_onto_a_pool_below_threshold(
        self, temp_home, monkeypatch, opener
    ):
        opener.body = _pool_body(five=40, seven=5)
        h = self._harness(temp_home, monkeypatch, unhealthy_ticks=1)
        outcome = self._tick(h, {"3": _usage(100)}, errors={"1": "network"})
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 2

    def test_switches_off_a_pool_at_threshold(self, temp_home, monkeypatch, opener):
        opener.body = _pool_body(five=95, seven=5)
        h = self._harness(temp_home, monkeypatch)
        h.switcher.switch_to("2", json_output=True)
        outcome = self._tick(h, {"1": _usage(5), "3": _usage(100)})
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 1

    def test_stays_on_a_pool_below_threshold(self, temp_home, monkeypatch, opener):
        opener.body = _pool_body(five=40, seven=5)
        h = self._harness(temp_home, monkeypatch)
        h.switcher.switch_to("2", json_output=True)
        outcome = self._tick(h, {"1": _usage(5), "3": _usage(5)})
        assert outcome is TickOutcome.NO_ACTION
        assert h.active_number() == 2
        assert [e.reason for e in h.events if isinstance(e, NoSwitchEvent)] == [
            "below-threshold"
        ]

    def test_unsupported_pool_excluded_by_default(
        self, temp_home, monkeypatch, opener
    ):
        opener.status = 404
        h = self._harness(temp_home, monkeypatch)
        outcome = self._tick(h, {"1": _usage(95), "3": _usage(100)})
        assert outcome is TickOutcome.BLOCKED
        assert h.active_number() == 1

    def test_unsupported_pool_is_last_resort_when_included(
        self, temp_home, monkeypatch, opener
    ):
        opener.status = 404
        h = self._harness(temp_home, monkeypatch, include_api_key_accounts=True)
        # A qualifying OAuth account still wins...
        outcome = self._tick(h, {"1": _usage(95), "3": _usage(10)})
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 3

    def test_unsupported_pool_used_when_oauth_exhausted(
        self, temp_home, monkeypatch, opener
    ):
        opener.status = 404
        h = self._harness(temp_home, monkeypatch, include_api_key_accounts=True)
        outcome = self._tick(h, {"1": _usage(100), "3": _usage(100)})
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 2

    def test_active_unsupported_pool_idles_engine(
        self, temp_home, monkeypatch, opener
    ):
        opener.status = 404
        h = self._harness(temp_home, monkeypatch)
        h.switcher.switch_to("2", json_output=True)
        outcome = self._tick(h, {"1": _usage(5), "3": _usage(5)})
        assert outcome is TickOutcome.NO_ACTION
        assert h.active_number() == 2
        assert [e.reason for e in h.events if isinstance(e, NoSwitchEvent)] == [
            "active-api-key"
        ]


# ---------------------------------------------------------------------------
# Statusline ingest keeps skipping base-URL slots
# ---------------------------------------------------------------------------


def test_statusline_ingest_skips_a_reporting_pool(temp_home: Path, opener):
    s = _pool_switcher()
    s.switch_to("2", json_output=True)
    _collect(s)
    assert s.account_is_quotaless("2") is False
    data = s._get_sequence_data()
    data["lastUpdated"] = "2020-01-01T00:00:00Z"
    s._write_json(s.sequence_file, data)
    payload = json.dumps({
        "session_id": "s",
        "model": {},
        "rate_limits": {
            "five_hour": {"used_percentage": 99, "resets_at": int(time.time() + 3600)},
            "seven_day": {"used_percentage": 99, "resets_at": int(time.time() + 86400)},
        },
    })
    before = s._usage_store.entries(
        {"2": (data["accounts"]["2"]["email"], "")}
    )["2"].last_good
    assert ingest_statusline(s, payload) is None
    after = s._usage_store.entries(
        {"2": (data["accounts"]["2"]["email"], "")}
    )["2"].last_good
    assert after == before
