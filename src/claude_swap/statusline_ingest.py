"""Adopt the rate-limit figures Claude Code hands its statusline command.

Claude Code passes the statusline command a JSON object on stdin whose
``rate_limits`` carries the live 5-hour and 7-day utilization of the account
that served the last reply (integer percentages, ``resets_at`` in epoch
seconds). It costs no request, so it keeps the active account's reading fresh
while cswap's own ``/api/oauth/usage`` polling is rate-limited: Claude Code's
sessions spend that endpoint's budget for the active account too.

``cswap ingest-statusline`` reads that object, works out which slot it
belongs to and hands the reading to :meth:`UsageStore.adopt` at age 0. It runs
from a statusline, so it never writes to stdout and never fails noisily:
anything unusable is logged at debug level and dropped.
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from claude_swap import oauth
from claude_swap.poll_policy import parse_reset_ts

if TYPE_CHECKING:
    from claude_swap.switcher import ClaudeAccountSwitcher

# Right after a switch a statusline can fire for a reply the previous account
# served while ~/.claude.json already names the new one.
SWITCH_SETTLE_S = 90.0
# Weekly reset times are fixed per account; a larger gap means the figures are
# another account's.
WEEKLY_RESET_TOLERANCE_S = 3600.0
WEEK_S = 7 * 86400.0


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _window(raw: object) -> dict | None:
    """One statusline window in the shape ``oauth.build_usage_result`` stores."""
    if not isinstance(raw, dict):
        return None
    pct = raw.get("used_percentage")
    if not _is_number(pct) or pct < 0:
        return None
    out: dict = {"pct": float(pct)}
    resets_at = raw.get("resets_at")
    if _is_number(resets_at) and resets_at > 0:
        iso = datetime.fromtimestamp(resets_at, tz=timezone.utc).isoformat()
        out["resets_at"] = iso
        out["countdown"], out["clock"] = oauth.format_reset(iso)
    return out


def usage_from_rate_limits(rate_limits: object) -> dict | None:
    """The usage dict for a statusline ``rate_limits`` object, or None."""
    if not isinstance(rate_limits, dict):
        return None
    usage: dict = {}
    for key in ("five_hour", "seven_day"):
        window = _window(rate_limits.get(key))
        if window is not None:
            usage[key] = window
    return usage or None


def _still_current(window: object, now: float) -> bool:
    if not isinstance(window, dict):
        return False
    reset_ts = parse_reset_ts(window.get("resets_at"))
    return reset_ts is None or reset_ts > now


def _merge_carried(usage: dict, last_good: dict | None, now: float) -> dict:
    """``usage`` plus what the statusline does not report, from ``last_good``.

    The statusline carries only the 5h/7d windows. Per-model weekly limits
    ("Fable"), spend, and any 5h/7d window it omitted are carried over from
    the slot's last-good reading so the menu does not lose them. Those
    carried-over figures keep their older age: they are as old as the reading
    they came from, even though the row's single ``fetchedAt`` now says fresh;
    cswap's own next successful fetch replaces them. Windows whose reset has
    already passed are dropped rather than carried, since they no longer
    describe the current window.
    """
    if not isinstance(last_good, dict):
        return usage
    merged = dict(usage)
    for key in ("five_hour", "seven_day", "spend"):
        if key not in merged and _still_current(last_good.get(key), now):
            merged[key] = last_good[key]
    scoped = last_good.get("scoped")
    if isinstance(scoped, list):
        kept = [w for w in scoped if _still_current(w, now)]
        if kept:
            merged["scoped"] = kept
    return merged


def _weekly_reset_mismatch(
    usage: dict, last_good: dict | None, now: float
) -> bool:
    """Whether the incoming 7-day reset contradicts the slot's stored one.

    A stored reset that has passed may roll forward by whole weeks.
    """
    incoming = parse_reset_ts((usage.get("seven_day") or {}).get("resets_at"))
    stored_window = last_good.get("seven_day") if isinstance(last_good, dict) else None
    stored = parse_reset_ts(
        stored_window.get("resets_at") if isinstance(stored_window, dict) else None
    )
    if incoming is None or stored is None:
        return False
    diff = incoming - stored
    if abs(diff) <= WEEKLY_RESET_TOLERANCE_S:
        return False
    if stored <= now and diff > 0:
        weeks = round(diff / WEEK_S)
        if weeks >= 1 and abs(diff - weeks * WEEK_S) <= WEEKLY_RESET_TOLERANCE_S:
            return False
    return True


def _last_switch_at(switcher: ClaudeAccountSwitcher, data: dict) -> float | None:
    """Epoch of the most recent account switch, manual or automatic.

    ``sequence.json``'s ``lastUpdated`` is rewritten by every switch path
    (and by rarer roster edits); the auto engine also stamps
    ``autoswitch_state.json``'s ``lastSwitchAt``. The later of the two wins.
    """
    from claude_swap.autoswitch import STATE_FILENAME

    stamps: list[float] = []
    updated = data.get("lastUpdated")
    if isinstance(updated, str):
        ts = parse_reset_ts(updated)
        if ts is not None:
            stamps.append(ts)
    try:
        state = json.loads(
            (switcher.backup_dir / STATE_FILENAME).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        state = None
    if isinstance(state, dict) and _is_number(state.get("lastSwitchAt")):
        stamps.append(float(state["lastSwitchAt"]))
    return max(stamps) if stamps else None


def _session_slot(
    switcher: ClaudeAccountSwitcher, data: dict, config_dir: str
) -> tuple[bool, str | None]:
    """``(inside_sessions, slot)`` for a ``CLAUDE_CONFIG_DIR``.

    ``inside_sessions`` is False when the directory is not under cswap's
    session profiles; ``slot`` is None when it is, but cannot be attributed
    (unknown profile, or an in-session ``/login`` re-pointed it at another
    account).
    """
    from claude_swap.session import read_session_identity, session_dir_for

    sessions_root = (switcher.backup_dir / "sessions").resolve()
    try:
        rel = Path(config_dir).resolve().relative_to(sessions_root)
    except ValueError:
        return False, None
    if not rel.parts:
        return True, None
    name = rel.parts[0]
    num = name.split("-", 1)[0]
    record = data.get("accounts", {}).get(num)
    if not isinstance(record, dict):
        return True, None
    email = record.get("email") or ""
    if not email or session_dir_for(switcher.backup_dir, num, email).name != name:
        return True, None
    identity = read_session_identity(sessions_root / name)
    if identity is not None and identity != (
        email, record.get("organizationUuid", "") or ""
    ):
        return True, None
    return True, num


def ingest_statusline(switcher: ClaudeAccountSwitcher, text: str) -> str | None:
    """Adopt one statusline payload's rate limits; the adopted slot, or None.

    Attribution: a ``cswap run`` session profile (``CLAUDE_CONFIG_DIR``) maps
    to its slot; otherwise the identity in ``.claude.json`` is matched to a
    slot. Quotaless slots (API-key, base-URL) are skipped. A reading is
    rejected within ``SWITCH_SETTLE_S`` of a switch (default login only: a
    switch never changes a session profile's account), and when its 7-day
    reset contradicts the slot's stored one. No network, no credential read.
    """
    log = switcher._logger
    try:
        payload = json.loads(text)
    except ValueError:
        log.debug("statusline ingest: payload is not JSON")
        return None
    if not isinstance(payload, dict):
        log.debug("statusline ingest: payload is not an object")
        return None
    usage = usage_from_rate_limits(payload.get("rate_limits"))
    if usage is None:
        log.debug("statusline ingest: no usable rate_limits")
        return None

    data = switcher._get_sequence_data()
    if not data or not data.get("accounts"):
        log.debug("statusline ingest: no managed accounts")
        return None
    now = switcher._usage_store.clock()

    num: str | None = None
    in_session = False
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    if config_dir:
        in_session, num = _session_slot(switcher, data, config_dir)
        if in_session and num is None:
            log.debug("statusline ingest: session profile not attributable")
            return None
    if not in_session:
        identity = switcher._get_current_account()
        if identity is None:
            log.debug("statusline ingest: no current login")
            return None
        num = switcher._find_account_slot(data, identity[0], identity[1])
        if num is None:
            log.debug("statusline ingest: current login is not managed")
            return None
    assert num is not None

    if switcher.account_is_quotaless(num):
        log.debug("statusline ingest: slot %s has no quota", num)
        return None

    if not in_session:
        last_switch = _last_switch_at(switcher, data)
        if last_switch is not None and now - last_switch < SWITCH_SETTLE_S:
            log.debug(
                "statusline ingest: switched %.0fs ago, reading may be the "
                "previous account's", now - last_switch,
            )
            return None

    record = data["accounts"][num]
    identities = {num: (record.get("email") or "", record.get("organizationUuid", "") or "")}
    last_good = switcher._usage_store.entries(identities)[num].last_good
    if _weekly_reset_mismatch(usage, last_good, now):
        log.debug(
            "statusline ingest: 7-day reset does not match slot %s's, rejected", num
        )
        return None

    usage = _merge_carried(usage, last_good, now)
    adopted = switcher._usage_store.adopt({num: (usage, 0.0)}, identities, hold_s=None)
    if num not in adopted:
        log.debug("statusline ingest: slot %s already has a newer reading", num)
        return None
    log.debug("statusline ingest: adopted reading for slot %s", num)
    return num


def run_ingest_statusline(debug: bool = False) -> None:
    """CLI entry: read stdin, ingest, and swallow every failure."""
    try:
        text = sys.stdin.read()
    except Exception:  # noqa: BLE001 - a statusline helper must never fail loudly
        return
    try:
        from claude_swap.switcher import ClaudeAccountSwitcher

        switcher = ClaudeAccountSwitcher(debug=debug)
    except Exception:  # noqa: BLE001
        return
    try:
        ingest_statusline(switcher, text)
    except Exception as e:  # noqa: BLE001
        switcher._logger.debug("statusline ingest failed: %r", e)
