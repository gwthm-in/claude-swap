"""Claude Code's user ``settings.json``: the one key cswap manages in it.

Base-URL accounts (``cswap add-token KEY --base-url URL``) point Claude Code at
a relay or gateway. For the default login that is done through
``env.ANTHROPIC_BASE_URL`` in ``<config_home>/settings.json``, which Claude
Code applies over the process environment at startup. Everything else in the
file belongs to the user and is carried through untouched.

The file is frequently a symlink into a dotfiles repository, so writes go
THROUGH the link to its target (a rename onto the link would detach it), and
the directory it lives in is never chmod'ed: ``~/.claude`` is Claude Code's.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
from pathlib import Path

from claude_swap.exceptions import ConfigError
from claude_swap.fsutil import read_text_with_retry, replace_with_retry

BASE_URL_ENV = "ANTHROPIC_BASE_URL"
ANTHROPIC_API_URL = "https://api.anthropic.com"


def read_settings(path: Path) -> dict | None:
    """Parse ``settings.json``. None when ABSENT; ConfigError when unreadable.

    Absent and unreadable are kept apart for the same reason
    ``CredentialStore._update_global_config`` keeps them apart: an absent
    file is a genuine empty start, an unreadable one must never be
    overwritten with a file that dropped everything it held.
    """
    if not path.exists():
        return None
    try:
        data = json.loads(read_text_with_retry(path))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ConfigError(
            f"{path} exists but could not be parsed ({e}). Repair it, then retry."
        ) from e
    except OSError as e:
        raise ConfigError(f"{path} exists but could not be read ({e}).") from e
    if not isinstance(data, dict):
        raise ConfigError(
            f"{path} holds {type(data).__name__}, not a JSON object. "
            "Repair it, then retry."
        )
    return data


def base_url_from_settings(data: dict | None) -> str | None:
    """``env.ANTHROPIC_BASE_URL`` from parsed settings, or None when unset."""
    if not data:
        return None
    env = data.get("env")
    if not isinstance(env, dict):
        return None
    value = env.get(BASE_URL_ENV)
    return value if isinstance(value, str) and value else None


def read_base_url(path: Path) -> str | None:
    """The configured ``env.ANTHROPIC_BASE_URL``; raises ConfigError when unreadable."""
    return base_url_from_settings(read_settings(path))


def write_base_url(path: Path, value: str | None) -> bool:
    """Set (``value``) or remove (``None``) ``env.ANTHROPIC_BASE_URL``.

    Read-mutate-write touching only that key. Returns whether the file
    changed; an already-matching file is not rewritten. Raises ConfigError
    when the file exists but cannot be parsed, or when the write fails.
    """
    target = Path(os.path.realpath(path)) if path.is_symlink() else path
    try:
        original_text = read_text_with_retry(target) if target.exists() else None
    except OSError as e:
        raise ConfigError(f"{path} exists but could not be read ({e}).") from e
    data = read_settings(target) if original_text is not None else None
    if base_url_from_settings(data) == value:
        return False
    if data is None:
        data = {}
    env = data.get("env")
    if value is None:
        if not isinstance(env, dict):
            return False
        env.pop(BASE_URL_ENV, None)
        if not env:
            data.pop("env", None)
    else:
        if env is not None and not isinstance(env, dict):
            raise ConfigError(
                f"{path} has a non-object 'env' entry; refusing to rewrite it. "
                "Repair it, then retry."
            )
        if env is None:
            env = {}
            data["env"] = env
        env[BASE_URL_ENV] = value

    content = json.dumps(data, indent=2, ensure_ascii=False)
    if original_text is None or original_text.endswith("\n"):
        content += "\n"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    except OSError as e:
        raise ConfigError(f"Could not write {path}: {e}") from e
    try:
        os.write(fd, content.encode("utf-8"))
        os.close(fd)
        fd = -1
        if sys.platform != "win32" and original_text is not None:
            os.chmod(tmp_path, stat.S_IMODE(os.stat(target).st_mode))
        replace_with_retry(tmp_path, str(target))
    except BaseException as e:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        if isinstance(e, OSError):
            raise ConfigError(f"Could not write {path}: {e}") from e
        raise
    return True
