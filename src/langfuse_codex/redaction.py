from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

TRUNCATION_MARKER = "\n...[truncated]"
SENSITIVE_KEY_RE = re.compile(
    r"(api[_-]?key|secret|token|password|authorization|cookie|session|private[_-]?key)",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"\b(?:sk|pk)-[A-Za-z0-9_-]{8,}\b")
BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._-]{8,}")
AUTH_HEADER_RE = re.compile(r"(?im)^(authorization|x-api-key)\s*:\s*.+$")
ENV_SECRET_LINE_RE = re.compile(
    r"(?im)^([A-Z0-9_]*(?:SECRET|TOKEN|PASSWORD|API_KEY|PRIVATE_KEY)[A-Z0-9_]*)=(.+)$"
)
PATH_KEY_NAMES = {
    "path",
    "paths",
    "file",
    "files",
    "filepath",
    "file_path",
    "target",
    "name",
}
IGNORED_PATH_KEYS = {"cwd", "directory", "workdir"}
IGNORED_EXECUTABLE_BASENAMES = {"bash", "env", "python", "python3", "sh", "uv", "zsh"}
PATH_RE = re.compile(
    r"(?P<path>(?:~?/|/)[^\s\"'`|<>]+|(?:\.\.?/)[^\s\"'`|<>]+|[A-Za-z0-9_.-]+/[^\s\"'`|<>]+)"
)


def truncate_string(value: str, max_bytes: int) -> tuple[str, bool]:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value, False

    truncated = encoded[: max_bytes - len(TRUNCATION_MARKER.encode("utf-8"))]
    while truncated:
        try:
            preview = truncated.decode("utf-8")
            break
        except UnicodeDecodeError:
            truncated = truncated[:-1]
    else:
        preview = ""

    return preview + TRUNCATION_MARKER, True


def redact_string(value: str) -> tuple[str, bool]:
    redacted = value
    changed = False

    for pattern, replacement in (
        (TOKEN_RE, "[REDACTED_TOKEN]"),
        (BEARER_RE, "Bearer [REDACTED]"),
        (AUTH_HEADER_RE, "\\1: [REDACTED]"),
        (ENV_SECRET_LINE_RE, "\\1=[REDACTED]"),
    ):
        updated, count = pattern.subn(replacement, redacted)
        if count:
            redacted = updated
            changed = True

    return redacted, changed


def sanitize_payload(value: Any, max_bytes: int) -> tuple[Any, dict[str, Any]]:
    metadata = {"truncated": False, "redacted": False}
    sanitized = _sanitize_value(value, max_bytes, metadata)
    return sanitized, metadata


def _sanitize_value(value: Any, max_bytes: int, metadata: dict[str, Any]) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, nested in value.items():
            if SENSITIVE_KEY_RE.search(str(key)):
                result[str(key)] = "[REDACTED]"
                metadata["redacted"] = True
                continue
            result[str(key)] = _sanitize_value(nested, max_bytes, metadata)
        return result

    if isinstance(value, list):
        return [_sanitize_value(item, max_bytes, metadata) for item in value]

    if isinstance(value, tuple):
        return [_sanitize_value(item, max_bytes, metadata) for item in value]

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")

    if isinstance(value, str):
        redacted, changed = redact_string(value)
        truncated, was_truncated = truncate_string(redacted, max_bytes)
        metadata["redacted"] = metadata["redacted"] or changed
        metadata["truncated"] = metadata["truncated"] or was_truncated
        return truncated

    return value


def extract_file_paths(*values: Any, cwd: str | None = None) -> list[str]:
    discovered: set[str] = set()
    for value in values:
        _collect_paths(value, cwd, discovered)
    return sorted(discovered)


def _collect_paths(value: Any, cwd: str | None, sink: set[str]) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).lower() in IGNORED_PATH_KEYS:
                continue
            if str(key).lower() in PATH_KEY_NAMES:
                _collect_string_path(str(nested), cwd, sink)
            _collect_paths(nested, cwd, sink)
        return

    if isinstance(value, (list, tuple, set)):
        for item in value:
            _collect_paths(item, cwd, sink)
        return

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")

    if isinstance(value, str):
        _collect_string_path(value, cwd, sink)


def _collect_string_path(value: str, cwd: str | None, sink: set[str]) -> None:
    candidate = value.strip()
    if not candidate:
        return

    if "\n" not in candidate and _looks_like_path(candidate):
        normalized = _normalize_path(candidate, cwd)
        if normalized:
            sink.add(normalized)
        return

    for match in PATH_RE.finditer(candidate):
        normalized = _normalize_path(match.group("path"), cwd)
        if normalized:
            sink.add(normalized)


def _looks_like_path(value: str) -> bool:
    if len(value) > 4096:
        return False
    if value.startswith(("/", "./", "../", "~/")):
        return True
    return "/" in value and " " not in value


def _normalize_path(value: str, cwd: str | None) -> str | None:
    if not value or value.startswith("http://") or value.startswith("https://"):
        return None

    expanded = os.path.expanduser(value)
    path = Path(expanded)
    if not path.is_absolute():
        if not cwd:
            return None
        path = Path(cwd) / path

    normalized = os.path.abspath(path)
    if _should_ignore_path(normalized):
        return None
    return normalized


def _should_ignore_path(path: str) -> bool:
    normalized = Path(path)
    if normalized.name in IGNORED_EXECUTABLE_BASENAMES and normalized.parent.name == "bin":
        return True
    return False
