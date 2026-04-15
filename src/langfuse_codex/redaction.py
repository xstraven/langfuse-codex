from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

TRUNCATION_MARKER = "\n...[truncated]"
SAFE_KEY_NAMES = {
    "call_id",
    "cwd",
    "duration_ms",
    "exit_code",
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "total_tokens",
    "last_token_usage",
    "max_output_tokens",
    "metadata",
    "model_context_window",
    "query",
    "session_id",
    "status",
    "tool_name",
    "trace_id",
    "turn_id",
    "yield_time_ms",
}
SENSITIVE_KEY_RE = re.compile(
    r"(^|[_-])(api[_-]?key|secret|password|authorization|cookie|private[_-]?key|client[_-]?secret|access[_-]?token|refresh[_-]?token|bearer)([_-]|$)",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"\b(?:sk|pk)-[A-Za-z0-9_-]{8,}\b")
BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._-]{8,}")
AUTH_HEADER_RE = re.compile(r"(?im)^(authorization|x-api-key)\s*:\s*.+$")
ENV_SECRET_LINE_RE = re.compile(
    r"(?im)^([A-Z0-9_]*(?:SECRET|PASSWORD|API_KEY|PRIVATE_KEY|ACCESS_TOKEN|REFRESH_TOKEN)[A-Z0-9_]*)=(.+)$"
)
PATCH_FILE_RE = re.compile(r"(?m)^\*\*\* (?:Add|Update|Delete) File: (.+)$")
PATCH_MOVE_RE = re.compile(r"(?m)^\*\*\* Move to: (.+)$")
ABSOLUTE_OR_EXPLICIT_RELATIVE_PATH_RE = re.compile(r"(?P<path>(?:~?/|/|\.\.?/)[^\s\"'`|<>:,;]+)")
PATH_KEY_NAMES = {
    "path",
    "paths",
    "file",
    "files",
    "filepath",
    "file_path",
    "target",
    "move_path",
}
IGNORED_PATH_KEYS = {"cwd", "directory", "workdir"}
IGNORED_EXECUTABLE_BASENAMES = {"bash", "env", "python", "python3", "sh", "uv", "zsh"}
READ_COMMAND_TYPES = {"read", "open", "view"}
WRITE_COMMAND_TYPES = {"write", "edit", "update", "create", "delete"}
SEARCH_COMMAND_TYPES = {"search", "list_files", "find"}


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


def extract_path_buckets(*values: Any, cwd: str | None = None) -> dict[str, list[str]]:
    buckets: dict[str, set[str]] = {
        "read_paths": set(),
        "write_paths": set(),
        "search_paths": set(),
        "referenced_paths": set(),
    }
    for value in values:
        _collect_paths(value, cwd, buckets, preferred_bucket=None, structured_only=False)
    return {key: sorted(paths) for key, paths in buckets.items() if paths}


def extract_file_paths(*values: Any, cwd: str | None = None) -> list[str]:
    buckets = extract_path_buckets(*values, cwd=cwd)
    union = {
        path
        for key in ("read_paths", "write_paths", "search_paths", "referenced_paths")
        for path in buckets.get(key, [])
    }
    return sorted(union)


def _sanitize_value(value: Any, max_bytes: int, metadata: dict[str, Any]) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, nested in value.items():
            key_str = str(key)
            key_lower = key_str.lower()
            if key_lower not in SAFE_KEY_NAMES and SENSITIVE_KEY_RE.search(key_str):
                result[key_str] = "[REDACTED]"
                metadata["redacted"] = True
                continue
            result[key_str] = _sanitize_value(nested, max_bytes, metadata)
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


def _collect_paths(
    value: Any,
    cwd: str | None,
    buckets: dict[str, set[str]],
    *,
    preferred_bucket: str | None,
    structured_only: bool,
) -> None:
    if isinstance(value, Mapping):
        consumed_keys = _collect_structured_paths(value, cwd, buckets)
        for key, nested in value.items():
            key_lower = str(key).lower()
            if key_lower in consumed_keys:
                continue
            if key_lower in IGNORED_PATH_KEYS:
                continue
            child_bucket = preferred_bucket
            if key_lower in PATH_KEY_NAMES:
                child_bucket = preferred_bucket or "referenced_paths"
            _collect_paths(
                nested,
                cwd,
                buckets,
                preferred_bucket=child_bucket,
                structured_only=structured_only or key_lower in PATH_KEY_NAMES,
            )
        return

    if isinstance(value, (list, tuple, set)):
        for item in value:
            _collect_paths(item, cwd, buckets, preferred_bucket=preferred_bucket, structured_only=structured_only)
        return

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")

    if not isinstance(value, str):
        return

    if _looks_like_patch(value):
        _collect_patch_paths(value, cwd, buckets)
        return

    if structured_only:
        _collect_string_paths(value, cwd, buckets, preferred_bucket or "referenced_paths", fallback=False)
        return

    _collect_string_paths(value, cwd, buckets, preferred_bucket or "referenced_paths", fallback=True)


def _collect_structured_paths(
    value: Mapping[str, Any],
    cwd: str | None,
    buckets: dict[str, set[str]],
) -> set[str]:
    consumed_keys: set[str] = set()

    command_type = str(value.get("type") or "").lower()
    direct_bucket = _bucket_for_command_type(command_type)
    if direct_bucket:
        for key in ("path", "target", "file_path"):
            path = _normalize_path(str(value.get(key) or ""), cwd)
            if path:
                buckets[direct_bucket].add(path)
                consumed_keys.add(key)

    parsed_cmd = value.get("parsed_cmd")
    if isinstance(parsed_cmd, list):
        for item in parsed_cmd:
            if not isinstance(item, Mapping):
                continue
            command_type = str(item.get("type") or "").lower()
            bucket = _bucket_for_command_type(command_type)
            for key in ("path", "target", "file_path"):
                path = _normalize_path(str(item.get(key) or ""), cwd)
                if path and bucket:
                    buckets[bucket].add(path)
        consumed_keys.add("parsed_cmd")

    changes = value.get("changes")
    if isinstance(changes, Mapping):
        for raw_path, change in changes.items():
            path = _normalize_path(str(raw_path), cwd)
            if path:
                buckets["write_paths"].add(path)
            if isinstance(change, Mapping):
                move_path = _normalize_path(str(change.get("move_path") or ""), cwd)
                if move_path:
                    buckets["write_paths"].add(move_path)
        consumed_keys.add("changes")
    return consumed_keys


def _collect_patch_paths(value: str, cwd: str | None, buckets: dict[str, set[str]]) -> None:
    for match in PATCH_FILE_RE.finditer(value):
        normalized = _normalize_path(match.group(1).strip(), cwd)
        if normalized:
            buckets["write_paths"].add(normalized)
    for match in PATCH_MOVE_RE.finditer(value):
        normalized = _normalize_path(match.group(1).strip(), cwd)
        if normalized:
            buckets["write_paths"].add(normalized)


def _collect_string_paths(
    value: str,
    cwd: str | None,
    buckets: dict[str, set[str]],
    bucket: str,
    *,
    fallback: bool,
) -> None:
    candidate = value.strip()
    if not candidate or "://" in candidate:
        return

    normalized = _normalize_path(candidate, cwd)
    if normalized:
        buckets[bucket].add(normalized)
        return

    if not fallback:
        return

    for match in ABSOLUTE_OR_EXPLICIT_RELATIVE_PATH_RE.finditer(candidate):
        normalized = _normalize_path(match.group("path"), cwd)
        if normalized:
            buckets[bucket].add(normalized)


def _bucket_for_command_type(command_type: str) -> str | None:
    if command_type in READ_COMMAND_TYPES:
        return "read_paths"
    if command_type in WRITE_COMMAND_TYPES:
        return "write_paths"
    if command_type in SEARCH_COMMAND_TYPES:
        return "search_paths"
    return "referenced_paths"


def _looks_like_patch(value: str) -> bool:
    return "*** Begin Patch" in value and "*** End Patch" in value


def _normalize_path(value: str, cwd: str | None) -> str | None:
    candidate = value.strip()
    if not candidate:
        return None
    if "://" in candidate or candidate.startswith("www."):
        return None
    if candidate.endswith(":") or "\n" in candidate:
        return None

    expanded = os.path.expanduser(candidate)
    path = Path(expanded)
    if not path.is_absolute():
        if not expanded.startswith(("./", "../")):
            return None
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
