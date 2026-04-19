from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

_ENV_CACHE: dict[Path, dict[str, str]] = {}


def main() -> int:
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("hook payload must be a JSON object")

        project_root = _project_root_from_payload(payload)
        values = _settings(project_root)
        queue_dir = _queue_dir(values)
        queue_file = _enqueue_payload(project_root, queue_dir, raw, payload, values)
        if queue_file is None:
            _log_failure(queue_dir, "queue full; dropping newest event")
    except Exception as error:
        _log_failure(_default_queue_dir(), f"{type(error).__name__}: {error}")
    return 0


def _project_root_from_payload(payload: dict) -> Path:
    cwd = payload.get("cwd")
    start = Path(cwd) if isinstance(cwd, str) and cwd else Path.cwd()
    return _discover_project_root(start)


def _discover_project_root(start: Path) -> Path:
    current = start.expanduser().resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
        if (candidate / ".env").exists() and (candidate / "README.md").exists():
            return candidate
    return current


def _settings(project_root: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    values.update(_read_dotenv(Path(os.environ.get("LANGFUSE_CODEX_CONFIG", _default_config_path()))))
    values.update(_read_dotenv(project_root / ".env"))
    values.update(os.environ)
    return values


def _queue_dir(values: dict[str, str]) -> Path:
    configured = values.get("LANGFUSE_CODEX_QUEUE_DIR")
    path = Path(configured).expanduser() if configured else _default_queue_dir()
    return path if path.is_absolute() else (_default_app_support_dir() / path).resolve()


def _default_queue_dir() -> Path:
    return _default_app_support_dir() / "queue"


def _default_config_path() -> Path:
    return _default_app_support_dir() / "env"


def _default_app_support_dir() -> Path:
    configured = os.environ.get("LANGFUSE_CODEX_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "langfuse-codex"
    return Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "langfuse-codex"


def _read_dotenv(path: Path) -> dict[str, str]:
    path = path.expanduser()
    if path in _ENV_CACHE:
        return _ENV_CACHE[path]

    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        _ENV_CACHE[path] = values
        return values

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        if stripped.startswith("export "):
            stripped = stripped.removeprefix("export ").lstrip()
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key:
            values[key] = _strip_env_value(value)

    _ENV_CACHE[path] = values
    return values


def _strip_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _enqueue_payload(
    project_root: Path,
    queue_dir: Path,
    raw: str,
    payload: dict,
    values: dict[str, str],
) -> Path | None:
    inbox_dir = queue_dir / "inbox"
    tmp_dir = queue_dir / "tmp"
    inbox_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    max_events = int(values.get("LANGFUSE_CODEX_QUEUE_MAX_EVENTS", "10000"))
    if max_events > 0 and _queue_size_at_least(queue_dir, max_events):
        return None

    event_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    event_id = uuid.uuid4().hex
    filename = f"{time.time_ns()}-{os.getpid()}-{event_hash[:12]}-{event_id}.json"
    envelope = {
        "event_id": event_id,
        "event_hash": event_hash,
        "created_at_ns": time.time_ns(),
        "project_root": str(project_root),
        "payload": payload,
    }

    tmp_path = tmp_dir / f"{filename}.tmp"
    final_path = inbox_dir / filename
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(envelope, handle, separators=(",", ":"), sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, final_path)
    return final_path


def _queue_size_at_least(queue_dir: Path, limit: int) -> bool:
    count = 0
    for subdir in ("inbox", "processing"):
        path = queue_dir / subdir
        if not path.exists():
            continue
        for _ in path.glob("*.json"):
            count += 1
            if count >= limit:
                return True
    return False


def _log_failure(queue_dir: Path, message: str) -> None:
    log_path = queue_dir / "enqueue.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        log_path = Path(tempfile.gettempdir()) / "langfuse-codex-enqueue.log"
    try:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}\n")
    except Exception:
        return


if __name__ == "__main__":
    raise SystemExit(main())
