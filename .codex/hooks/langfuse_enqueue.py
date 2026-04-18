from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - Codex hooks are currently disabled on Windows.
    fcntl = None

_ENV_CACHE: dict[Path, dict[str, str]] = {}


def main() -> int:
    raw = sys.stdin.read()
    project_root = Path(__file__).resolve().parents[2]
    state_dir = _state_dir(project_root)

    try:
        if _delivery_mode(project_root) == "direct":
            _run_direct(project_root, raw)
            return 0

        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("hook payload must be a JSON object")

        queue_file = _enqueue_payload(project_root, state_dir, raw, payload)
        if queue_file is not None and _autostart_sender(project_root):
            _spawn_sender_if_inactive(project_root, state_dir)
    except Exception as error:
        _log_failure(state_dir, f"{type(error).__name__}: {error}")

    return 0


def _delivery_mode(project_root: Path) -> str:
    return _env(project_root, "LANGFUSE_CODEX_DELIVERY_MODE", "queued").lower()


def _autostart_sender(project_root: Path) -> bool:
    disabled_values = {"0", "false", "no"}
    return _env(project_root, "LANGFUSE_CODEX_SENDER_AUTOSTART", "1").lower() not in disabled_values


def _state_dir(project_root: Path) -> Path:
    configured = _env(project_root, "LANGFUSE_CODEX_STATE_DIR", "")
    path = Path(configured) if configured else project_root / ".codex" / "langfuse-state"
    return path if path.is_absolute() else (project_root / path).resolve()


def _env(project_root: Path, key: str, default: str) -> str:
    value = os.environ.get(key)
    if value is not None:
        return value
    return _read_dotenv(project_root).get(key, default)


def _read_dotenv(project_root: Path) -> dict[str, str]:
    project_root = project_root.resolve()
    if project_root in _ENV_CACHE:
        return _ENV_CACHE[project_root]

    values: dict[str, str] = {}
    dotenv_path = project_root / ".env"
    try:
        lines = dotenv_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        _ENV_CACHE[project_root] = values
        return values

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        if stripped.startswith("export "):
            stripped = stripped.removeprefix("export ").lstrip()
        key, value = stripped.split("=", 1)
        key = key.strip()
        if not key:
            continue
        values[key] = _strip_env_value(value)

    _ENV_CACHE[project_root] = values
    return values


def _strip_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _enqueue_payload(project_root: Path, state_dir: Path, raw: str, payload: dict) -> Path | None:
    queue_dir = state_dir / "queue"
    inbox_dir = queue_dir / "inbox"
    tmp_dir = queue_dir / "tmp"
    inbox_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    max_events = int(_env(project_root, "LANGFUSE_CODEX_QUEUE_MAX_EVENTS", "10000"))
    if max_events > 0 and _queue_size_at_least(queue_dir, max_events):
        oldest = _oldest_event(queue_dir)
        _log_failure(state_dir, f"queue full; dropping newest event; oldest={oldest}")
        return None

    event_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    event_id = uuid.uuid4().hex
    filename = f"{time.time_ns()}-{os.getpid()}-{event_hash[:12]}-{event_id}.json"
    envelope = {
        "event_id": event_id,
        "event_hash": event_hash,
        "created_at_ns": time.time_ns(),
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


def _oldest_event(queue_dir: Path) -> str | None:
    candidates = []
    for subdir in ("inbox", "processing"):
        path = queue_dir / subdir
        if path.exists():
            candidates.extend(path.glob("*.json"))
    if not candidates:
        return None
    return str(min(candidates, key=lambda path: path.name))


def _spawn_sender_if_inactive(project_root: Path, state_dir: Path) -> None:
    queue_dir = state_dir / "queue"
    queue_dir.mkdir(parents=True, exist_ok=True)
    if _sender_pid_active(queue_dir):
        return

    lock_path = queue_dir / "sender.lock"
    lock_file = lock_path.open("a+", encoding="utf-8")
    locked = False
    try:
        if fcntl is not None:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except BlockingIOError:
                return

        if _sender_pid_active(queue_dir):
            return

        log_path = queue_dir / "sender.log"
        with log_path.open("a", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                [
                    "uv",
                    "run",
                    "--project",
                    str(project_root),
                    "codex-langfuse-sender",
                    "--cwd",
                    str(project_root),
                ],
                cwd=str(project_root),
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=log_handle,
                start_new_session=True,
                close_fds=True,
            )
        (queue_dir / "sender.pid").write_text(str(process.pid), encoding="utf-8")
    finally:
        if locked:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        lock_file.close()


def _sender_pid_active(queue_dir: Path) -> bool:
    try:
        pid = int((queue_dir / "sender.pid").read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _run_direct(project_root: Path, raw: str) -> None:
    subprocess.run(
        ["uv", "run", "--project", str(project_root), "codex-langfuse-hook"],
        input=raw,
        text=True,
        cwd=str(project_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _log_failure(state_dir: Path, message: str) -> None:
    log_path = state_dir / "queue" / "enqueue.log"
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
