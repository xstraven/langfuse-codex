from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import fcntl

from langfuse_codex.config import HookConfig, discover_project_root, resolve_queue_dir
from langfuse_codex.hook import HookProcessor
from langfuse_codex.state import SessionStateStore
from langfuse_codex.tracer import LangfuseTracer
from langfuse_codex.transcript import TranscriptParser

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ServiceQueuedEvent:
    event_id: str
    project_root: Path
    payload: dict[str, Any]
    event_hash: str | None = None
    created_at_ns: int | None = None
    attempts: int = 0
    next_attempt_at_ns: int = 0


ProcessorFactory = Callable[[HookConfig], HookProcessor]


class CodexLangfuseService:
    def __init__(
        self,
        *,
        queue_dir: Path,
        processor_factory: ProcessorFactory | None = None,
        logger: logging.Logger | None = None,
        idle_sleep_seconds: float = 0.5,
    ) -> None:
        self.queue_dir = queue_dir
        self.inbox_dir = queue_dir / "inbox"
        self.processing_dir = queue_dir / "processing"
        self.tmp_dir = queue_dir / "tmp"
        self.lock_path = queue_dir / "service.lock"
        self.pid_path = queue_dir / "service.pid"
        self._lock_file = None
        self._processor_factory = processor_factory or self._default_processor_factory
        self.logger = logger or LOGGER
        self.idle_sleep_seconds = idle_sleep_seconds

    def run(self, *, once: bool = False) -> int:
        self._ensure_dirs()
        if not self.acquire_lock():
            self.logger.info("Langfuse Codex service already active")
            return 0

        try:
            self.resume_stale_processing()
            while True:
                event_path = self._next_ready_event_path()
                if event_path is None:
                    if once:
                        return 0
                    time.sleep(self.idle_sleep_seconds)
                    continue

                self.process_event_file(event_path)
        finally:
            self.release_lock()

    def acquire_lock(self) -> bool:
        self._ensure_dirs()
        self._lock_file = self.lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock_file.close()
            self._lock_file = None
            return False

        self._lock_file.seek(0)
        self._lock_file.truncate()
        self._lock_file.write(str(os.getpid()))
        self._lock_file.flush()
        self.pid_path.write_text(str(os.getpid()), encoding="utf-8")
        return True

    def release_lock(self) -> None:
        if self._lock_file is None:
            return
        try:
            self.pid_path.unlink(missing_ok=True)
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            self._lock_file.close()
            self._lock_file = None

    def resume_stale_processing(self) -> None:
        self._ensure_dirs()
        for path in sorted(self.processing_dir.glob("*.json")):
            target = self.inbox_dir / path.name
            if target.exists():
                target = self.inbox_dir / f"{path.stem}.{os.getpid()}.json"
            os.replace(path, target)

    def process_event_file(self, path: Path) -> bool:
        event: ServiceQueuedEvent | None = None
        try:
            event = self._read_event(path)
            self._process_event(event)
            path.unlink(missing_ok=True)
            return True
        except Exception:
            self.logger.exception("Failed to process queued Langfuse Codex hook event", extra={"path": str(path)})
            if event is None:
                event = self._fallback_failed_event(path)
            self._return_to_inbox_with_backoff(path, event)
            return False

    def _process_event(self, event: ServiceQueuedEvent) -> None:
        config = HookConfig.from_service_environment(event.project_root)
        session_id = event.payload.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("Queued event missing session_id")

        if self._is_processed(config, session_id, event):
            return

        processor = self._processor_factory(config)
        processor.handle(event.payload)
        self._remember_processed(config, session_id, event)

    def _default_processor_factory(self, config: HookConfig) -> HookProcessor:
        return HookProcessor(
            config=config,
            state_store=SessionStateStore(config.state_dir),
            tracer=LangfuseTracer(config, logger=self.logger),
            parser=TranscriptParser(self.logger, capture_agent_messages=config.capture_agent_messages),
        )

    def _is_processed(self, config: HookConfig, session_id: str, event: ServiceQueuedEvent) -> bool:
        with SessionStateStore(config.state_dir).open(session_id) as state:
            return any(state.has_processed_queue_event(marker) for marker in self._event_markers(event))

    def _remember_processed(self, config: HookConfig, session_id: str, event: ServiceQueuedEvent) -> None:
        with SessionStateStore(config.state_dir).open(session_id) as state:
            for marker in self._event_markers(event):
                state.remember_queue_event(marker)

    def _event_markers(self, event: ServiceQueuedEvent) -> list[str]:
        markers = [f"id:{event.event_id}"]
        if event.event_hash:
            markers.append(f"sha256:{event.event_hash}")
        return markers

    def _next_ready_event_path(self) -> Path | None:
        self._ensure_dirs()
        now = time.time_ns()
        for path in sorted(self.inbox_dir.glob("*.json")):
            if not self._event_ready(path, now):
                continue
            processing_path = self.processing_dir / path.name
            try:
                os.replace(path, processing_path)
            except FileNotFoundError:
                continue
            return processing_path
        return None

    def _event_ready(self, path: Path, now_ns: int) -> bool:
        try:
            with path.open("r", encoding="utf-8") as handle:
                envelope = json.load(handle)
        except Exception:
            return True
        next_attempt_at_ns = envelope.get("next_attempt_at_ns", 0)
        return not isinstance(next_attempt_at_ns, int) or next_attempt_at_ns <= now_ns

    def _read_event(self, path: Path) -> ServiceQueuedEvent:
        with path.open("r", encoding="utf-8") as handle:
            envelope = json.load(handle)

        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("Queued event payload must be an object")

        project_root = envelope.get("project_root") or payload.get("cwd") or Path.cwd()
        return ServiceQueuedEvent(
            event_id=str(envelope.get("event_id") or path.stem),
            event_hash=envelope.get("event_hash"),
            created_at_ns=_optional_int(envelope.get("created_at_ns")),
            project_root=discover_project_root(Path(str(project_root))),
            payload=payload,
            attempts=int(envelope.get("attempts", 0)),
            next_attempt_at_ns=int(envelope.get("next_attempt_at_ns", 0)),
        )

    def _fallback_failed_event(self, path: Path) -> ServiceQueuedEvent:
        return ServiceQueuedEvent(
            event_id=path.stem,
            project_root=Path.cwd(),
            payload={"session_id": "malformed", "hook_event_name": "MalformedQueueEvent"},
        )

    def _return_to_inbox_with_backoff(self, path: Path, event: ServiceQueuedEvent) -> None:
        event.attempts += 1
        delay_seconds = min(300, 2 ** min(event.attempts - 1, 8))
        event.next_attempt_at_ns = time.time_ns() + int(delay_seconds * 1_000_000_000)

        target = self.inbox_dir / path.name
        if target.exists():
            target = self.inbox_dir / f"{path.stem}.{time.time_ns()}.json"
        self._write_event(target, event)
        path.unlink(missing_ok=True)

    def _write_event(self, path: Path, event: ServiceQueuedEvent) -> None:
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        envelope = {
            "event_id": event.event_id,
            "event_hash": event.event_hash,
            "created_at_ns": event.created_at_ns,
            "project_root": str(event.project_root),
            "payload": event.payload,
            "attempts": event.attempts,
            "next_attempt_at_ns": event.next_attempt_at_ns,
        }
        tmp_path = self.tmp_dir / f"{path.name}.{os.getpid()}.tmp"
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(envelope, handle, separators=(",", ":"), sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)

    def _ensure_dirs(self) -> None:
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        self.processing_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_service_logging()
    service = CodexLangfuseService(queue_dir=args.queue_dir)
    return service.run(once=args.once)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the global Codex Langfuse queue service")
    parser.add_argument("--once", action="store_true", help="Drain currently ready events and exit")
    parser.add_argument(
        "--queue-dir",
        type=Path,
        default=resolve_queue_dir(),
        help="Global queue directory to drain",
    )
    return parser.parse_args(argv)


def _configure_service_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LANGFUSE_CODEX_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
