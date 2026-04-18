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

from langfuse_codex.config import HookConfig
from langfuse_codex.hook import HookProcessor, _configure_logging
from langfuse_codex.state import SessionStateStore
from langfuse_codex.tracer import LangfuseTracer
from langfuse_codex.transcript import TranscriptParser

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class QueuedEvent:
    event_id: str
    payload: dict[str, Any]
    event_hash: str | None = None


ProcessorFactory = Callable[[], HookProcessor]


class QueueSender:
    def __init__(
        self,
        *,
        config: HookConfig,
        processor_factory: ProcessorFactory | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.logger = logger or LOGGER
        self.queue_dir = config.state_dir / "queue"
        self.inbox_dir = self.queue_dir / "inbox"
        self.processing_dir = self.queue_dir / "processing"
        self.lock_path = self.queue_dir / "sender.lock"
        self.pid_path = self.queue_dir / "sender.pid"
        self._lock_file = None
        self._processor_factory = processor_factory or self._default_processor_factory
        self._processor: HookProcessor | None = None

    def run(self, *, once: bool = False) -> int:
        self._ensure_dirs()
        if not self.acquire_lock():
            self.logger.debug("Langfuse Codex sender already active")
            return 0

        try:
            self.resume_stale_processing()
            idle_started_at = time.monotonic()
            while True:
                event_path = self._next_event_path()
                if event_path is None:
                    if once or (time.monotonic() - idle_started_at) >= self.config.sender_idle_seconds:
                        return 0
                    time.sleep(0.25)
                    continue

                idle_started_at = time.monotonic()
                processed = self.process_event_file(event_path)
                if not processed:
                    return 0
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
        try:
            event = self._read_event(path)
            session_id = event.payload.get("session_id")
            if not isinstance(session_id, str) or not session_id:
                raise ValueError("Queued event missing session_id")

            if self._is_processed(session_id, event):
                path.unlink(missing_ok=True)
                return True

            self.processor.handle(event.payload)
            self._remember_processed(session_id, event)
            path.unlink(missing_ok=True)
            return True
        except Exception:
            self.logger.exception("Failed to process queued Langfuse Codex hook event", extra={"path": str(path)})
            self._return_to_inbox(path)
            return False

    @property
    def processor(self) -> HookProcessor:
        if self._processor is None:
            self._processor = self._processor_factory()
        return self._processor

    def _default_processor_factory(self) -> HookProcessor:
        return HookProcessor(
            config=self.config,
            state_store=SessionStateStore(self.config.state_dir),
            tracer=LangfuseTracer(self.config, logger=self.logger),
            parser=TranscriptParser(self.logger, capture_agent_messages=self.config.capture_agent_messages),
        )

    def _is_processed(self, session_id: str, event: QueuedEvent) -> bool:
        with SessionStateStore(self.config.state_dir).open(session_id) as state:
            return any(state.has_processed_queue_event(marker) for marker in self._event_markers(event))

    def _remember_processed(self, session_id: str, event: QueuedEvent) -> None:
        with SessionStateStore(self.config.state_dir).open(session_id) as state:
            for marker in self._event_markers(event):
                state.remember_queue_event(marker)

    def _event_markers(self, event: QueuedEvent) -> list[str]:
        markers = [f"id:{event.event_id}"]
        if event.event_hash:
            markers.append(f"sha256:{event.event_hash}")
        return markers

    def _next_event_path(self) -> Path | None:
        self._ensure_dirs()
        for path in sorted(self.inbox_dir.glob("*.json")):
            processing_path = self.processing_dir / path.name
            try:
                os.replace(path, processing_path)
            except FileNotFoundError:
                continue
            return processing_path
        return None

    def _read_event(self, path: Path) -> QueuedEvent:
        with path.open("r", encoding="utf-8") as handle:
            envelope = json.load(handle)

        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("Queued event payload must be an object")

        event_id = envelope.get("event_id") or path.stem
        return QueuedEvent(
            event_id=str(event_id),
            payload=payload,
            event_hash=envelope.get("event_hash"),
        )

    def _return_to_inbox(self, path: Path) -> None:
        if not path.exists():
            return
        target = self.inbox_dir / path.name
        if target.exists():
            target = self.inbox_dir / f"{path.stem}.{int(time.time_ns())}.json"
        os.replace(path, target)

    def _ensure_dirs(self) -> None:
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        self.processing_dir.mkdir(parents=True, exist_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cwd = args.cwd or Path.cwd()
    config = HookConfig.from_environment(cwd)
    _configure_logging(config)
    sender = QueueSender(config=config, logger=LOGGER)
    return sender.run(once=args.once)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drain queued Codex hook events into Langfuse")
    parser.add_argument("--once", action="store_true", help="Drain currently queued events and exit")
    parser.add_argument("--cwd", type=Path, help="Project cwd used to discover config")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
