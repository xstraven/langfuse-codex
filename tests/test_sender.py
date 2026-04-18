import json
from pathlib import Path

from langfuse_codex.sender import QueueSender


class FakeConfig:
    def __init__(self, tmp_path: Path):
        self.project_root = tmp_path
        self.state_dir = tmp_path / "state"
        self.sender_idle_seconds = 0.01


class FakeProcessor:
    def __init__(self, seen: list[str], *, fail_on: str | None = None):
        self.seen = seen
        self.fail_on = fail_on

    def handle(self, payload):
        value = payload["value"]
        if value == self.fail_on:
            raise RuntimeError("boom")
        self.seen.append(value)


def _write_event(
    queue_dir: Path,
    name: str,
    payload: dict,
    *,
    event_id: str | None = None,
    event_hash: str | None = None,
) -> Path:
    inbox = queue_dir / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    path = inbox / name
    path.write_text(
        json.dumps(
            {
                "event_id": event_id or name.removesuffix(".json"),
                "event_hash": event_hash or f"hash-{name}",
                "payload": payload,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_sender_drains_events_in_order_and_marks_processed(tmp_path: Path) -> None:
    config = FakeConfig(tmp_path)
    seen: list[str] = []
    queue_dir = config.state_dir / "queue"
    payload_one = {"session_id": "session-1", "hook_event_name": "SessionStart", "value": "one"}
    payload_two = {"session_id": "session-1", "hook_event_name": "Stop", "value": "two"}
    _write_event(queue_dir, "001.json", payload_one, event_id="event-1")
    _write_event(queue_dir, "002.json", payload_two, event_id="event-2")

    sender = QueueSender(config=config, processor_factory=lambda: FakeProcessor(seen))

    assert sender.run(once=True) == 0
    assert seen == ["one", "two"]
    assert not list((queue_dir / "inbox").glob("*.json"))
    assert not list((queue_dir / "processing").glob("*.json"))


def test_sender_skips_already_processed_event(tmp_path: Path) -> None:
    config = FakeConfig(tmp_path)
    seen: list[str] = []
    queue_dir = config.state_dir / "queue"
    _write_event(
        queue_dir,
        "001.json",
        {"session_id": "session-1", "hook_event_name": "Stop", "value": "one"},
        event_id="event-1",
    )
    sender = QueueSender(config=config, processor_factory=lambda: FakeProcessor(seen))
    assert sender.run(once=True) == 0

    _write_event(
        queue_dir,
        "001-again.json",
        {"session_id": "session-1", "hook_event_name": "Stop", "value": "one-again"},
        event_id="event-1",
    )

    assert sender.run(once=True) == 0
    assert seen == ["one"]


def test_sender_skips_already_processed_event_hash(tmp_path: Path) -> None:
    config = FakeConfig(tmp_path)
    seen: list[str] = []
    queue_dir = config.state_dir / "queue"
    _write_event(
        queue_dir,
        "001.json",
        {"session_id": "session-1", "hook_event_name": "Stop", "value": "one"},
        event_id="event-1",
        event_hash="same-hash",
    )
    sender = QueueSender(config=config, processor_factory=lambda: FakeProcessor(seen))
    assert sender.run(once=True) == 0

    _write_event(
        queue_dir,
        "001-again.json",
        {"session_id": "session-1", "hook_event_name": "Stop", "value": "one-again"},
        event_id="event-2",
        event_hash="same-hash",
    )

    assert sender.run(once=True) == 0
    assert seen == ["one"]


def test_sender_resumes_stale_processing_files(tmp_path: Path) -> None:
    config = FakeConfig(tmp_path)
    queue_dir = config.state_dir / "queue"
    processing = queue_dir / "processing"
    processing.mkdir(parents=True, exist_ok=True)
    stale = processing / "001.json"
    stale.write_text(
        json.dumps(
            {
                "event_id": "event-1",
                "payload": {"session_id": "session-1", "hook_event_name": "Stop", "value": "one"},
            }
        ),
        encoding="utf-8",
    )

    sender = QueueSender(config=config, processor_factory=lambda: FakeProcessor([]))
    sender.resume_stale_processing()

    assert not stale.exists()
    assert (queue_dir / "inbox" / "001.json").exists()


def test_sender_lock_prevents_second_active_sender(tmp_path: Path) -> None:
    config = FakeConfig(tmp_path)
    first = QueueSender(config=config, processor_factory=lambda: FakeProcessor([]))
    second = QueueSender(config=config, processor_factory=lambda: FakeProcessor([]))

    assert first.acquire_lock() is True
    try:
        assert second.acquire_lock() is False
    finally:
        first.release_lock()


def test_sender_leaves_failed_event_retryable(tmp_path: Path) -> None:
    config = FakeConfig(tmp_path)
    seen: list[str] = []
    queue_dir = config.state_dir / "queue"
    _write_event(
        queue_dir,
        "001.json",
        {"session_id": "session-1", "hook_event_name": "Stop", "value": "bad"},
        event_id="event-1",
    )

    sender = QueueSender(config=config, processor_factory=lambda: FakeProcessor(seen, fail_on="bad"))

    assert sender.run(once=True) == 0
    assert seen == []
    assert list((queue_dir / "inbox").glob("*.json"))
