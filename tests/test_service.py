import json
import time
from pathlib import Path

from langfuse_codex.service import CodexLangfuseService


class FakeProcessor:
    def __init__(self, seen: list[tuple[str, str, str | None]], config, *, fail: bool = False):
        self.seen = seen
        self.config = config
        self.fail = fail

    def handle(self, payload):
        if self.fail:
            raise RuntimeError("boom")
        self.seen.append((str(self.config.project_root), str(self.config.state_dir), self.config.public_key))


def _make_project(root: Path, name: str) -> Path:
    project = root / name
    project.mkdir()
    (project / ".git").mkdir()
    return project


def _write_event(
    queue_dir: Path,
    name: str,
    project_root: Path,
    payload: dict,
    *,
    event_id: str | None = None,
    event_hash: str | None = None,
    attempts: int = 0,
    next_attempt_at_ns: int = 0,
) -> Path:
    inbox = queue_dir / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    path = inbox / name
    path.write_text(
        json.dumps(
            {
                "event_id": event_id or name.removesuffix(".json"),
                "event_hash": event_hash or f"hash-{name}",
                "project_root": str(project_root),
                "created_at_ns": time.time_ns(),
                "attempts": attempts,
                "next_attempt_at_ns": next_attempt_at_ns,
                "payload": payload,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_service_drains_multiple_projects_without_mixing_state(tmp_path: Path, monkeypatch) -> None:
    state_root = tmp_path / "state-root"
    queue_dir = tmp_path / "queue"
    project_one = _make_project(tmp_path, "one")
    project_two = _make_project(tmp_path, "two")
    monkeypatch.setenv("LANGFUSE_CODEX_STATE_ROOT", str(state_root))

    _write_event(
        queue_dir,
        "001.json",
        project_one,
        {"session_id": "session-1", "hook_event_name": "SessionStart", "cwd": str(project_one)},
    )
    _write_event(
        queue_dir,
        "002.json",
        project_two,
        {"session_id": "session-2", "hook_event_name": "SessionStart", "cwd": str(project_two)},
    )

    seen: list[tuple[str, str, str | None]] = []
    service = CodexLangfuseService(
        queue_dir=queue_dir,
        processor_factory=lambda config: FakeProcessor(seen, config),
    )

    assert service.run(once=True) == 0
    assert [item[0] for item in seen] == [str(project_one), str(project_two)]
    state_dirs = [item[1] for item in seen]
    assert len(set(state_dirs)) == 2
    assert all(str(state_root) in state_dir for state_dir in state_dirs)
    assert not list((queue_dir / "inbox").glob("*.json"))
    assert not list((queue_dir / "processing").glob("*.json"))


def test_service_project_env_overrides_global_config(tmp_path: Path, monkeypatch) -> None:
    queue_dir = tmp_path / "queue"
    project = _make_project(tmp_path, "project")
    config_path = tmp_path / "global.env"
    config_path.write_text(
        "LANGFUSE_PUBLIC_KEY=global-public\n"
        "LANGFUSE_SECRET_KEY=global-secret\n"
        "LANGFUSE_BASE_URL=https://global.example\n",
        encoding="utf-8",
    )
    (project / ".env").write_text("LANGFUSE_PUBLIC_KEY=project-public\n", encoding="utf-8")
    monkeypatch.setenv("LANGFUSE_CODEX_CONFIG", str(config_path))
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)

    _write_event(
        queue_dir,
        "001.json",
        project,
        {"session_id": "session-1", "hook_event_name": "SessionStart", "cwd": str(project)},
    )
    seen: list[tuple[str, str, str | None]] = []

    service = CodexLangfuseService(
        queue_dir=queue_dir,
        processor_factory=lambda config: FakeProcessor(seen, config),
    )

    assert service.run(once=True) == 0
    assert seen[0][2] == "project-public"


def test_service_failed_event_is_requeued_with_backoff(tmp_path: Path, monkeypatch) -> None:
    queue_dir = tmp_path / "queue"
    project = _make_project(tmp_path, "project")
    monkeypatch.setenv("LANGFUSE_CODEX_STATE_ROOT", str(tmp_path / "state-root"))
    _write_event(
        queue_dir,
        "001.json",
        project,
        {"session_id": "session-1", "hook_event_name": "Stop", "cwd": str(project)},
    )

    service = CodexLangfuseService(
        queue_dir=queue_dir,
        processor_factory=lambda config: FakeProcessor([], config, fail=True),
    )

    assert service.run(once=True) == 0
    queued = list((queue_dir / "inbox").glob("*.json"))
    assert len(queued) == 1
    envelope = json.loads(queued[0].read_text(encoding="utf-8"))
    assert envelope["attempts"] == 1
    assert envelope["next_attempt_at_ns"] > time.time_ns()
    assert not list((queue_dir / "processing").glob("*.json"))


def test_service_skips_duplicate_queue_event(tmp_path: Path, monkeypatch) -> None:
    queue_dir = tmp_path / "queue"
    project = _make_project(tmp_path, "project")
    monkeypatch.setenv("LANGFUSE_CODEX_STATE_ROOT", str(tmp_path / "state-root"))
    payload = {"session_id": "session-1", "hook_event_name": "SessionStart", "cwd": str(project)}
    _write_event(queue_dir, "001.json", project, payload, event_id="same-id")

    seen: list[tuple[str, str, str | None]] = []
    service = CodexLangfuseService(
        queue_dir=queue_dir,
        processor_factory=lambda config: FakeProcessor(seen, config),
    )
    assert service.run(once=True) == 0

    _write_event(queue_dir, "002.json", project, payload, event_id="same-id")

    assert service.run(once=True) == 0
    assert len(seen) == 1
