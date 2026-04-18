import ast
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ENQUEUE_SCRIPT = Path(__file__).resolve().parents[1] / ".codex" / "hooks" / "langfuse_enqueue.py"


def _run_enqueue(payload: str, state_dir: Path) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "LANGFUSE_CODEX_DELIVERY_MODE": "queued",
        "LANGFUSE_CODEX_STATE_DIR": str(state_dir),
        "LANGFUSE_CODEX_SENDER_AUTOSTART": "0",
    }
    return subprocess.run(
        [sys.executable, str(ENQUEUE_SCRIPT)],
        input=payload,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )


def test_enqueue_script_writes_one_atomic_queue_file(tmp_path: Path) -> None:
    payload = {
        "session_id": "session-1",
        "hook_event_name": "UserPromptSubmit",
        "turn_id": "turn-1",
        "cwd": str(tmp_path),
        "transcript_path": None,
        "model": "gpt-5.4",
        "prompt": "hello",
    }

    result = _run_enqueue(json.dumps(payload), tmp_path / "state")

    assert result.returncode == 0
    queued = list((tmp_path / "state" / "queue" / "inbox").glob("*.json"))
    assert len(queued) == 1
    envelope = json.loads(queued[0].read_text(encoding="utf-8"))
    assert envelope["payload"] == payload
    assert envelope["event_id"]
    assert envelope["event_hash"]


def test_enqueue_script_fails_open_for_invalid_payload_and_bad_state_dir(tmp_path: Path) -> None:
    bad_state_dir = tmp_path / "state-is-file"
    bad_state_dir.write_text("not a directory", encoding="utf-8")

    result = _run_enqueue("{not-json", bad_state_dir)

    assert result.returncode == 0


def test_enqueue_script_does_not_import_project_or_langfuse_modules() -> None:
    tree = ast.parse(ENQUEUE_SCRIPT.read_text(encoding="utf-8"))
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])

    assert "langfuse" not in imported_roots
    assert "langfuse_codex" not in imported_roots


def test_enqueue_script_reads_queue_settings_from_dotenv(tmp_path: Path, monkeypatch) -> None:
    module = _load_enqueue_module()
    monkeypatch.delenv("LANGFUSE_CODEX_STATE_DIR", raising=False)
    monkeypatch.delenv("LANGFUSE_CODEX_DELIVERY_MODE", raising=False)
    (tmp_path / ".env").write_text(
        "LANGFUSE_CODEX_STATE_DIR='.codex/custom-state'\n"
        "LANGFUSE_CODEX_DELIVERY_MODE=queued\n",
        encoding="utf-8",
    )

    assert module._state_dir(tmp_path) == (tmp_path / ".codex" / "custom-state").resolve()
    assert module._delivery_mode(tmp_path) == "queued"


def test_enqueue_script_completes_without_network_wait(tmp_path: Path) -> None:
    payload = {
        "session_id": "session-1",
        "hook_event_name": "PreToolUse",
        "turn_id": "turn-1",
        "cwd": str(tmp_path),
        "transcript_path": None,
        "model": "gpt-5.4",
        "tool_name": "Bash",
        "tool_use_id": "tool-1",
        "tool_input": {"command": "printf hello"},
    }

    started = time.perf_counter()
    result = _run_enqueue(json.dumps(payload), tmp_path / "state")
    elapsed = time.perf_counter() - started

    assert result.returncode == 0
    assert elapsed < 2.0


def _load_enqueue_module():
    spec = importlib.util.spec_from_file_location("langfuse_enqueue_test", ENQUEUE_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
