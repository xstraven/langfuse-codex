import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import langfuse_codex.enqueue as enqueue


def _run_global_enqueue(payload: dict, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", "from langfuse_codex.enqueue import main; raise SystemExit(main())"],
        input=json.dumps(payload),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, **env},
        check=False,
    )


def test_global_enqueue_writes_atomic_queue_file_with_project_metadata(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / ".git").mkdir()
    queue_dir = tmp_path / "queue"
    payload = {
        "session_id": "session-1",
        "hook_event_name": "UserPromptSubmit",
        "turn_id": "turn-1",
        "cwd": str(project),
        "transcript_path": None,
        "model": "gpt-5.4",
        "prompt": "hello",
    }

    result = _run_global_enqueue(payload, {"LANGFUSE_CODEX_QUEUE_DIR": str(queue_dir)})

    assert result.returncode == 0
    queued = list((queue_dir / "inbox").glob("*.json"))
    assert len(queued) == 1
    envelope = json.loads(queued[0].read_text(encoding="utf-8"))
    assert envelope["payload"] == payload
    assert envelope["project_root"] == str(project)
    assert envelope["event_id"]
    assert envelope["event_hash"]
    assert envelope["created_at_ns"]


def test_global_enqueue_module_uses_only_stdlib_imports() -> None:
    tree = ast.parse(Path(enqueue.__file__).read_text(encoding="utf-8"))
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])

    assert imported_roots <= set(sys.stdlib_module_names) | {"__future__"}


def test_global_enqueue_reads_queue_dir_from_global_config(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config_path = tmp_path / "config.env"
    queue_dir = tmp_path / "configured-queue"
    config_path.write_text(f"LANGFUSE_CODEX_QUEUE_DIR='{queue_dir}'\n", encoding="utf-8")
    payload = {
        "session_id": "session-1",
        "hook_event_name": "Stop",
        "cwd": str(project),
    }

    result = _run_global_enqueue(payload, {"LANGFUSE_CODEX_CONFIG": str(config_path)})

    assert result.returncode == 0
    assert len(list((queue_dir / "inbox").glob("*.json"))) == 1
