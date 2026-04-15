from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import pytest
from dotenv import load_dotenv
from langfuse import Langfuse

from langfuse_codex.config import HookConfig
from langfuse_codex.hook import HookProcessor
from langfuse_codex.state import SessionStateStore
from langfuse_codex.tracer import LangfuseTracer
from langfuse_codex.transcript import TranscriptParser


def _integration_enabled() -> bool:
    load_dotenv(Path(".env"), override=False)
    return (
        os.getenv("LANGFUSE_RUN_INTEGRATION") == "1"
        and bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
        and bool(os.getenv("LANGFUSE_SECRET_KEY"))
        and bool(os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST"))
    )


pytestmark = pytest.mark.skipif(
    not _integration_enabled(),
    reason="Set LANGFUSE_RUN_INTEGRATION=1 with Langfuse credentials to run the live integration test.",
)


def test_hook_processor_writes_trace_to_langfuse(tmp_path: Path) -> None:
    transcript_path = tmp_path / "transcript.jsonl"
    session_id = f"langfuse-codex-session-{uuid.uuid4().hex[:10]}"
    turn_id = f"langfuse-codex-turn-{uuid.uuid4().hex[:10]}"

    transcript_entries = [
        {
            "type": "session_meta",
            "payload": {
                "id": session_id,
                "cwd": str(Path.cwd()),
                "originator": "Codex Desktop",
                "cli_version": "0.119.0-alpha.28",
                "source": "desktop",
                "model_provider": "openai",
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "task_started",
                "turn_id": turn_id,
                "started_at": 100,
                "model_context_window": 258400,
                "collaboration_mode_kind": "default",
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "call_id": "call-live-1",
                "name": "exec_command",
                "arguments": json.dumps({"cmd": "sed -n '1,5p' README.md"}),
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "exec_command_end",
                "call_id": "call-live-1",
                "process_id": "proc-live-1",
                "turn_id": turn_id,
                "command": ["/bin/zsh", "-lc", "sed -n '1,5p' README.md"],
                "cwd": str(Path.cwd()),
                "parsed_cmd": [{"type": "read", "path": str(Path.cwd() / "README.md")}],
                "aggregated_output": "trace smoke output",
                "stdout": "",
                "stderr": "",
                "exit_code": 0,
                "status": "completed",
                "duration": {"secs": 0, "nanos": 1},
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "call-live-1",
                "output": "tool finished",
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "total_token_usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                    "last_token_usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                    "model_context_window": 258400,
                },
                "rate_limits": {"primary": {"used_percent": 2.0}},
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "task_complete",
                "turn_id": turn_id,
                "completed_at": 120,
                "duration_ms": 2000,
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "turn_id": turn_id,
                "content": [{"type": "output_text", "text": "Langfuse integration finished"}],
            },
        },
    ]
    transcript_path.write_text("\n".join(json.dumps(entry) for entry in transcript_entries) + "\n", encoding="utf-8")

    config = HookConfig.from_environment(Path.cwd())
    tracer = LangfuseTracer(config)
    processor = HookProcessor(
        config=config,
        state_store=SessionStateStore(tmp_path / "state"),
        tracer=tracer,
        parser=TranscriptParser(),
    )

    common_payload = {
        "session_id": session_id,
        "turn_id": turn_id,
        "cwd": str(Path.cwd()),
        "transcript_path": str(transcript_path),
        "model": "gpt-5.4",
    }
    processor.handle({**common_payload, "hook_event_name": "UserPromptSubmit", "prompt": "Run live trace"})
    processor.handle(
        {
            **common_payload,
            "hook_event_name": "PostToolUse",
            "tool_name": "Bash",
            "tool_use_id": "tool-live-1",
        }
    )
    processor.handle({**common_payload, "hook_event_name": "Stop"})
    tracer.flush()
    trace_id = tracer.client.create_trace_id(seed=f"{session_id}:{turn_id}")
    tracer.shutdown()

    api_client = Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        base_url=os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST"),
    )

    try:
        for _ in range(30):
            try:
                trace = api_client.api.trace.get(trace_id, fields="observations,io")
            except Exception:
                time.sleep(1)
                continue
            observation_names = {observation.name for observation in trace.observations}
            if (
                "terminal.exec" in observation_names
                and "assistant-response" in observation_names
                and trace.output
            ):
                break
            time.sleep(1)
        else:
            pytest.fail("Trace never became available in Langfuse.")

        exec_observations = [observation for observation in trace.observations if observation.name == "terminal.exec"]
        assistant_observations = [observation for observation in trace.observations if observation.name == "assistant-response"]
        assert trace.id == trace_id
        assert trace.name == "codex-turn: Run live trace"
        assert trace.session_id == session_id
        assert trace.input
        assert trace.output
        assert trace.metadata["turn_id"] == turn_id
        assert trace.metadata["token_summary"]["total_token_usage"]["total_tokens"] == 15
        assert trace.metadata["task_duration_ms"] == 2000
        assert len(exec_observations) == 1
        assert assistant_observations
        exec_metadata = exec_observations[0].metadata
        assert exec_metadata["read_paths"] == [str(Path.cwd() / "README.md")]
    finally:
        api_client.api.trace.delete(trace_id)
        api_client.shutdown()
