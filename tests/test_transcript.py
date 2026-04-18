from langfuse_codex.state import SessionState
from langfuse_codex.transcript import TranscriptParser


def test_parse_exec_command_into_one_logical_observation() -> None:
    parser = TranscriptParser()
    state = SessionState(session_id="session-1")
    entries = [
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "call_id": "call-1",
                "name": "exec_command",
                "arguments": "{\"cmd\": \"sed -n '1,5p' README.md\", \"max_output_tokens\": 200}",
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "exec_command_end",
                "call_id": "call-1",
                "process_id": "123",
                "turn_id": "turn-1",
                "command": ["/bin/zsh", "-lc", "sed -n '1,5p' README.md"],
                "cwd": "/repo",
                "parsed_cmd": [{"type": "read", "path": "/repo/README.md"}],
                "aggregated_output": "hello",
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
                "call_id": "call-1",
                "output": "Chunk output",
            },
        },
    ]

    parsed = parser.parse_entries(entries, state=state, fallback_turn_id="turn-1", cwd="/repo")

    assert len(parsed.observations) == 1
    observation = parsed.observations[0]
    assert observation.tool_name == "terminal.exec"
    assert observation.metadata["call_id"] == "call-1"
    assert observation.read_paths == ["/repo/README.md"]
    assert observation.exit_code == 0
    assert set(observation.codex_event_types) == {"function_call", "exec_command_end"}


def test_parse_patch_apply_and_task_metadata() -> None:
    parser = TranscriptParser()
    state = SessionState(session_id="session-1")
    entries = [
        {
            "type": "event_msg",
            "payload": {
                "type": "task_started",
                "turn_id": "turn-1",
                "started_at": 100,
                "model_context_window": 258400,
                "collaboration_mode_kind": "default",
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "custom_tool_call",
                "call_id": "call-patch-1",
                "turn_id": "turn-1",
                "status": "completed",
                "name": "apply_patch",
                "input": "*** Begin Patch\n*** Update File: /repo/src/langfuse_codex/hook.py\n*** End Patch\n",
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "patch_apply_end",
                "call_id": "call-patch-1",
                "turn_id": "turn-1",
                "stdout": "Success",
                "stderr": "",
                "success": True,
                "changes": {
                    "/repo/src/langfuse_codex/hook.py": {
                        "type": "update",
                        "unified_diff": "@@ -1 +1 @@\n-old\n+new\n",
                        "move_path": None,
                    }
                },
                "status": "completed",
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
                "turn_id": "turn-1",
                "completed_at": 120,
                "duration_ms": 2000,
            },
        },
    ]

    parsed = parser.parse_entries(entries, state=state, fallback_turn_id="turn-1", cwd="/repo")

    assert [observation.tool_name for observation in parsed.observations] == [
        "task.started",
        "codex.patch_apply",
        "task.completed",
    ]
    patch_observation = parsed.observations[1]
    assert patch_observation.write_paths == ["/repo/src/langfuse_codex/hook.py"]
    assert patch_observation.metadata["change_summary"]["modified"] == 1
    assert parsed.turn_updates["turn-1"].trace_metadata["token_summary"]["total_token_usage"]["total_tokens"] == 15
    assert parsed.turn_updates["turn-1"].trace_metadata["task_duration_ms"] == 2000


def test_parse_assistant_message() -> None:
    parser = TranscriptParser()
    state = SessionState(session_id="session-1")
    entries = [
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "turn_id": "turn-1",
                "content": [{"type": "output_text", "text": "Done"}],
            },
        }
    ]

    parsed = parser.parse_entries(entries, state=state, fallback_turn_id="turn-1", cwd="/repo")

    assert parsed.turn_updates["turn-1"].assistant_message == "Done"


def test_parse_session_metadata_from_current_transcript_events() -> None:
    parser = TranscriptParser()
    state = SessionState(session_id="session-1")
    entries = [
        {
            "type": "event_msg",
            "payload": {
                "type": "user_message",
                "message": "Inspect this",
                "images": ["img-1"],
                "local_images": [{"path": "/tmp/local.png"}],
                "text_elements": [{"text": "selected"}],
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "thread_name_updated",
                "thread_id": "session-1",
                "thread_name": "Inspect hooks",
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "reasoning",
                "summary": [{"text": "Checked docs"}],
                "encrypted_content": "opaque",
            },
        },
    ]

    parsed = parser.parse_entries(entries, state=state, fallback_turn_id="turn-1", cwd="/repo")
    metadata = parsed.turn_updates["turn-1"].trace_metadata

    assert state.session_metadata["thread_name"] == "Inspect hooks"
    assert metadata["thread_name"] == "Inspect hooks"
    assert metadata["user_message"]["images_count"] == 1
    assert metadata["user_message"]["local_images_count"] == 1
    assert metadata["user_message"]["text_elements_count"] == 1
    assert metadata["reasoning_summary"] == [{"text": "Checked docs"}]
