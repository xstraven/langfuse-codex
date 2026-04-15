from langfuse_codex.state import SessionState
from langfuse_codex.transcript import TranscriptParser


def test_parse_exec_command_and_function_call_output() -> None:
    parser = TranscriptParser()
    state = SessionState(session_id="session-1")
    entries = [
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "call_id": "call-1",
                "name": "exec_command",
                "arguments": "{\"cmd\": \"sed -n '1,5p' README.md\"}",
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

    assert len(parsed.observations) == 2
    assert parsed.observations[0].name == "exec_command"
    assert parsed.observations[0].metadata["used_file_paths"] == ["/repo/README.md"]
    assert parsed.observations[1].name == "exec_command"
    assert parsed.observations[1].metadata["call_id"] == "call-1"


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

    assert parsed.assistant_messages["turn-1"] == "Done"

