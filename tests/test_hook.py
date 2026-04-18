import json
from pathlib import Path

from langfuse_codex.hook import HookProcessor
from langfuse_codex.state import SessionStateStore
from langfuse_codex.transcript import TranscriptParser


class FakeTracer:
    def __init__(self) -> None:
        self.turn_roots: list[tuple[str, str]] = []
        self.observations: list[tuple[str, str]] = []
        self.observation_records = []
        self.responses: list[tuple[str, str]] = []
        self.trace_names: list[str] = []
        self.trace_metadata: list[dict[str, object]] = []

    def ensure_turn_root(self, state, trace_context, *, prompt=None):
        turn_state = state.turns.get(trace_context.turn_id)
        if not turn_state:
            from langfuse_codex.state import TurnState

            turn_state = TurnState(trace_id=f"trace-{trace_context.turn_id}", root_observation_id="root")
            state.turns[trace_context.turn_id] = turn_state
        if prompt:
            turn_state.prompt = prompt
        turn_state.trace_name = trace_context.trace_name
        turn_state.trace_metadata.update(trace_context.trace_metadata)
        self.turn_roots.append((trace_context.session_id, trace_context.turn_id))
        self.trace_names.append(trace_context.trace_name)
        self.trace_metadata.append(dict(trace_context.trace_metadata))
        return turn_state

    def record_observation(self, turn_state, trace_context, observation):
        self.observations.append((trace_context.turn_id, observation.name))
        self.observation_records.append(observation)

    def record_assistant_response(self, turn_state, trace_context, message):
        self.responses.append((trace_context.turn_id, message))

    def flush(self) -> None:
        return None


class FakeConfig:
    def __init__(self, project_root: Path, state_dir: Path):
        self.project_root = project_root
        self.state_dir = state_dir


def test_hook_processor_emits_observations_and_final_response(tmp_path: Path) -> None:
    transcript_path = tmp_path / "transcript.jsonl"
    transcript_entries = [
        {
            "type": "event_msg",
            "payload": {
                "type": "exec_command_end",
                "call_id": "call-1",
                "process_id": "proc-1",
                "turn_id": "turn-1",
                "command": ["/bin/zsh", "-lc", "ls src"],
                "cwd": str(tmp_path),
                "parsed_cmd": [{"type": "list_files", "path": str(tmp_path / "src")}],
                "aggregated_output": "hook.py",
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
                "type": "message",
                "role": "assistant",
                "turn_id": "turn-1",
                "content": [{"type": "output_text", "text": "Finished"}],
            },
        },
    ]
    transcript_path.write_text("\n".join(json.dumps(entry) for entry in transcript_entries) + "\n", encoding="utf-8")

    tracer = FakeTracer()
    processor = HookProcessor(
        config=FakeConfig(tmp_path, tmp_path / "state"),
        state_store=SessionStateStore(tmp_path / "state"),
        tracer=tracer,
        parser=TranscriptParser(),
    )

    session_payload = {
        "session_id": "session-1",
        "hook_event_name": "UserPromptSubmit",
        "turn_id": "turn-1",
        "cwd": str(tmp_path),
        "prompt": "Trace this turn",
        "transcript_path": str(transcript_path),
        "model": "gpt-5.4",
    }
    processor.handle(session_payload)

    post_tool_payload = {
        "session_id": "session-1",
        "hook_event_name": "PostToolUse",
        "turn_id": "turn-1",
        "cwd": str(tmp_path),
        "transcript_path": str(transcript_path),
        "model": "gpt-5.4",
        "tool_name": "Bash",
        "tool_use_id": "tool-1",
    }
    processor.handle(post_tool_payload)

    stop_payload = {
        "session_id": "session-1",
        "hook_event_name": "Stop",
        "turn_id": "turn-1",
        "cwd": str(tmp_path),
        "transcript_path": str(transcript_path),
        "model": "gpt-5.4",
        "stop_hook_active": False,
    }
    processor.handle(stop_payload)

    assert ("turn-1", "terminal.exec") in tracer.observations
    assert ("turn-1", "Finished") in tracer.responses
    assert tracer.trace_names[-1] == "codex-turn: Trace this turn"
    assert tracer.trace_metadata[-1]["stop_hook_active"] is False


def test_hook_processor_captures_pre_tool_use_and_permission_mode(tmp_path: Path) -> None:
    transcript_path = tmp_path / "transcript.jsonl"
    transcript_path.write_text("", encoding="utf-8")
    tracer = FakeTracer()
    processor = HookProcessor(
        config=FakeConfig(tmp_path, tmp_path / "state"),
        state_store=SessionStateStore(tmp_path / "state"),
        tracer=tracer,
        parser=TranscriptParser(),
    )

    common = {
        "session_id": "session-1",
        "turn_id": "turn-1",
        "cwd": str(tmp_path),
        "transcript_path": str(transcript_path),
        "model": "gpt-5.4",
        "permission_mode": "acceptEdits",
    }
    processor.handle({**common, "hook_event_name": "UserPromptSubmit", "prompt": "Trace pre hooks"})
    processor.handle(
        {
            **common,
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "tool_use_id": "tool-1",
            "tool_input": {"command": "printf hello"},
        }
    )
    processor.handle(
        {
            **common,
            "hook_event_name": "PostToolUse",
            "tool_name": "Bash",
            "tool_use_id": "tool-1",
            "tool_input": {"command": "printf hello"},
            "tool_response": "hello",
        }
    )

    assert tracer.trace_metadata[0]["permission_mode"] == "acceptEdits"
    observation = tracer.observation_records[-1]
    assert observation.tool_name == "terminal.exec"
    assert observation.input_data == {"command": "printf hello"}
    assert observation.output_data == "hello"
    assert observation.metadata["tool_use_id"] == "tool-1"
    assert observation.metadata["permission_mode"] == "acceptEdits"
    assert set(observation.codex_event_types) == {"pre_tool_use", "post_tool_use"}
