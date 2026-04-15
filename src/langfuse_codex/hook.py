from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from langfuse_codex.config import HookConfig
from langfuse_codex.state import SessionStateStore
from langfuse_codex.tracer import LangfuseTracer, TraceContext
from langfuse_codex.transcript import ObservationRecord, TranscriptParser, read_transcript_delta

LOGGER = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        payload = _load_payload(args.stdin_file)
        cwd = payload.get("cwd") or str(Path.cwd())
        config = HookConfig.from_environment(cwd)
        _configure_logging(config)

        processor = HookProcessor(
            config=config,
            state_store=SessionStateStore(config.state_dir),
            tracer=LangfuseTracer(config, logger=LOGGER),
            parser=TranscriptParser(LOGGER),
        )
        processor.handle(payload)
    except Exception:
        LOGGER.exception("Langfuse Codex hook failed open")
    return 0


class HookProcessor:
    def __init__(
        self,
        *,
        config: HookConfig,
        state_store: SessionStateStore,
        tracer: LangfuseTracer,
        parser: TranscriptParser,
    ) -> None:
        self.config = config
        self.state_store = state_store
        self.tracer = tracer
        self.parser = parser

    def handle(self, payload: dict[str, Any]) -> None:
        session_id = payload["session_id"]
        hook_event_name = payload["hook_event_name"]
        cwd = payload.get("cwd") or str(self.config.project_root)

        with self.state_store.open(session_id) as state:
            state.transcript_path = payload.get("transcript_path") or state.transcript_path
            state.session_metadata.update(
                {key: str(value) for key, value in {"cwd": cwd, "model": payload.get("model")}.items() if value}
            )

            if hook_event_name == "SessionStart":
                self._handle_session_start(state, payload)
                return

            if hook_event_name == "UserPromptSubmit":
                self._handle_user_prompt_submit(state, payload)
                return

            if hook_event_name == "PostToolUse":
                self._handle_post_tool_use(state, payload)
                return

            if hook_event_name == "Stop":
                self._handle_stop(state, payload)
                return

            LOGGER.debug("Ignoring unsupported hook event", extra={"hook_event_name": hook_event_name})

    def _handle_session_start(self, state, payload: dict[str, Any]) -> None:
        state.active_turn_id = None
        if payload.get("transcript_path"):
            state.transcript_path = payload["transcript_path"]

    def _handle_user_prompt_submit(self, state, payload: dict[str, Any]) -> None:
        turn_id = payload["turn_id"]
        state.active_turn_id = turn_id
        trace_context = self._build_trace_context(state, payload, turn_id)
        self.tracer.ensure_turn_root(state, trace_context, prompt=payload.get("prompt"))
        self.tracer.flush()

    def _handle_post_tool_use(self, state, payload: dict[str, Any]) -> None:
        turn_id = payload.get("turn_id") or state.active_turn_id
        if turn_id:
            state.active_turn_id = turn_id

        grouped = self._process_transcript_delta(state, payload, fallback_turn_id=turn_id)
        if not grouped and turn_id:
            fallback = self._fallback_post_tool_observation(payload, turn_id)
            if fallback:
                grouped = {turn_id: [fallback]}

        self._emit_grouped_observations(state, payload, grouped)
        self.tracer.flush()

    def _handle_stop(self, state, payload: dict[str, Any]) -> None:
        turn_id = payload.get("turn_id") or state.active_turn_id
        grouped = self._process_transcript_delta(state, payload, fallback_turn_id=turn_id)
        if turn_id:
            pending = self.parser.flush_pending_calls(state, turn_id)
            if pending:
                grouped.setdefault(turn_id, []).extend(pending)

        self._emit_grouped_observations(state, payload, grouped)

        if turn_id:
            trace_context = self._build_trace_context(state, payload, turn_id)
            turn_state = self.tracer.ensure_turn_root(state, trace_context)
            assistant_message = payload.get("last_assistant_message") or turn_state.last_assistant_message
            if assistant_message:
                self.tracer.record_assistant_response(turn_state, trace_context, assistant_message)
                turn_state.last_assistant_message = assistant_message
            turn_state.closed = True
            state.active_turn_id = None
            state.clear_pending_for_turn(turn_id)

        self.tracer.flush()

    def _process_transcript_delta(
        self,
        state,
        payload: dict[str, Any],
        *,
        fallback_turn_id: str | None,
    ) -> dict[str, list[ObservationRecord]]:
        entries, new_offset = read_transcript_delta(state.transcript_path, state.transcript_offset)
        state.transcript_offset = new_offset
        if not entries:
            return {}

        parsed = self.parser.parse_entries(
            entries,
            state=state,
            fallback_turn_id=fallback_turn_id,
            cwd=payload.get("cwd") or str(self.config.project_root),
        )
        for turn_id, message in parsed.assistant_messages.items():
            turn_state = state.turns.get(turn_id)
            if turn_state:
                turn_state.last_assistant_message = message

        grouped: dict[str, list[ObservationRecord]] = {}
        for observation in parsed.observations:
            grouped.setdefault(observation.turn_id, []).append(observation)
        return grouped

    def _emit_grouped_observations(
        self,
        state,
        payload: dict[str, Any],
        grouped: dict[str, list[ObservationRecord]],
    ) -> None:
        for turn_id, observations in grouped.items():
            trace_context = self._build_trace_context(state, payload, turn_id)
            turn_state = self.tracer.ensure_turn_root(state, trace_context)
            for observation in observations:
                self.tracer.record_observation(turn_state, trace_context, observation)

    def _build_trace_context(self, state, payload: dict[str, Any], turn_id: str) -> TraceContext:
        return TraceContext(
            session_id=state.session_id,
            turn_id=turn_id,
            cwd=payload.get("cwd") or state.session_metadata.get("cwd") or str(self.config.project_root),
            model=payload.get("model") or state.session_metadata.get("model"),
            transcript_path=state.transcript_path,
        )

    def _fallback_post_tool_observation(
        self,
        payload: dict[str, Any],
        turn_id: str,
    ) -> ObservationRecord | None:
        tool_input = payload.get("tool_input")
        tool_response = payload.get("tool_response")
        if tool_input is None and tool_response is None:
            return None

        return ObservationRecord(
            stable_key=f"post-tool-use:{payload.get('tool_use_id') or turn_id}",
            name=str(payload.get("tool_name") or "Bash"),
            kind="tool",
            turn_id=turn_id,
            input_data=tool_input,
            output_data=tool_response,
            metadata={"source": "hook_payload_fallback"},
        )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Codex hook ingestor for Langfuse")
    parser.add_argument("--stdin-file", type=Path, help="Read hook JSON from a file instead of stdin")
    return parser.parse_args(argv)


def _load_payload(stdin_file: Path | None) -> dict[str, Any]:
    raw = stdin_file.read_text(encoding="utf-8") if stdin_file else sys.stdin.read()
    if not raw.strip():
        raise ValueError("Hook payload is empty")
    return json.loads(raw)


def _configure_logging(config: HookConfig) -> None:
    config.state_dir.mkdir(parents=True, exist_ok=True)
    log_path = config.state_dir / "hook.log"
    root = logging.getLogger()
    if root.handlers:
        return

    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stderr),
        ],
    )

