from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langfuse_codex.redaction import extract_file_paths
from langfuse_codex.state import SessionState

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ObservationRecord:
    stable_key: str
    name: str
    kind: str
    turn_id: str
    input_data: Any = None
    output_data: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedDelta:
    observations: list[ObservationRecord] = field(default_factory=list)
    assistant_messages: dict[str, str] = field(default_factory=dict)


def read_transcript_delta(transcript_path: str | None, offset: int) -> tuple[list[dict[str, Any]], int]:
    if not transcript_path:
        return [], offset

    path = Path(transcript_path)
    if not path.exists():
        return [], offset

    file_size = path.stat().st_size
    if file_size < offset:
        offset = 0

    with path.open("rb") as handle:
        handle.seek(offset)
        chunk = handle.read()

    if not chunk:
        return [], offset

    last_newline = chunk.rfind(b"\n")
    if last_newline == -1:
        return [], offset

    complete = chunk[: last_newline + 1]
    new_offset = offset + last_newline + 1
    entries: list[dict[str, Any]] = []
    for raw_line in complete.splitlines():
        if not raw_line.strip():
            continue
        try:
            entries.append(json.loads(raw_line))
        except json.JSONDecodeError:
            LOGGER.warning("Skipping malformed transcript line", extra={"line": raw_line[:200]})
    return entries, new_offset


def hash_entry(entry: dict[str, Any]) -> str:
    payload = json.dumps(entry, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class TranscriptParser:
    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or LOGGER

    def parse_entries(
        self,
        entries: list[dict[str, Any]],
        *,
        state: SessionState,
        fallback_turn_id: str | None,
        cwd: str,
    ) -> ParsedDelta:
        parsed = ParsedDelta()

        for entry in entries:
            event_hash = hash_entry(entry)
            if event_hash in state.recent_event_hashes:
                continue
            state.remember_event_hash(event_hash)

            payload = entry.get("payload", {})
            payload_type = payload.get("type")
            turn_id = payload.get("turn_id") or fallback_turn_id or state.active_turn_id

            if entry.get("type") == "response_item":
                if payload_type == "function_call":
                    self._remember_function_call(state, payload, turn_id)
                    continue

                if payload_type == "function_call_output":
                    record = self._build_function_output_record(state, payload, turn_id, cwd)
                    if record:
                        parsed.observations.append(record)
                    continue

                if payload_type == "message" and payload.get("role") == "assistant" and turn_id:
                    text = extract_message_text(payload)
                    if text:
                        parsed.assistant_messages[turn_id] = text
                        continue

                if payload_type == "web_search_call":
                    continue

            if entry.get("type") != "event_msg":
                continue

            if payload_type == "exec_command_end" and turn_id:
                parsed.observations.append(self._build_exec_command_record(payload, turn_id, cwd))
                continue

            if payload_type == "web_search_end" and turn_id:
                parsed.observations.append(self._build_web_search_record(payload, turn_id, cwd))
                continue

            if payload_type and payload_type.endswith("_begin") and turn_id:
                self._remember_begin_event(state, payload_type, payload, turn_id)
                continue

            if payload_type and payload_type.endswith("_end") and turn_id:
                record = self._build_end_event_record(state, payload_type, payload, turn_id, cwd)
                if record:
                    parsed.observations.append(record)
                continue

            if payload_type in {"view_image_tool_call", "request_user_input"} and turn_id:
                parsed.observations.append(
                    ObservationRecord(
                        stable_key=f"{payload_type}:{payload.get('call_id') or entry.get('timestamp')}",
                        name=payload_type,
                        kind="tool",
                        turn_id=turn_id,
                        input_data=payload,
                        metadata={"payload_type": payload_type},
                    )
                )
                continue

            if payload_type and payload_type not in {
                "token_count",
                "turn_context",
                "rate_limits",
            }:
                self.logger.debug("Ignoring unsupported transcript event", extra={"payload_type": payload_type})

        return parsed

    def flush_pending_calls(self, state: SessionState, turn_id: str) -> list[ObservationRecord]:
        records: list[ObservationRecord] = []
        for call_id, pending in list(state.pending_tool_calls.items()):
            if pending.get("turn_id") != turn_id:
                continue
            records.append(
                ObservationRecord(
                    stable_key=f"incomplete:{call_id}",
                    name=str(pending.get("name") or "tool-call"),
                    kind="tool",
                    turn_id=turn_id,
                    input_data=pending.get("input"),
                    metadata={"status": "pending", "call_id": call_id},
                )
            )
            state.pending_tool_calls.pop(call_id, None)
        return records

    def _remember_function_call(
        self,
        state: SessionState,
        payload: dict[str, Any],
        turn_id: str | None,
    ) -> None:
        call_id = payload.get("call_id")
        if not call_id or not turn_id:
            return
        state.pending_tool_calls[call_id] = {
            "turn_id": turn_id,
            "name": payload.get("name"),
            "input": _maybe_parse_json(payload.get("arguments")),
            "raw_arguments": payload.get("arguments"),
        }

    def _build_function_output_record(
        self,
        state: SessionState,
        payload: dict[str, Any],
        turn_id: str | None,
        cwd: str,
    ) -> ObservationRecord | None:
        call_id = payload.get("call_id")
        pending = state.pending_tool_calls.pop(call_id, {}) if call_id else {}
        record_turn_id = pending.get("turn_id") or turn_id
        if not record_turn_id:
            return None

        input_data = pending.get("input")
        output_data = _maybe_parse_json(payload.get("output"))
        metadata = {"call_id": call_id, "payload_type": "function_call_output"}
        used_file_paths = extract_file_paths(input_data, output_data, cwd=cwd)
        if used_file_paths:
            metadata["used_file_paths"] = used_file_paths

        return ObservationRecord(
            stable_key=f"function-call:{call_id}",
            name=str(pending.get("name") or "function_call"),
            kind="tool",
            turn_id=record_turn_id,
            input_data=input_data,
            output_data=output_data,
            metadata=metadata,
        )

    def _build_exec_command_record(
        self,
        payload: dict[str, Any],
        turn_id: str,
        cwd: str,
    ) -> ObservationRecord:
        input_data = {
            "command": payload.get("command"),
            "cwd": payload.get("cwd"),
            "parsed_cmd": payload.get("parsed_cmd"),
        }
        output_data = {
            "aggregated_output": payload.get("aggregated_output"),
            "stdout": payload.get("stdout"),
            "stderr": payload.get("stderr"),
            "exit_code": payload.get("exit_code"),
            "status": payload.get("status"),
        }
        metadata = {
            "call_id": payload.get("call_id"),
            "process_id": payload.get("process_id"),
            "duration": payload.get("duration"),
            "source": payload.get("source"),
        }
        used_file_paths = extract_file_paths(input_data, output_data, cwd=cwd)
        if used_file_paths:
            metadata["used_file_paths"] = used_file_paths

        return ObservationRecord(
            stable_key=f"exec-command:{payload.get('call_id') or payload.get('process_id')}",
            name="exec_command",
            kind="tool",
            turn_id=turn_id,
            input_data=input_data,
            output_data=output_data,
            metadata=metadata,
        )

    def _build_web_search_record(
        self,
        payload: dict[str, Any],
        turn_id: str,
        cwd: str,
    ) -> ObservationRecord:
        action = payload.get("action") or {}
        input_data = {"query": payload.get("query"), "action": action}
        metadata = {"call_id": payload.get("call_id"), "payload_type": "web_search_end"}
        used_file_paths = extract_file_paths(action, cwd=cwd)
        if used_file_paths:
            metadata["used_file_paths"] = used_file_paths

        return ObservationRecord(
            stable_key=f"web-search:{payload.get('call_id') or payload.get('query')}",
            name=str(action.get("type") or "web_search"),
            kind="tool",
            turn_id=turn_id,
            input_data=input_data,
            metadata=metadata,
        )

    def _remember_begin_event(
        self,
        state: SessionState,
        payload_type: str,
        payload: dict[str, Any],
        turn_id: str,
    ) -> None:
        key = _pending_key(payload_type, payload)
        if not key:
            return
        state.pending_events[key] = {"turn_id": turn_id, "payload_type": payload_type, "payload": payload}

    def _build_end_event_record(
        self,
        state: SessionState,
        payload_type: str,
        payload: dict[str, Any],
        turn_id: str,
        cwd: str,
    ) -> ObservationRecord | None:
        key = _pending_key(payload_type, payload)
        begin_event = state.pending_events.pop(key, None) if key else None
        name = _tool_name_from_payload(payload, default=payload_type.removesuffix("_end"))
        input_data = begin_event["payload"] if begin_event else _extract_input_payload(payload)
        output_data = _extract_output_payload(payload)
        metadata = {
            "payload_type": payload_type,
            "call_id": payload.get("call_id"),
            "tool_name": payload.get("tool_name"),
        }
        if begin_event:
            metadata["begin_payload_type"] = begin_event.get("payload_type")

        used_file_paths = extract_file_paths(input_data, output_data, payload, cwd=cwd)
        if used_file_paths:
            metadata["used_file_paths"] = used_file_paths

        return ObservationRecord(
            stable_key=f"{payload_type}:{payload.get('call_id') or payload.get('id') or payload.get('process_id') or turn_id}",
            name=name,
            kind="tool",
            turn_id=begin_event["turn_id"] if begin_event else turn_id,
            input_data=input_data,
            output_data=output_data,
            metadata=metadata,
        )


def extract_message_text(payload: dict[str, Any]) -> str:
    parts = []
    for item in payload.get("content", []):
        item_type = item.get("type")
        if item_type in {"output_text", "input_text"}:
            text = item.get("text")
            if text:
                parts.append(text)
        elif item_type == "text":
            text = item.get("text") or item.get("content")
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def _maybe_parse_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    if stripped[0] not in {"{", "[", "\""}:
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _pending_key(payload_type: str, payload: dict[str, Any]) -> str | None:
    base = payload_type.removesuffix("_begin").removesuffix("_end")
    identifier = payload.get("call_id") or payload.get("tool_use_id") or payload.get("id") or payload.get("process_id")
    if not identifier:
        return None
    return f"{base}:{identifier}"


def _tool_name_from_payload(payload: dict[str, Any], default: str) -> str:
    if payload.get("tool_name"):
        return str(payload["tool_name"])
    if payload.get("name"):
        return str(payload["name"])
    action = payload.get("action")
    if isinstance(action, dict) and action.get("type"):
        return str(action["type"])
    return default


def _extract_input_payload(payload: dict[str, Any]) -> Any:
    for key in ("tool_input", "input", "action", "query", "command", "arguments"):
        if key in payload:
            return payload[key]
    return payload


def _extract_output_payload(payload: dict[str, Any]) -> Any:
    for key in ("tool_response", "output", "result", "response", "aggregated_output", "stdout", "stderr"):
        if key in payload:
            return payload[key]
    return None

