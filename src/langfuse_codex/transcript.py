from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langfuse_codex.redaction import extract_path_buckets
from langfuse_codex.state import SessionState

LOGGER = logging.getLogger(__name__)

TRACE_SESSION_KEYS = ("cwd", "cli_version", "originator", "source", "model_provider")


@dataclass(slots=True)
class ObservationRecord:
    logical_id: str
    turn_id: str
    kind: str
    tool_family: str | None
    tool_name: str
    status: str | None = None
    input_data: Any = None
    output_data: Any = None
    duration_ms: int | None = None
    exit_code: int | None = None
    codex_event_types: list[str] = field(default_factory=list)
    read_paths: list[str] = field(default_factory=list)
    write_paths: list[str] = field(default_factory=list)
    search_paths: list[str] = field(default_factory=list)
    referenced_paths: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    debug_payloads: dict[str, Any] = field(default_factory=dict)
    pty_session_id: str | int | None = None

    @property
    def name(self) -> str:
        return self.tool_name


@dataclass(slots=True)
class TurnUpdate:
    assistant_message: str | None = None
    trace_metadata: dict[str, Any] = field(default_factory=dict)

    def merge(self, other: "TurnUpdate") -> None:
        if other.assistant_message:
            self.assistant_message = other.assistant_message
        self.trace_metadata.update(other.trace_metadata)


@dataclass(slots=True)
class ParsedDelta:
    observations: list[ObservationRecord] = field(default_factory=list)
    turn_updates: dict[str, TurnUpdate] = field(default_factory=dict)

    def update_turn(self, turn_id: str) -> TurnUpdate:
        if turn_id not in self.turn_updates:
            self.turn_updates[turn_id] = TurnUpdate()
        return self.turn_updates[turn_id]


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
    def __init__(self, logger: logging.Logger | None = None, *, capture_agent_messages: bool = False):
        self.logger = logger or LOGGER
        self.capture_agent_messages = capture_agent_messages

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

            entry_type = entry.get("type")
            payload = entry.get("payload", {})
            payload_type = payload.get("type")
            turn_id = payload.get("turn_id") or fallback_turn_id or state.active_turn_id

            if entry_type == "session_meta":
                self._remember_session_meta(state, payload)
                continue

            if entry_type == "turn_context" and turn_id:
                self._remember_turn_context(state, parsed, turn_id, payload)
                continue

            if entry_type == "response_item":
                if payload_type == "function_call":
                    self._remember_function_call(state, payload, turn_id, cwd)
                    continue

                if payload_type == "function_call_output":
                    record = self._handle_function_call_output(state, payload, turn_id, cwd)
                    if record:
                        parsed.observations.append(record)
                    continue

                if payload_type == "custom_tool_call":
                    self._remember_custom_tool_call(state, payload, turn_id, cwd)
                    continue

                if payload_type == "custom_tool_call_output":
                    record = self._handle_custom_tool_call_output(state, payload, turn_id, cwd)
                    if record:
                        parsed.observations.append(record)
                    continue

                if payload_type == "message" and payload.get("role") == "assistant" and turn_id:
                    text = extract_message_text(payload)
                    if text:
                        parsed.update_turn(turn_id).assistant_message = text
                    continue

                if payload_type == "reasoning" and turn_id:
                    self._remember_reasoning_summary(parsed, turn_id, payload)
                    continue

            if entry_type != "event_msg":
                continue

            if payload_type == "user_message" and turn_id:
                self._remember_user_message(parsed, turn_id, payload)
                continue

            if payload_type == "thread_name_updated":
                self._remember_thread_name(state, parsed, turn_id, payload)
                continue

            if payload_type == "task_started" and turn_id:
                parsed.update_turn(turn_id).trace_metadata.update(
                    {
                        "task_started_at": payload.get("started_at"),
                        "model_context_window": payload.get("model_context_window"),
                        "collaboration_mode": payload.get("collaboration_mode_kind"),
                    }
                )
                parsed.observations.append(self._build_task_event(payload, turn_id, "task.started"))
                continue

            if payload_type == "task_complete" and turn_id:
                parsed.update_turn(turn_id).trace_metadata.update(
                    {
                        "task_completed_at": payload.get("completed_at"),
                        "task_duration_ms": payload.get("duration_ms"),
                    }
                )
                parsed.observations.append(self._build_task_event(payload, turn_id, "task.completed"))
                continue

            if payload_type == "token_count":
                self._remember_token_count(parsed, turn_id, payload)
                continue

            if payload_type == "rate_limits":
                self._remember_rate_limits(parsed, turn_id, payload)
                continue

            if payload_type == "exec_command_end" and turn_id:
                parsed.observations.append(self._build_exec_command_record(state, payload, turn_id, cwd))
                continue

            if payload_type == "web_search_end" and turn_id:
                parsed.observations.append(self._build_web_search_record(payload, turn_id, cwd))
                continue

            if payload_type == "patch_apply_end" and turn_id:
                parsed.observations.append(self._build_patch_apply_record(state, payload, turn_id, cwd))
                continue

            if payload_type and payload_type.endswith("_begin") and turn_id:
                self._remember_begin_event(state, payload_type, payload, turn_id)
                continue

            if payload_type and payload_type.endswith("_end") and turn_id:
                record = self._build_end_event_record(state, payload_type, payload, turn_id, cwd)
                if record:
                    parsed.observations.append(record)
                continue

            if payload_type == "view_image_tool_call" and turn_id:
                parsed.observations.append(
                    self._build_single_event_observation(payload, turn_id, cwd, tool_family="codex", tool_name="codex.view_image")
                )
                continue

            if payload_type == "request_user_input" and turn_id:
                parsed.observations.append(
                    self._build_single_event_observation(
                        payload,
                        turn_id,
                        cwd,
                        tool_family="codex",
                        tool_name="codex.request_user_input",
                    )
                )
                continue

            if payload_type == "agent_message" and turn_id and self.capture_agent_messages:
                parsed.observations.append(
                    ObservationRecord(
                        logical_id=f"agent-message:{entry.get('timestamp')}",
                        turn_id=turn_id,
                        kind="event",
                        tool_family=None,
                        tool_name="agent.message",
                        status="completed",
                        output_data=payload.get("message"),
                        codex_event_types=["agent_message"],
                        debug_payloads={"agent_message": payload},
                    )
                )
                continue

            if payload_type and payload_type not in {"rate_limits", "turn_context"}:
                self.logger.debug("Ignoring unsupported transcript event", extra={"payload_type": payload_type})

        return parsed

    def flush_pending_calls(self, state: SessionState, turn_id: str, cwd: str) -> list[ObservationRecord]:
        records: list[ObservationRecord] = []
        for call_id, pending in list(state.pending_tool_calls.items()):
            if pending.get("turn_id") != turn_id:
                continue
            _ensure_pending_defaults(pending, call_id, turn_id=turn_id)
            records.append(self._finalize_pending_call(call_id, pending, cwd, force_pending=True))
            state.pending_tool_calls.pop(call_id, None)
        return records

    def remember_pre_tool_use(
        self,
        state: SessionState,
        payload: dict[str, Any],
        *,
        turn_id: str,
        cwd: str,
    ) -> None:
        call_id = str(payload.get("tool_use_id") or f"pre-tool-use:{turn_id}")
        raw_name = _raw_name_from_hook_tool(payload.get("tool_name"))
        pending = self._new_pending_call(
            turn_id=turn_id,
            call_id=call_id,
            raw_name=raw_name,
            input_data=payload.get("tool_input"),
            payload_type="pre_tool_use",
            payload=payload,
            cwd=cwd,
            status="pending",
        )
        pending["metadata"].update(
            {
                "source": "hook_payload",
                "tool_use_id": payload.get("tool_use_id"),
                "hook_tool_name": payload.get("tool_name"),
                "permission_mode": payload.get("permission_mode"),
            }
        )
        self._store_pending_call(state, call_id, pending)

    def pop_pending_hook_tool_use(
        self,
        state: SessionState,
        payload: dict[str, Any],
        *,
        turn_id: str,
        cwd: str,
    ) -> ObservationRecord | None:
        call_id = str(payload.get("tool_use_id") or "")
        pending = state.pending_tool_calls.pop(call_id, None) if call_id else None
        if not pending:
            return None

        _ensure_pending_defaults(
            pending,
            call_id,
            turn_id=turn_id,
            raw_name=_raw_name_from_hook_tool(payload.get("tool_name")),
        )
        pending["status"] = "completed"
        pending["output_data"] = payload.get("tool_response")
        pending["metadata"].update(
            {
                "source": "hook_payload_fallback",
                "tool_use_id": payload.get("tool_use_id"),
                "hook_tool_name": payload.get("tool_name"),
                "permission_mode": payload.get("permission_mode"),
            }
        )
        pending["codex_event_types"].append("post_tool_use")
        pending["debug_payloads"]["post_tool_use"] = payload
        self._merge_paths(
            pending,
            extract_path_buckets(payload.get("tool_input"), payload.get("tool_response"), cwd=cwd),
        )
        return self._finalize_pending_call(call_id, pending, cwd)

    def _remember_session_meta(self, state: SessionState, payload: dict[str, Any]) -> None:
        for key in TRACE_SESSION_KEYS:
            value = payload.get(key)
            if value is not None:
                state.session_metadata[key] = str(value)

    def _remember_turn_context(
        self,
        state: SessionState,
        parsed: ParsedDelta,
        turn_id: str,
        payload: dict[str, Any],
    ) -> None:
        collaboration_mode = payload.get("collaboration_mode")
        mode_value = None
        if isinstance(collaboration_mode, dict):
            mode_value = collaboration_mode.get("mode")
        elif payload.get("collaboration_mode_kind"):
            mode_value = payload.get("collaboration_mode_kind")

        trace_update = parsed.update_turn(turn_id).trace_metadata
        if mode_value:
            trace_update["collaboration_mode"] = mode_value
        if payload.get("cwd"):
            state.session_metadata["cwd"] = str(payload["cwd"])
        if payload.get("model"):
            state.session_metadata["model"] = str(payload["model"])

    def _remember_user_message(self, parsed: ParsedDelta, turn_id: str, payload: dict[str, Any]) -> None:
        update = parsed.update_turn(turn_id)
        update.trace_metadata["user_message"] = {
            "images_count": len(payload.get("images") or []),
            "local_images_count": len(payload.get("local_images") or []),
            "text_elements_count": len(payload.get("text_elements") or []),
        }
        if payload.get("images"):
            update.trace_metadata["user_message"]["images"] = payload.get("images")
        if payload.get("local_images"):
            update.trace_metadata["user_message"]["local_images"] = payload.get("local_images")
        if payload.get("text_elements"):
            update.trace_metadata["user_message"]["text_elements"] = payload.get("text_elements")

    def _remember_thread_name(
        self,
        state: SessionState,
        parsed: ParsedDelta,
        turn_id: str | None,
        payload: dict[str, Any],
    ) -> None:
        thread_name = payload.get("thread_name")
        if not thread_name:
            return
        state.session_metadata["thread_name"] = str(thread_name)
        if turn_id:
            parsed.update_turn(turn_id).trace_metadata["thread_name"] = str(thread_name)

    def _remember_reasoning_summary(self, parsed: ParsedDelta, turn_id: str, payload: dict[str, Any]) -> None:
        summary = payload.get("summary")
        if summary:
            parsed.update_turn(turn_id).trace_metadata["reasoning_summary"] = summary

    def _remember_rate_limits(self, parsed: ParsedDelta, turn_id: str | None, payload: dict[str, Any]) -> None:
        if turn_id:
            parsed.update_turn(turn_id).trace_metadata["rate_limits"] = payload.get("rate_limits") or payload

    def _remember_function_call(
        self,
        state: SessionState,
        payload: dict[str, Any],
        turn_id: str | None,
        cwd: str,
    ) -> None:
        call_id = payload.get("call_id")
        if not call_id or not turn_id:
            return

        input_data = _maybe_parse_json(payload.get("arguments"))
        pending = self._new_pending_call(
            turn_id=turn_id,
            call_id=call_id,
            raw_name=str(payload.get("name") or "function_call"),
            input_data=input_data,
            payload_type="function_call",
            payload=payload,
            cwd=cwd,
        )
        self._store_pending_call(state, call_id, pending)

    def _remember_custom_tool_call(
        self,
        state: SessionState,
        payload: dict[str, Any],
        turn_id: str | None,
        cwd: str,
    ) -> None:
        call_id = payload.get("call_id")
        if not call_id or not turn_id:
            return

        input_data = payload.get("input")
        pending = self._new_pending_call(
            turn_id=turn_id,
            call_id=call_id,
            raw_name=str(payload.get("name") or "custom_tool_call"),
            input_data=input_data,
            payload_type="custom_tool_call",
            payload=payload,
            cwd=cwd,
            status=payload.get("status"),
        )
        self._store_pending_call(state, call_id, pending)

    def _handle_function_call_output(
        self,
        state: SessionState,
        payload: dict[str, Any],
        turn_id: str | None,
        cwd: str,
    ) -> ObservationRecord | None:
        call_id = payload.get("call_id")
        pending = state.pending_tool_calls.get(call_id) if call_id else None
        if not pending:
            return None
        _ensure_pending_defaults(pending, str(call_id), turn_id=turn_id, raw_name="function_call")

        pending["output_data"] = _maybe_parse_json(payload.get("output"))
        pending["codex_event_types"].append("function_call_output")
        pending["debug_payloads"]["function_call_output"] = payload

        if pending["raw_name"] == "exec_command":
            return None

        state.pending_tool_calls.pop(call_id, None)
        return self._finalize_pending_call(call_id, pending, cwd)

    def _handle_custom_tool_call_output(
        self,
        state: SessionState,
        payload: dict[str, Any],
        turn_id: str | None,
        cwd: str,
    ) -> ObservationRecord | None:
        call_id = payload.get("call_id")
        pending = state.pending_tool_calls.get(call_id) if call_id else None
        if not pending:
            return None
        _ensure_pending_defaults(pending, str(call_id), turn_id=turn_id, raw_name="custom_tool_call")

        pending["output_data"] = _maybe_parse_json(payload.get("output"))
        pending["codex_event_types"].append("custom_tool_call_output")
        pending["debug_payloads"]["custom_tool_call_output"] = payload

        if pending["raw_name"] == "apply_patch":
            return None

        state.pending_tool_calls.pop(call_id, None)
        return self._finalize_pending_call(call_id, pending, cwd)

    def _build_exec_command_record(
        self,
        state: SessionState,
        payload: dict[str, Any],
        turn_id: str,
        cwd: str,
    ) -> ObservationRecord:
        call_id = str(payload.get("call_id") or payload.get("process_id") or f"exec:{turn_id}")
        pending = state.pending_tool_calls.pop(call_id, None)
        if not pending:
            pending = self._new_pending_call(
                turn_id=turn_id,
                call_id=call_id,
                raw_name="exec_command",
                input_data=None,
                payload_type="exec_command_end",
                payload=payload,
                cwd=cwd,
            )
        else:
            _ensure_pending_defaults(pending, call_id, turn_id=turn_id, raw_name="exec_command")
            pending["codex_event_types"].append("exec_command_end")
            pending["debug_payloads"]["exec_command_end"] = payload

        pending["tool_family"] = "terminal"
        pending["tool_name"] = "terminal.exec"
        pending["status"] = payload.get("status")
        pending["duration_ms"] = _duration_to_ms(payload.get("duration"))
        pending["exit_code"] = payload.get("exit_code")
        pending["metadata"].update(
            {
                "call_id": payload.get("call_id"),
                "process_id": payload.get("process_id"),
                "source": payload.get("source"),
            }
        )
        pending["input_data"] = {
            "command": payload.get("command"),
            "cwd": payload.get("cwd"),
            "parsed_cmd": payload.get("parsed_cmd"),
        }
        pending["output_data"] = {
            "aggregated_output": payload.get("aggregated_output"),
            "stdout": payload.get("stdout"),
            "stderr": payload.get("stderr"),
            "status": payload.get("status"),
            "exit_code": payload.get("exit_code"),
        }
        self._merge_paths(pending, extract_path_buckets(payload.get("parsed_cmd"), cwd=cwd))
        return self._finalize_pending_call(call_id, pending, cwd)

    def _build_web_search_record(
        self,
        payload: dict[str, Any],
        turn_id: str,
        cwd: str,
    ) -> ObservationRecord:
        action = payload.get("action") or {}
        action_type = str(action.get("type") or "search")
        tool_name = {
            "search": "web.search",
            "open_page": "web.open_page",
            "find_in_page": "web.find_in_page",
        }.get(action_type, f"web.{action_type}")
        path_buckets = extract_path_buckets(action, cwd=cwd)

        return ObservationRecord(
            logical_id=f"web-search:{payload.get('call_id') or payload.get('query')}",
            turn_id=turn_id,
            kind="tool",
            tool_family="web",
            tool_name=tool_name,
            status=payload.get("status") or "completed",
            input_data={"query": payload.get("query"), "action": action},
            codex_event_types=["web_search_end"],
            metadata={
                "call_id": payload.get("call_id"),
                "action_type": action_type,
            },
            debug_payloads={"web_search_end": payload},
            **path_buckets,
        )

    def _build_patch_apply_record(
        self,
        state: SessionState,
        payload: dict[str, Any],
        turn_id: str,
        cwd: str,
    ) -> ObservationRecord:
        call_id = str(payload.get("call_id") or f"patch:{turn_id}")
        pending = state.pending_tool_calls.pop(call_id, None)
        if not pending:
            pending = self._new_pending_call(
                turn_id=turn_id,
                call_id=call_id,
                raw_name="apply_patch",
                input_data=None,
                payload_type="patch_apply_end",
                payload=payload,
                cwd=cwd,
            )
        else:
            _ensure_pending_defaults(pending, call_id, turn_id=turn_id, raw_name="apply_patch")
            pending["codex_event_types"].append("patch_apply_end")
            pending["debug_payloads"]["patch_apply_end"] = payload

        pending["tool_family"] = "codex"
        pending["tool_name"] = "codex.patch_apply"
        pending["status"] = payload.get("status") or ("completed" if payload.get("success") else "failed")
        pending["output_data"] = {
            "stdout": payload.get("stdout"),
            "stderr": payload.get("stderr"),
            "success": payload.get("success"),
            "changes": payload.get("changes"),
        }
        pending["metadata"]["change_summary"] = _summarize_patch_changes(payload.get("changes"))
        self._merge_paths(pending, extract_path_buckets(payload.get("changes"), pending.get("input_data"), cwd=cwd))
        return self._finalize_pending_call(call_id, pending, cwd)

    def _build_task_event(self, payload: dict[str, Any], turn_id: str, tool_name: str) -> ObservationRecord:
        return ObservationRecord(
            logical_id=f"{tool_name}:{turn_id}",
            turn_id=turn_id,
            kind="event",
            tool_family=None,
            tool_name=tool_name,
            status="completed",
            output_data=payload,
            duration_ms=payload.get("duration_ms"),
            codex_event_types=[str(payload.get("type"))],
            debug_payloads={str(payload.get("type")): payload},
        )

    def _build_single_event_observation(
        self,
        payload: dict[str, Any],
        turn_id: str,
        cwd: str,
        *,
        tool_family: str,
        tool_name: str,
    ) -> ObservationRecord:
        path_buckets = extract_path_buckets(payload, cwd=cwd)
        return ObservationRecord(
            logical_id=f"{tool_name}:{payload.get('call_id') or payload.get('path') or turn_id}",
            turn_id=turn_id,
            kind="tool",
            tool_family=tool_family,
            tool_name=tool_name,
            status=payload.get("status") or "completed",
            input_data=payload,
            codex_event_types=[str(payload.get("type"))],
            debug_payloads={str(payload.get("type")): payload},
            **path_buckets,
        )

    def _remember_token_count(self, parsed: ParsedDelta, turn_id: str | None, payload: dict[str, Any]) -> None:
        if not turn_id:
            return

        info = payload.get("info") or {}
        update = parsed.update_turn(turn_id)
        update.trace_metadata["token_summary"] = {
            "total_token_usage": info.get("total_token_usage"),
            "last_token_usage": info.get("last_token_usage"),
            "model_context_window": info.get("model_context_window"),
        }
        update.trace_metadata["rate_limits"] = payload.get("rate_limits")

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
        base_name = _tool_name_from_payload(payload, default=payload_type.removesuffix("_end"))
        tool_family, tool_name = _normalize_tool_identity(base_name)
        input_data = begin_event["payload"] if begin_event else _extract_input_payload(payload)
        output_data = _extract_output_payload(payload)
        path_buckets = extract_path_buckets(input_data, output_data, payload, cwd=cwd)

        return ObservationRecord(
            logical_id=f"{payload_type}:{payload.get('call_id') or payload.get('id') or payload.get('process_id') or turn_id}",
            turn_id=begin_event["turn_id"] if begin_event else turn_id,
            kind="tool",
            tool_family=tool_family,
            tool_name=tool_name,
            status=payload.get("status") or "completed",
            input_data=input_data,
            output_data=output_data,
            duration_ms=payload.get("duration_ms"),
            codex_event_types=[payload_type, begin_event["payload_type"]] if begin_event else [payload_type],
            metadata={
                "call_id": payload.get("call_id"),
                "tool_name": payload.get("tool_name"),
            },
            debug_payloads={
                payload_type: payload,
                **({begin_event["payload_type"]: begin_event["payload"]} if begin_event else {}),
            },
            **path_buckets,
        )

    def _new_pending_call(
        self,
        *,
        turn_id: str,
        call_id: str,
        raw_name: str,
        input_data: Any,
        payload_type: str,
        payload: dict[str, Any],
        cwd: str,
        status: str | None = None,
    ) -> dict[str, Any]:
        tool_family, tool_name = _normalize_tool_identity(raw_name)
        pending = {
            "turn_id": turn_id,
            "raw_name": raw_name,
            "tool_family": tool_family,
            "tool_name": tool_name,
            "status": status,
            "input_data": input_data,
            "output_data": None,
            "duration_ms": None,
            "exit_code": None,
            "codex_event_types": [payload_type],
            "metadata": {"call_id": call_id},
            "debug_payloads": {payload_type: payload},
            "read_paths": [],
            "write_paths": [],
            "search_paths": [],
            "referenced_paths": [],
            "pty_session_id": input_data.get("session_id") if isinstance(input_data, dict) else None,
        }
        self._merge_paths(pending, extract_path_buckets(input_data, payload, cwd=cwd))
        return pending

    def _store_pending_call(self, state: SessionState, call_id: str, pending: dict[str, Any]) -> None:
        existing = state.pending_tool_calls.get(call_id)
        if not existing:
            state.pending_tool_calls[call_id] = pending
            return

        _ensure_pending_defaults(existing, call_id, turn_id=pending.get("turn_id"), raw_name=pending.get("raw_name"))
        existing["turn_id"] = pending.get("turn_id") or existing.get("turn_id")
        existing["raw_name"] = pending.get("raw_name") or existing.get("raw_name")
        existing["tool_family"] = pending.get("tool_family") or existing.get("tool_family")
        existing["tool_name"] = pending.get("tool_name") or existing.get("tool_name")
        existing["status"] = pending.get("status") or existing.get("status")
        if existing.get("input_data") is None:
            existing["input_data"] = pending.get("input_data")
        if existing.get("output_data") is None:
            existing["output_data"] = pending.get("output_data")
        existing["metadata"].update(pending.get("metadata", {}))
        existing["debug_payloads"].update(pending.get("debug_payloads", {}))
        for event_type in pending.get("codex_event_types", []):
            if event_type not in existing["codex_event_types"]:
                existing["codex_event_types"].append(event_type)
        self._merge_paths(
            existing,
            {key: pending.get(key, []) for key in ("read_paths", "write_paths", "search_paths", "referenced_paths")},
        )

    def _finalize_pending_call(
        self,
        call_id: str,
        pending: dict[str, Any],
        cwd: str,
        *,
        force_pending: bool = False,
    ) -> ObservationRecord:
        _ensure_pending_defaults(pending, call_id)
        metadata = dict(pending["metadata"])
        if force_pending and not pending.get("status"):
            metadata["pending"] = True

        return ObservationRecord(
            logical_id=str(pending["metadata"].get("call_id") or call_id),
            turn_id=str(pending["turn_id"]),
            kind="tool",
            tool_family=pending.get("tool_family"),
            tool_name=str(pending["tool_name"]),
            status=pending.get("status") or ("pending" if force_pending else "completed"),
            input_data=pending.get("input_data"),
            output_data=pending.get("output_data"),
            duration_ms=pending.get("duration_ms"),
            exit_code=pending.get("exit_code"),
            codex_event_types=list(dict.fromkeys(pending.get("codex_event_types", []))),
            read_paths=list(dict.fromkeys(pending.get("read_paths", []))),
            write_paths=list(dict.fromkeys(pending.get("write_paths", []))),
            search_paths=list(dict.fromkeys(pending.get("search_paths", []))),
            referenced_paths=list(dict.fromkeys(pending.get("referenced_paths", []))),
            metadata=metadata,
            debug_payloads=dict(pending.get("debug_payloads", {})),
            pty_session_id=pending.get("pty_session_id"),
        )

    def _merge_paths(self, pending: dict[str, Any], buckets: dict[str, list[str]]) -> None:
        for key in ("read_paths", "write_paths", "search_paths", "referenced_paths"):
            values = pending.setdefault(key, [])
            for value in buckets.get(key, []):
                if value not in values:
                    values.append(value)


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


def _normalize_tool_identity(raw_name: str) -> tuple[str | None, str]:
    if raw_name == "exec_command":
        return "terminal", "terminal.exec"
    if raw_name == "write_stdin":
        return "terminal", "terminal.poll"
    if raw_name == "apply_patch":
        return "codex", "codex.patch_apply"
    if raw_name == "request_user_input":
        return "codex", "codex.request_user_input"
    if raw_name.startswith("mcp_") or raw_name == "mcp_tool_call":
        return "mcp", f"mcp.{raw_name.removeprefix('mcp_')}"
    if raw_name.startswith("web."):
        return "web", raw_name
    return "codex", raw_name


def _raw_name_from_hook_tool(tool_name: Any) -> str:
    if tool_name == "Bash":
        return "exec_command"
    if tool_name:
        return str(tool_name)
    return "hook_tool"


def _ensure_pending_defaults(
    pending: dict[str, Any],
    call_id: str,
    *,
    turn_id: str | None = None,
    raw_name: str | None = None,
) -> None:
    pending_raw_name = str(pending.get("raw_name") or raw_name or pending.get("tool_name") or "tool")
    tool_family, tool_name = _normalize_tool_identity(pending_raw_name)
    pending.setdefault("turn_id", turn_id)
    pending.setdefault("raw_name", pending_raw_name)
    pending.setdefault("tool_family", tool_family)
    pending.setdefault("tool_name", tool_name)
    pending.setdefault("status", None)
    pending.setdefault("input_data", None)
    pending.setdefault("output_data", None)
    pending.setdefault("duration_ms", None)
    pending.setdefault("exit_code", None)
    pending.setdefault("codex_event_types", [])
    pending.setdefault("metadata", {"call_id": call_id})
    pending.setdefault("debug_payloads", {})
    pending.setdefault("read_paths", [])
    pending.setdefault("write_paths", [])
    pending.setdefault("search_paths", [])
    pending.setdefault("referenced_paths", [])
    pending.setdefault("pty_session_id", None)
    pending["metadata"].setdefault("call_id", call_id)


def _duration_to_ms(value: Any) -> int | None:
    if isinstance(value, (int, float)):
        return int(float(value) * 1000)
    if not isinstance(value, dict):
        return None
    secs = int(value.get("secs", 0))
    nanos = int(value.get("nanos", 0))
    return secs * 1000 + nanos // 1_000_000


def _summarize_patch_changes(changes: Any) -> dict[str, int]:
    summary = {"added": 0, "modified": 0, "deleted": 0, "touched": 0}
    if not isinstance(changes, dict):
        return summary

    summary["touched"] = len(changes)
    for details in changes.values():
        change_type = str(details.get("type") or "").lower() if isinstance(details, dict) else ""
        if change_type == "add":
            summary["added"] += 1
        elif change_type == "delete":
            summary["deleted"] += 1
        else:
            summary["modified"] += 1
    return summary
