from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import langfuse as langfuse_module
from langfuse import Langfuse
from langfuse._client.attributes import (
    LangfuseOtelSpanAttributes,
    _flatten_and_serialize_metadata,
    create_trace_attributes,
)

from langfuse_codex.config import HookConfig
from langfuse_codex.redaction import sanitize_payload
from langfuse_codex.state import SessionState, TurnState
from langfuse_codex.transcript import ObservationRecord

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TraceContext:
    session_id: str
    turn_id: str
    cwd: str
    model: str | None
    transcript_path: str | None
    trace_name: str
    trace_metadata: dict[str, Any]


class LangfuseTracer:
    def __init__(
        self,
        config: HookConfig,
        *,
        client: Langfuse | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.logger = logger or LOGGER
        self.client = client or Langfuse(
            public_key=config.public_key,
            secret_key=config.secret_key,
            base_url=config.base_url,
            tracing_enabled=config.tracing_enabled,
        )

    def ensure_turn_root(
        self,
        state: SessionState,
        trace_context: TraceContext,
        *,
        prompt: str | None = None,
    ) -> TurnState:
        turn_state = state.turns.get(trace_context.turn_id)
        if turn_state:
            if prompt and not turn_state.prompt:
                turn_state.prompt = prompt
            if trace_context.trace_name:
                turn_state.trace_name = trace_context.trace_name
            if trace_context.trace_metadata:
                turn_state.trace_metadata.update(trace_context.trace_metadata)
            return turn_state

        trace_id = self.client.create_trace_id(seed=f"{trace_context.session_id}:{trace_context.turn_id}")
        trace_meta = self._trace_metadata(trace_context)
        sanitized_prompt, _ = sanitize_payload(prompt, self.config.max_payload_bytes)
        metadata = {"turn_id": trace_context.turn_id}
        if trace_context.cwd:
            metadata["cwd"] = trace_context.cwd
        if trace_context.model:
            metadata["model"] = trace_context.model

        with self.client.start_as_current_observation(
            trace_context={"trace_id": trace_id},
            name=trace_context.trace_name,
            as_type="agent",
            input=sanitized_prompt,
            metadata=metadata,
        ):
            self._apply_trace_attributes(
                trace_context,
                trace_input=sanitized_prompt,
                trace_output=None,
            )
            root_observation_id = self.client.get_current_observation_id()

        turn_state = TurnState(
            trace_id=trace_id,
            root_observation_id=root_observation_id,
            trace_name=trace_context.trace_name,
            prompt=prompt,
            trace_metadata=trace_meta,
        )
        state.turns[trace_context.turn_id] = turn_state
        return turn_state

    def record_observation(
        self,
        turn_state: TurnState,
        trace_context: TraceContext,
        observation: ObservationRecord,
    ) -> None:
        sanitized_input, input_meta = sanitize_payload(observation.input_data, self.config.max_payload_bytes)
        sanitized_output, output_meta = sanitize_payload(observation.output_data, self.config.max_payload_bytes)
        metadata = {
            "tool_family": observation.tool_family,
            "tool_name": observation.tool_name,
            "status": observation.status,
            "duration_ms": observation.duration_ms,
            "exit_code": observation.exit_code,
            "codex_event_types": observation.codex_event_types,
            "read_paths": observation.read_paths,
            "write_paths": observation.write_paths,
            "search_paths": observation.search_paths,
            "referenced_paths": observation.referenced_paths,
            "pty_session_id": observation.pty_session_id,
            "debug_payloads": observation.debug_payloads,
            **dict(observation.metadata),
        }
        sanitized_metadata, metadata_meta = sanitize_payload(metadata, self.config.max_payload_bytes)
        metadata = dict(sanitized_metadata)
        metadata["input_sanitization"] = input_meta
        metadata["output_sanitization"] = output_meta
        metadata["metadata_sanitization"] = metadata_meta

        parent = {"trace_id": turn_state.trace_id}
        if turn_state.root_observation_id:
            parent["parent_span_id"] = turn_state.root_observation_id

        with self.client.start_as_current_observation(
            trace_context=parent,
            name=observation.name,
            as_type="tool" if observation.kind == "tool" else "span",
            input=sanitized_input,
            output=sanitized_output,
            metadata=metadata,
        ):
            self._apply_trace_attributes(
                trace_context,
                trace_input=self._trace_input(turn_state),
                trace_output=self._trace_output(turn_state),
            )

    def record_assistant_response(
        self,
        turn_state: TurnState,
        trace_context: TraceContext,
        message: str,
    ) -> None:
        sanitized_output, output_meta = sanitize_payload(message, self.config.max_payload_bytes)
        parent = {"trace_id": turn_state.trace_id}
        if turn_state.root_observation_id:
            parent["parent_span_id"] = turn_state.root_observation_id
        turn_state.last_assistant_message = message

        with self.client.start_as_current_observation(
            trace_context=parent,
            name="assistant-response",
            as_type="span",
            output=sanitized_output,
            metadata={
                "output_sanitization": output_meta,
            },
        ):
            self._apply_trace_attributes(
                trace_context,
                trace_input=self._trace_input(turn_state),
                trace_output=sanitized_output,
            )

    def flush(self) -> None:
        self.client.flush()

    def shutdown(self) -> None:
        self.client.shutdown()

    def _trace_metadata(self, context: TraceContext) -> dict[str, Any]:
        metadata = dict(context.trace_metadata)
        if context.cwd:
            metadata.setdefault("cwd", context.cwd)
        if context.model:
            metadata.setdefault("model", context.model[:200])
        return metadata

    def _trace_input(self, turn_state: TurnState) -> Any:
        sanitized, _ = sanitize_payload(turn_state.prompt, self.config.max_payload_bytes)
        return sanitized

    def _trace_output(self, turn_state: TurnState) -> Any:
        sanitized, _ = sanitize_payload(turn_state.last_assistant_message, self.config.max_payload_bytes)
        return sanitized

    def _apply_trace_attributes(
        self,
        context: TraceContext,
        *,
        trace_input: Any,
        trace_output: Any,
    ) -> None:
        trace_meta = self._trace_metadata(context)
        current_span = self.client._get_current_otel_span()
        if current_span is None or not current_span.is_recording():
            return

        with langfuse_module.propagate_attributes(
            session_id=context.session_id,
            trace_name=context.trace_name,
            metadata={key: str(value)[:200] for key, value in trace_meta.items() if value is not None},
        ):
            current_span.set_attribute(LangfuseOtelSpanAttributes.TRACE_NAME, context.trace_name)
            current_span.set_attribute(LangfuseOtelSpanAttributes.TRACE_SESSION_ID, context.session_id)
            for key, value in _flatten_and_serialize_metadata(trace_meta, "trace").items():
                current_span.set_attribute(key, value)
            for key, value in create_trace_attributes(input=trace_input, output=trace_output).items():
                current_span.set_attribute(key, value)
            self.client.set_current_trace_io(input=trace_input, output=trace_output)
