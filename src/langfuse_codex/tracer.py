from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import langfuse as langfuse_module
from langfuse import Langfuse

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
            return turn_state

        trace_id = self.client.create_trace_id(seed=f"{trace_context.session_id}:{trace_context.turn_id}")
        trace_meta = self._trace_metadata(trace_context)

        with self.client.start_as_current_observation(
            trace_context={"trace_id": trace_id},
            name="codex-turn",
            as_type="agent",
            input=prompt,
            metadata={"turn_id": trace_context.turn_id, "cwd": trace_context.cwd, "model": trace_context.model},
        ):
            with langfuse_module.propagate_attributes(
                session_id=trace_context.session_id,
                trace_name="codex-turn",
                metadata=trace_meta,
            ):
                root_observation_id = self.client.get_current_observation_id()

        turn_state = TurnState(
            trace_id=trace_id,
            root_observation_id=root_observation_id,
            prompt=prompt,
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

        metadata = dict(observation.metadata)
        metadata["input_sanitization"] = input_meta
        metadata["output_sanitization"] = output_meta
        metadata["turn_id"] = observation.turn_id

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
            with langfuse_module.propagate_attributes(
                session_id=trace_context.session_id,
                trace_name="codex-turn",
                metadata=self._trace_metadata(trace_context),
            ):
                pass

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

        with langfuse_module.propagate_attributes(
            session_id=trace_context.session_id,
            trace_name="codex-turn",
            metadata=self._trace_metadata(trace_context),
        ):
            with self.client.start_as_current_observation(
                trace_context=parent,
                name="assistant-response",
                as_type="span",
                output=sanitized_output,
                metadata={
                    "turn_id": trace_context.turn_id,
                    "output_sanitization": output_meta,
                },
            ):
                pass

    def flush(self) -> None:
        self.client.flush()

    def shutdown(self) -> None:
        self.client.shutdown()

    def _trace_metadata(self, context: TraceContext) -> dict[str, str]:
        metadata = {"cwd": context.cwd}
        if context.model:
            metadata["model"] = context.model[:200]
        if context.transcript_path:
            metadata["transcript_path"] = context.transcript_path[:200]
        return metadata
