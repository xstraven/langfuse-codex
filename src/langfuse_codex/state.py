from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import fcntl

MAX_RECENT_EVENT_HASHES = 2048
MAX_TURNS_PER_SESSION = 32


@dataclass(slots=True)
class TurnState:
    trace_id: str
    root_observation_id: str | None = None
    trace_name: str | None = None
    prompt: str | None = None
    last_assistant_message: str | None = None
    trace_metadata: dict[str, Any] = field(default_factory=dict)
    closed: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TurnState":
        return cls(
            trace_id=data["trace_id"],
            root_observation_id=data.get("root_observation_id"),
            trace_name=data.get("trace_name"),
            prompt=data.get("prompt"),
            last_assistant_message=data.get("last_assistant_message"),
            trace_metadata=dict(data.get("trace_metadata", {})),
            closed=bool(data.get("closed", False)),
        )


@dataclass(slots=True)
class SessionState:
    session_id: str
    transcript_path: str | None = None
    transcript_offset: int = 0
    active_turn_id: str | None = None
    recent_event_hashes: list[str] = field(default_factory=list)
    turns: dict[str, TurnState] = field(default_factory=dict)
    pending_tool_calls: dict[str, dict[str, Any]] = field(default_factory=dict)
    pending_events: dict[str, dict[str, Any]] = field(default_factory=dict)
    session_metadata: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any], session_id: str) -> "SessionState":
        return cls(
            session_id=session_id,
            transcript_path=data.get("transcript_path"),
            transcript_offset=int(data.get("transcript_offset", 0)),
            active_turn_id=data.get("active_turn_id"),
            recent_event_hashes=list(data.get("recent_event_hashes", [])),
            turns={
                turn_id: TurnState.from_dict(turn_data)
                for turn_id, turn_data in data.get("turns", {}).items()
            },
            pending_tool_calls=dict(data.get("pending_tool_calls", {})),
            pending_events=dict(data.get("pending_events", {})),
            session_metadata=dict(data.get("session_metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "transcript_path": self.transcript_path,
            "transcript_offset": self.transcript_offset,
            "active_turn_id": self.active_turn_id,
            "recent_event_hashes": self.recent_event_hashes[-MAX_RECENT_EVENT_HASHES:],
            "turns": {turn_id: asdict(turn_state) for turn_id, turn_state in self.turns.items()},
            "pending_tool_calls": self.pending_tool_calls,
            "pending_events": self.pending_events,
            "session_metadata": self.session_metadata,
        }

    def remember_event_hash(self, event_hash: str) -> None:
        self.recent_event_hashes.append(event_hash)
        if len(self.recent_event_hashes) > MAX_RECENT_EVENT_HASHES:
            self.recent_event_hashes = self.recent_event_hashes[-MAX_RECENT_EVENT_HASHES:]

    def trim_turns(self) -> None:
        if len(self.turns) <= MAX_TURNS_PER_SESSION:
            return

        removable = [turn_id for turn_id, state in self.turns.items() if state.closed]
        for turn_id in removable[: len(self.turns) - MAX_TURNS_PER_SESSION]:
            self.turns.pop(turn_id, None)

    def clear_pending_for_turn(self, turn_id: str) -> None:
        self.pending_tool_calls = {
            key: value for key, value in self.pending_tool_calls.items() if value.get("turn_id") != turn_id
        }
        self.pending_events = {
            key: value for key, value in self.pending_events.items() if value.get("turn_id") != turn_id
        }


class SessionStateStore:
    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def open(self, session_id: str) -> Iterator[SessionState]:
        lock_path = self.state_dir / f"{session_id}.lock"
        state_path = self.state_dir / f"{session_id}.json"

        with lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            state = self._load_state(session_id, state_path)
            try:
                yield state
            finally:
                state.trim_turns()
                self._save_state(state_path, state)
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _load_state(self, session_id: str, state_path: Path) -> SessionState:
        if not state_path.exists():
            return SessionState(session_id=session_id)
        with state_path.open("r", encoding="utf-8") as handle:
            return SessionState.from_dict(json.load(handle), session_id=session_id)

    def _save_state(self, state_path: Path, state: SessionState) -> None:
        tmp_path = state_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(state.to_dict(), handle, indent=2, sort_keys=True)
        os.replace(tmp_path, state_path)
