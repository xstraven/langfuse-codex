from pathlib import Path

from langfuse_codex.state import SessionStateStore


def test_session_state_round_trip(tmp_path: Path) -> None:
    store = SessionStateStore(tmp_path)

    with store.open("session-1") as state:
        state.transcript_offset = 42
        state.active_turn_id = "turn-1"
        state.session_metadata["cwd"] = "/repo"

    with store.open("session-1") as state:
        assert state.transcript_offset == 42
        assert state.active_turn_id == "turn-1"
        assert state.session_metadata["cwd"] == "/repo"

