from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def discover_project_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
        if (candidate / ".env").exists() and (candidate / "README.md").exists():
            return candidate
    return current


@dataclass(frozen=True, slots=True)
class HookConfig:
    project_root: Path
    state_dir: Path
    max_payload_bytes: int
    log_level: str
    capture_agent_messages: bool
    public_key: str | None
    secret_key: str | None
    base_url: str | None

    @property
    def tracing_enabled(self) -> bool:
        return bool(self.public_key and self.secret_key and self.base_url)

    @classmethod
    def from_environment(cls, cwd: str | os.PathLike[str]) -> "HookConfig":
        project_root = discover_project_root(Path(cwd))
        load_dotenv(project_root / ".env", override=False)

        state_dir = Path(
            os.getenv("LANGFUSE_CODEX_STATE_DIR", project_root / ".codex" / "langfuse-state")
        )
        if not state_dir.is_absolute():
            state_dir = (project_root / state_dir).resolve()

        return cls(
            project_root=project_root,
            state_dir=state_dir,
            max_payload_bytes=int(os.getenv("LANGFUSE_CODEX_MAX_PAYLOAD_BYTES", "131072")),
            log_level=os.getenv("LANGFUSE_CODEX_LOG_LEVEL", "INFO").upper(),
            capture_agent_messages=os.getenv("LANGFUSE_CODEX_CAPTURE_AGENT_MESSAGES", "").lower() in {"1", "true", "yes"},
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            base_url=os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST"),
        )
