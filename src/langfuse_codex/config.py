from __future__ import annotations

import hashlib
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values, load_dotenv


def discover_project_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
        if (candidate / ".env").exists() and (candidate / "README.md").exists():
            return candidate
    return current


def default_app_support_dir() -> Path:
    configured = os.getenv("LANGFUSE_CODEX_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "langfuse-codex"
    return Path(os.getenv("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "langfuse-codex"


def default_config_path() -> Path:
    return default_app_support_dir() / "env"


def default_queue_dir() -> Path:
    return default_app_support_dir() / "queue"


def default_state_root() -> Path:
    return default_app_support_dir() / "state"


def global_config_values() -> dict[str, str]:
    values: dict[str, str] = {}
    values.update(_dotenv_dict(Path(os.getenv("LANGFUSE_CODEX_CONFIG", default_config_path()))))
    values.update(os.environ)
    return values


def resolve_queue_dir(values: Mapping[str, str] | None = None) -> Path:
    source = values or global_config_values()
    configured = source.get("LANGFUSE_CODEX_QUEUE_DIR")
    path = Path(configured).expanduser() if configured else default_queue_dir()
    return path if path.is_absolute() else (default_app_support_dir() / path).resolve()


def project_state_dir_name(project_root: Path) -> str:
    resolved = str(project_root.resolve())
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", project_root.name).strip(".-")
    return f"{name or 'project'}-{digest}"


@dataclass(frozen=True, slots=True)
class HookConfig:
    project_root: Path
    state_dir: Path
    max_payload_bytes: int
    log_level: str
    capture_agent_messages: bool
    delivery_mode: str
    sender_idle_seconds: float
    queue_max_events: int
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

        return cls.from_values(
            project_root=project_root,
            values=os.environ,
            default_state_dir=project_root / ".codex" / "langfuse-state",
        )

    @classmethod
    def from_service_environment(cls, cwd: str | os.PathLike[str]) -> "HookConfig":
        project_root = discover_project_root(Path(cwd))
        values: dict[str, str] = {}
        values.update(_dotenv_dict(Path(os.getenv("LANGFUSE_CODEX_CONFIG", default_config_path()))))
        values.update(_dotenv_dict(project_root / ".env"))
        values.update(os.environ)

        state_root = Path(values.get("LANGFUSE_CODEX_STATE_ROOT", default_state_root()))
        if not state_root.is_absolute():
            state_root = (default_app_support_dir() / state_root).resolve()
        default_state_dir = state_root / project_state_dir_name(project_root)

        return cls.from_values(
            project_root=project_root,
            values=values,
            default_state_dir=default_state_dir,
        )

    @classmethod
    def from_values(
        cls,
        *,
        project_root: Path,
        values: Mapping[str, str],
        default_state_dir: Path,
    ) -> "HookConfig":
        state_dir = Path(values.get("LANGFUSE_CODEX_STATE_DIR", default_state_dir))
        if not state_dir.is_absolute():
            state_dir = (project_root / state_dir).resolve()

        return cls(
            project_root=project_root,
            state_dir=state_dir,
            max_payload_bytes=int(values.get("LANGFUSE_CODEX_MAX_PAYLOAD_BYTES", "131072")),
            log_level=values.get("LANGFUSE_CODEX_LOG_LEVEL", "INFO").upper(),
            capture_agent_messages=values.get("LANGFUSE_CODEX_CAPTURE_AGENT_MESSAGES", "").lower()
            in {"1", "true", "yes"},
            delivery_mode=values.get("LANGFUSE_CODEX_DELIVERY_MODE", "queued").lower(),
            sender_idle_seconds=float(values.get("LANGFUSE_CODEX_SENDER_IDLE_SECONDS", "30")),
            queue_max_events=int(values.get("LANGFUSE_CODEX_QUEUE_MAX_EVENTS", "10000")),
            public_key=values.get("LANGFUSE_PUBLIC_KEY"),
            secret_key=values.get("LANGFUSE_SECRET_KEY"),
            base_url=values.get("LANGFUSE_BASE_URL") or values.get("LANGFUSE_HOST"),
        )


def _dotenv_dict(path: Path) -> dict[str, str]:
    path = path.expanduser()
    if not path.exists():
        return {}
    values = dotenv_values(path)
    return {key: value for key, value in values.items() if value is not None}
