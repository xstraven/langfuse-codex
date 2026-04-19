from __future__ import annotations

import argparse
import os
import plistlib
import subprocess
import sys
from pathlib import Path
from typing import Any

from langfuse_codex.config import default_app_support_dir

LABEL = "com.langfuse.codex"


def launch_agents_dir() -> Path:
    return Path.home() / "Library" / "LaunchAgents"


def plist_path() -> Path:
    return launch_agents_dir() / f"{LABEL}.plist"


def log_dir() -> Path:
    return Path.home() / "Library" / "Logs" / "langfuse-codex"


def log_path() -> Path:
    return log_dir() / "service.log"


def service_target() -> str:
    return f"gui/{os.getuid()}/{LABEL}"


def bootstrap_target() -> str:
    return f"gui/{os.getuid()}"


def service_program_arguments() -> list[str]:
    return [
        sys.executable,
        "-m",
        "langfuse_codex.service",
    ]


def build_plist(program_arguments: list[str] | None = None) -> dict[str, Any]:
    service_log_path = log_path()
    return {
        "Label": LABEL,
        "ProgramArguments": program_arguments or service_program_arguments(),
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(service_log_path),
        "StandardErrorPath": str(service_log_path),
        "WorkingDirectory": str(default_app_support_dir()),
    }


def write_plist(path: Path | None = None, *, program_arguments: list[str] | None = None) -> Path:
    target = path or plist_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    default_app_support_dir().mkdir(parents=True, exist_ok=True)
    log_dir().mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        plistlib.dump(build_plist(program_arguments), handle, sort_keys=True)
    return target


def install() -> int:
    target = write_plist()
    _run_launchctl(["bootout", bootstrap_target(), str(target)], check=False)
    _run_launchctl(["bootstrap", bootstrap_target(), str(target)])
    _run_launchctl(["enable", service_target()], check=False)
    _run_launchctl(["kickstart", "-k", service_target()], check=False)
    print(f"Installed {LABEL} at {target}")
    return 0


def uninstall() -> int:
    target = plist_path()
    _run_launchctl(["bootout", bootstrap_target(), str(target)], check=False)
    target.unlink(missing_ok=True)
    print(f"Uninstalled {LABEL}")
    return 0


def start() -> int:
    _run_launchctl(["enable", service_target()], check=False)
    _run_launchctl(["kickstart", "-k", service_target()])
    return 0


def stop() -> int:
    _run_launchctl(["disable", service_target()], check=False)
    _run_launchctl(["kill", "TERM", service_target()], check=False)
    return 0


def restart() -> int:
    _run_launchctl(["enable", service_target()], check=False)
    _run_launchctl(["kickstart", "-k", service_target()])
    return 0


def status() -> int:
    return _run_launchctl(["print", service_target()], check=False).returncode


def logs() -> int:
    print(log_path())
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return args.func()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage the macOS launchd agent for langfuse-codex")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, func in {
        "install": install,
        "uninstall": uninstall,
        "start": start,
        "stop": stop,
        "restart": restart,
        "status": status,
        "logs": logs,
    }.items():
        command = subparsers.add_parser(name)
        command.set_defaults(func=func)
    return parser.parse_args(argv)


def _run_launchctl(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["launchctl", *args], text=True, check=check)


if __name__ == "__main__":
    raise SystemExit(main())
