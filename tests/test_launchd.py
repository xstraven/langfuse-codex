import plistlib
import sys
from pathlib import Path

from langfuse_codex import launchd


def test_launchd_builds_plist_with_absolute_program_and_logs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("LANGFUSE_CODEX_HOME", raising=False)

    plist = launchd.build_plist()

    assert plist["Label"] == "com.langfuse.codex"
    assert plist["RunAtLoad"] is True
    assert plist["KeepAlive"] is True
    assert plist["ProgramArguments"][0] == sys.executable
    assert Path(plist["ProgramArguments"][0]).is_absolute()
    assert plist["ProgramArguments"][1:3] == ["-m", "langfuse_codex.service"]
    assert str(tmp_path / "Library" / "Logs" / "langfuse-codex" / "service.log") == plist["StandardOutPath"]
    assert plist["StandardOutPath"] == plist["StandardErrorPath"]


def test_launchd_write_plist_creates_directories_and_valid_plist(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("LANGFUSE_CODEX_HOME", raising=False)
    target = tmp_path / "LaunchAgents" / "com.langfuse.codex.plist"

    written = launchd.write_plist(target)

    assert written == target
    with written.open("rb") as handle:
        plist = plistlib.load(handle)
    assert plist["Label"] == "com.langfuse.codex"
    assert (tmp_path / "Library" / "Application Support" / "langfuse-codex").is_dir()
    assert (tmp_path / "Library" / "Logs" / "langfuse-codex").is_dir()
