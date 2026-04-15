from langfuse_codex.redaction import extract_file_paths, sanitize_payload


def test_sanitize_payload_redacts_and_truncates() -> None:
    payload = {
        "Authorization": "Bearer secret-token-value",
        "text": "sk-lf-1234567890 " + "x" * 300,
    }

    sanitized, metadata = sanitize_payload(payload, 80)

    assert sanitized["Authorization"] == "[REDACTED]"
    assert "[REDACTED_TOKEN]" in sanitized["text"]
    assert metadata["redacted"] is True
    assert metadata["truncated"] is True


def test_extract_file_paths_prefers_structured_values() -> None:
    payload = {
        "path": "src/langfuse_codex/hook.py",
        "command": "sed -n '1,5p' src/langfuse_codex/transcript.py",
    }

    paths = extract_file_paths(payload, cwd="/Users/david/projects/langfuse-codex")

    assert "/Users/david/projects/langfuse-codex/src/langfuse_codex/hook.py" in paths
    assert "/Users/david/projects/langfuse-codex/src/langfuse_codex/transcript.py" in paths

