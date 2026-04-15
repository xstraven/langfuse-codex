from langfuse_codex.redaction import extract_path_buckets, sanitize_payload


def test_sanitize_payload_redacts_and_truncates_without_hiding_safe_operational_keys() -> None:
    payload = {
        "Authorization": "Bearer secret-token-value",
        "max_output_tokens": 3000,
        "session_id": 12345,
        "text": "sk-lf-1234567890 " + "x" * 300,
    }

    sanitized, metadata = sanitize_payload(payload, 80)

    assert sanitized["Authorization"] == "[REDACTED]"
    assert sanitized["max_output_tokens"] == 3000
    assert sanitized["session_id"] == 12345
    assert "[REDACTED_TOKEN]" in sanitized["text"]
    assert metadata["redacted"] is True
    assert metadata["truncated"] is True


def test_extract_path_buckets_prefers_structured_values_and_ignores_urls() -> None:
    payload = {
        "parsed_cmd": [{"type": "read", "path": "/Users/david/projects/langfuse-codex/src/langfuse_codex/hook.py"}],
        "command": "open https://help.openai.com/en/articles/123",
        "text": "see https://help.openai.com and /not/a/url",
    }

    buckets = extract_path_buckets(payload, cwd="/Users/david/projects/langfuse-codex")

    assert buckets["read_paths"] == ["/Users/david/projects/langfuse-codex/src/langfuse_codex/hook.py"]
    assert "referenced_paths" not in buckets


def test_extract_path_buckets_reads_patch_headers_as_writes() -> None:
    patch = """*** Begin Patch
*** Update File: /Users/david/projects/langfuse-codex/src/langfuse_codex/transcript.py
*** End Patch
"""

    buckets = extract_path_buckets(patch, cwd="/Users/david/projects/langfuse-codex")

    assert buckets["write_paths"] == ["/Users/david/projects/langfuse-codex/src/langfuse_codex/transcript.py"]
