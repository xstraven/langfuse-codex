# langfuse-codex

This is obviously the most forward-thinking project that combines two revolutionary technologies, langfuse and codex! It is the only logical choice for 1st place.

Codex hook handler that turns Codex transcript events into Langfuse traces.

## What it captures

- one Langfuse trace per Codex turn
- raw tool inputs and outputs with truncation at `128 KiB` by default
- used file paths derived from structured transcript fields first, with a command-string fallback
- Codex session grouping via Langfuse `session_id`
- assistant turn-complete output as a child event on stop

The hook process treats Codex's `transcript_path` JSONL as the source of truth. Hook payloads are used mainly to know when to parse and how to associate a turn.

## Setup

1. Ensure Codex hooks are enabled in `~/.codex/config.toml`:

   ```toml
   [features]
   codex_hooks = true
   ```

2. Keep Langfuse credentials in `.env` at the repo root. This project reads:

   - `LANGFUSE_PUBLIC_KEY`
   - `LANGFUSE_SECRET_KEY`
   - `LANGFUSE_BASE_URL` or `LANGFUSE_HOST`

3. Optional tuning:

   - `LANGFUSE_CODEX_STATE_DIR` defaults to `.codex/langfuse-state`
   - `LANGFUSE_CODEX_MAX_PAYLOAD_BYTES` defaults to `131072`
   - `LANGFUSE_CODEX_LOG_LEVEL` defaults to `INFO`

4. The repo-local hook config lives in `.codex/hooks.json` and runs:

   ```bash
   uv run --project "$(git rev-parse --show-toplevel)" codex-langfuse-hook
   ```

## Development

Run the test suite:

```bash
uv run --extra dev pytest
```

Run the live Langfuse integration check:

```bash
LANGFUSE_RUN_INTEGRATION=1 uv run --extra dev pytest tests/test_integration_langfuse.py -q
```

Run the hook manually with a fixture or saved payload:

```bash
uv run codex-langfuse-hook < payload.json
```

The hook writes local logs and state under `.codex/langfuse-state/`.
