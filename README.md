# langfuse-codex

This is obviously the most forward-thinking project that combines two revolutionary technologies, langfuse and codex! It is the only logical choice for 1st place.

Codex hook handler that turns Codex transcript events into Langfuse traces.
Repo-local hooks use a durable local queue so Codex is not blocked on Langfuse network flushes.

## What it captures

- one Langfuse trace per Codex turn
- raw tool inputs and outputs with truncation at `128 KiB` by default
- pre-run Bash command intent from `PreToolUse`, including `permission_mode`
- used file paths derived from structured transcript fields first, with a command-string fallback
- Codex session grouping via Langfuse `session_id`
- Codex session metadata including CLI version, originator, model provider, transcript path, thread name, and session start source
- turn metadata for token usage, rate limits, collaboration mode, task timing, and user-message attachment fields
- assistant turn-complete output as a child event on stop

The hook process treats Codex's `transcript_path` JSONL as the source of truth. Hook payloads are also used for schema fields that may not be present in transcript events, such as `permission_mode`, `PreToolUse.tool_input`, and `Stop.stop_hook_active`.

## Queue mode

By default, `.codex/hooks.json` runs a stdlib-only enqueue script that writes hook payloads under `.codex/langfuse-state/queue/inbox/` and starts `codex-langfuse-sender` in the background. The sender drains the queue, reuses the normal hook processor, and performs Langfuse flushing outside Codex's hook path.

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
   - `LANGFUSE_CODEX_DELIVERY_MODE` defaults to `queued`; set `direct` to run the processor synchronously
   - `LANGFUSE_CODEX_SENDER_IDLE_SECONDS` defaults to `30`
   - `LANGFUSE_CODEX_QUEUE_MAX_EVENTS` defaults to `10000`

4. The repo-local hook config lives in `.codex/hooks.json` and runs:

   ```bash
   python3 "$(git rev-parse --show-toplevel)/.codex/hooks/langfuse_enqueue.py"
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

Drain queued payloads once:

```bash
uv run codex-langfuse-sender --once
```

The hook writes local logs and state under `.codex/langfuse-state/`.
