# langfuse-codex

This is obviously the most forward-thinking project that combines two revolutionary technologies, langfuse and codex! It is the only logical choice for 1st place.

Codex hook handler that turns Codex transcript events into Langfuse traces.
Hooks use a durable local queue so Codex is not blocked on Langfuse network flushes.

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

## Global service mode

The recommended setup is a once-installed per-user macOS `launchd` service. Project hooks run `codex-langfuse-enqueue`, a stdlib-only command that writes hook payloads to a global queue under `~/Library/Application Support/langfuse-codex/queue/`. The `codex-langfuse-service` daemon drains that queue in the background, routes each event back to its project, and writes per-project state under `~/Library/Application Support/langfuse-codex/state/`.

This means another project only needs a `.codex/hooks.json` that calls `codex-langfuse-enqueue`.

### macOS launchd

Install or update the launch agent:

```bash
codex-langfuse-launchd install
```

Useful service commands:

```bash
codex-langfuse-launchd status
codex-langfuse-launchd restart
codex-langfuse-launchd stop
codex-langfuse-launchd start
codex-langfuse-launchd uninstall
```

The launch agent is written to `~/Library/LaunchAgents/com.langfuse.codex.plist` and logs to `~/Library/Logs/langfuse-codex/service.log`.

## Setup

1. Ensure Codex hooks are enabled in `~/.codex/config.toml`:

   ```toml
   [features]
   codex_hooks = true
   ```

2. Keep Langfuse credentials in the global config file at `~/Library/Application Support/langfuse-codex/env`:

   ```bash
   mkdir -p "$HOME/Library/Application Support/langfuse-codex"
   cat > "$HOME/Library/Application Support/langfuse-codex/env" <<'EOF'
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_BASE_URL=https://cloud.langfuse.com
   EOF
   ```

   A project `.env` can override the global values. The project-level direct/dev path also still reads `.env` at the repo root. This project reads:

   - `LANGFUSE_PUBLIC_KEY`
   - `LANGFUSE_SECRET_KEY`
   - `LANGFUSE_BASE_URL` or `LANGFUSE_HOST`

3. Optional tuning:

   - `LANGFUSE_CODEX_CONFIG` defaults to `~/Library/Application Support/langfuse-codex/env`
   - `LANGFUSE_CODEX_QUEUE_DIR` defaults to `~/Library/Application Support/langfuse-codex/queue`
   - `LANGFUSE_CODEX_STATE_ROOT` defaults to `~/Library/Application Support/langfuse-codex/state`
   - `LANGFUSE_CODEX_STATE_DIR` defaults to `.codex/langfuse-state` for direct/repo-local mode and a project-hashed directory under `LANGFUSE_CODEX_STATE_ROOT` for service mode
   - `LANGFUSE_CODEX_MAX_PAYLOAD_BYTES` defaults to `131072`
   - `LANGFUSE_CODEX_LOG_LEVEL` defaults to `INFO`
   - `LANGFUSE_CODEX_DELIVERY_MODE` defaults to `queued`; set `direct` to run the processor synchronously
   - `LANGFUSE_CODEX_SENDER_IDLE_SECONDS` defaults to `30`
   - `LANGFUSE_CODEX_QUEUE_MAX_EVENTS` defaults to `10000`

4. The portable hook config lives in `.codex/hooks.json` and runs:

   ```bash
   codex-langfuse-enqueue
   ```

## Repo-local queue mode

The legacy `.codex/hooks/langfuse_enqueue.py` script remains available for development or backward compatibility. It writes hook payloads under `.codex/langfuse-state/queue/inbox/` and can start `codex-langfuse-sender` opportunistically. The global service path disables the need for per-repo sender autostart because `launchd` owns the background process.

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

Drain the global service queue once:

```bash
uv run codex-langfuse-service --once
```

In repo-local mode, the hook writes logs and state under `.codex/langfuse-state/`. In global service mode, logs and state live under `~/Library/Application Support/langfuse-codex/` and `~/Library/Logs/langfuse-codex/`.
