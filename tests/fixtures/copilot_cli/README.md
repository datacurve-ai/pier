# Copilot CLI event fixtures

Sanitized excerpts of real `~/.copilot/session-state/*/events.jsonl` streams,
used to test the ATIF conversion in `pier.agents.installed.copilot_cli` against
event shapes the CLI actually emits.

Prompts, code, paths, repository names, URLs and token-like values are replaced
with deterministic synthetic data; event types, ordering, structure and token
counts are preserved. Regenerate them with
`uv run python scripts/sanitize_copilot_fixtures.py`.

| Fixture | Covers |
| --- | --- |
| `session_basic.jsonl` | A complete run: user turn, tool calls, shutdown metrics |
| `session_subagents.jsonl` | Delegated `task` runs tagged with `agentId` |
| `session_compaction.jsonl` | `session.resume` plus two *failed* compactions |
| `session_timeout.jsonl` | A stream killed mid-write, ending in a truncated line |
| `session_model_change.jsonl` | `session.model_change` and AIU accounting |
| `session_system_events.jsonl` | `system.message` steps and an `abort` |
