# Pier job configs

This directory contains reusable example job configs for `pier run --config ...`. Keep paths portable and never commit real API keys.

Run the Respan dual-agent example from the repository root:

```bash
export RESPAN_API_KEY='...'
uv run pier run --config configs/respan-dual-agent-local-task.yaml --yes
```

The config targets `examples/tasks/hello-world`, a single Harbor-format task included in this repository. A `tasks:` entry points at a directory containing `task.toml`, `instruction.md`, `environment/`, and `tests/`.

A `datasets:` entry points at a directory of multiple Harbor task directories. Dataset entries can filter and sample tasks, for example:

```yaml
datasets:
  - path: path/to/local-dataset
    task_names: ["fix-*", "debug-*A"]
    n_tasks: 10
    sample_seed: 0
```

Notes:
- `--config` currently accepts `.yaml` and `.json`; `.yml` is not supported.
- `${RESPAN_API_KEY}` is resolved from the process environment or `--env-file` at runtime.
- Agent `env` values are serialized with secret templating so generated job configs do not expose known sensitive values.
