# pier

Pier is a [Harbor](https://www.harborframework.com/docs/tasks)-compatible framework for evaluating coding agents in sandboxed environments. It reads Harbor's task format and runs trials against it.

```bash
pier run -p path/to/task --agent claude-code --env modal
```

## Why pier

Pier is a fork. We wanted a smaller, more opinionated base to build on. On top of Harbor, Pier adds:

- **Installed agents in air-gapped tasks (`allow_internet = false`).** When the agent runs *inside* the sandbox (Claude Code, Codex, etc.), both the install step and the inference call need the network. Pier lets agents declare their install scripts and a network allowlist, which `docker` and `modal` environments honor when setting up the sandbox.
- **Augmented ATIF v1.7.** Strict one step per API turn, strict reasoning vs agent message separation, no fabricated assistant text, `peak_context_tokens`, `summarization_count`, `llm_call_count`, real upstream timestamps.
- **A chat-style trajectory viewer** (`pier view`).
- **`pier critique run`** for inspecting completed trials with a fresh agent in a fresh sandbox.

## What works today

- **Task format:** Harbor-compatible.
- **Environments:** `docker`, `modal`. Per-agent install specs and network allowlists are honored on both, so installed agents work under `allow_internet = false`.
- **Agents:** `nop`, `oracle`, `claude-code`, `codex`, `gemini-cli`, `opencode`, `mini-swe-agent`. All emit augmented ATIF v1.7.
- **Datasets:** local Harbor-format task directories via `-p` / `--path`.
- **CLI:** `pier run`, `pier job`, `pier view`, `pier critique run`, `pier check` / `pier analyze` (vendored from Harbor)

Pier does not currently resolve or download Harbor registry datasets directly.

## Install

```bash
uv tool install datacurve-pier
# or
pip install datacurve-pier
```

## Run

```bash
export ANTHROPIC_API_KEY=...
pier run -p path/to/task --agent claude-code --env modal --env-file .env
```

Run a local dataset, optionally a deterministic random subset:

```bash
pier run -p path/to/dataset --agent claude-code --env modal
pier run -p path/to/dataset --n-tasks 10 --sample-seed 0
```

To use a Harbor registry dataset, download it with Harbor first, then point Pier at it:

```bash
uv run --directory ~/code/harbor harbor download swebenchpro -o ~/code/pier/datasets
uv run pier run -p datasets/swebenchpro --n-tasks 10 --sample-seed 0
```

Trials land under `jobs/<timestamp_or_name>/<trial_id>/`. See `pier run --help`, `pier job --help`, `pier critique --help`, and `pier view --help` for everything else.

## Job configs

`pier run` and `pier critique run` can load a job config with `--config`:

```bash
export RESPAN_API_KEY=...
uv run pier run --config configs/respan-dual-agent-local-task.yaml --yes
```

Config files can live anywhere you can pass to `--config`. Supported extensions are `.yaml` and `.json`; `.yml` is not supported today.

A job config implements `pier.models.job.config.JobConfig`. The common top-level shape is:

```yaml
job_name: my-job
jobs_dir: jobs
n_attempts: 1
n_concurrent_trials: 1
environment:
  type: docker        # or modal
agents:
  - name: claude-code
    model_name: claude-opus-4-7
    env:
      ANTHROPIC_AUTH_TOKEN: ${RESPAN_API_KEY}
    kwargs:
      reasoning_effort: max
tasks:
  - path: path/to/one-task
# Or, for a directory containing many task directories:
# datasets:
#   - path: path/to/local-dataset
#     n_tasks: 10
#     sample_seed: 0
```

For local Harbor directories, `tasks:` entries point at individual task directories containing `task.toml`, `instruction.md`, `environment/`, and `tests/`. `datasets:` entries point at directories of task directories; dataset configs can filter with `task_names` / `exclude_task_names` and sample with `n_tasks` / `sample_seed`.

Runtime secrets should not be committed. Put `${VAR_NAME}` placeholders in config files, then provide the real values in your shell or with `--env-file`. Pier resolves those placeholders when constructing the agent or environment, and serializes known sensitive env values back as templates so job artifacts do not leak real keys.

Agent entries use `model_name` for trial metadata, `env` for runtime environment variables, and agent-specific `kwargs` for tool configuration.

See `configs/README.md` and `configs/respan-dual-agent-local-task.yaml` for a runnable Respan-backed Claude Code + Codex config example.
