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
- **Agents:** `nop`, `oracle`, `claude-code`, `codex`, `cursor-cli`, `gemini-cli`, `opencode`, `mini-swe-agent`. All emit augmented ATIF v1.7.
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

## Agent runtime configuration

Use `agent.model_name` for trial metadata, `agent.env` for runtime env vars, and agent-specific `kwargs` for tool config. Pier's network allowlist also reads URLs out of those configs (Codex `config_toml`, OpenCode `opencode_config`, mini-swe `config_yaml`), so any base URL you set is allowlisted without code changes.

A few things we've learned plumbing this through Respan and OpenRouter:

**Claude Code** routes through the Anthropic face from Respan. Plan mode is disabled by default (`--disallowedTools EnterPlanMode`).

```yaml
- name: claude-code
  model_name: claude-opus-4-7
  env:
    ANTHROPIC_AUTH_TOKEN: ${RESPAN_API_KEY}
    ANTHROPIC_BASE_URL: https://endpoint.respan.ai/api/anthropic
    ANTHROPIC_CUSTOM_HEADERS: "X-Respan-Route-Provider: vertex_ai"
  kwargs:
    reasoning_effort: max
```

**Codex** needs a `[model_providers.<name>]` block with `wire_api = "responses"` (not WebSockets, which Codex defaults to and Respan doesn't speak).

```yaml
- name: codex
  model_name: openai/gpt-5.5
  env: { RESPAN_API_KEY: ${RESPAN_API_KEY} }
  kwargs:
    config_toml: |
      model_provider = "respan"
      [model_providers.respan]
      name = "Respan Gateway"
      base_url = "https://endpoint.respan.ai/api/"
      wire_api = "responses"
      env_key = "RESPAN_API_KEY"
    reasoning_effort: xhigh
```

**Gemini CLI**:

```yaml
- name: gemini-cli
  model_name: gemini/gemini-3.1-pro-preview
  env:
    GEMINI_API_KEY: ${RESPAN_API_KEY}
    GOOGLE_GENERATIVE_AI_API_KEY: ${RESPAN_API_KEY}
    GEMINI_API_BASE: https://endpoint.respan.ai/api/google/vertexai/v1beta
    GOOGLE_GEMINI_BASE_URL: https://endpoint.respan.ai/api/google/vertexai/
```

**Cursor CLI** uses the installed `cursor-agent` binary, so it fits the same
inside-the-sandbox path as Claude Code, Codex, Gemini CLI, and OpenCode. Use
`cursor/composer-2.5` for Composer 2.5 trial metadata and pass `CURSOR_API_KEY`
through your env file.

```yaml
- name: cursor-cli
  model_name: cursor/composer-2.5
  env:
    CURSOR_API_KEY: ${CURSOR_API_KEY}
```

**OpenCode** uses `opencode_config` to add unknown providers or override known ones. To redirect Google to Respan, override just `options.baseURL`; to add a fully custom provider, use `opencode_config.provider.<name>` with the npm package, options, and models.

**mini-swe-agent** picks a native adapter from the model-name prefix: `openai/...` → `litellm_response` (OpenAI Responses end-to-end), `openrouter/...` → `openrouter` (BYOK costs from `cost_details.upstream_inference_cost`), everything else → LiteLLM auto.

For Gemini 3 via mini-swe-agent/LiteLLM, omitting `reasoning_effort` uses the Gemini API default high/dynamic thinking level, but it does not request readable thought summaries. Set `kwargs.reasoning_effort: high` explicitly when you want LiteLLM to send `includeThoughts` and preserve returned summaries as reasoning content.

```yaml
- name: mini-swe-agent
  model_name: openrouter/qwen/qwen3.6-plus
  env: { OPENROUTER_API_KEY: ${OPENROUTER_API_KEY} }
  kwargs:
    set_cache_control: default_end
```

**MiniMax Coding Plan** is supported on `mini-swe-agent`, `opencode`, `claude-code`, and `codex`. The canonical LiteLLM provider prefix is `minimax/` and the auth env var is `MINIMAX_API_KEY`. Coding Plan subscription keys must use the Anthropic-compatible endpoint `https://api.minimax.io/anthropic`; the OpenAI-compatible path returns 404 for those keys. Use `MINIMAX_BASE_URL` or `ANTHROPIC_BASE_URL` to override per agent (China region: `https://api.minimaxi.com/anthropic`).

```yaml
- name: mini-swe-agent
  model_name: minimax/MiniMax-M3
  env: { MINIMAX_API_KEY: ${MINIMAX_API_KEY} }
- name: opencode
  model_name: minimax/MiniMax-M3
  env: { MINIMAX_API_KEY: ${MINIMAX_API_KEY} }
  # Optional override for China region or custom endpoint:
  # env: { MINIMAX_API_KEY: ${MINIMAX_API_KEY}, MINIMAX_BASE_URL: ${MINIMAX_BASE_URL} }
- name: claude-code
  model_name: minimax/MiniMax-M3
  env:
    ANTHROPIC_BASE_URL: https://api.minimax.io/anthropic
    ANTHROPIC_AUTH_TOKEN: ${MINIMAX_API_KEY}
- name: codex
  model_name: minimax/MiniMax-M3
  env: { OPENAI_BASE_URL: https://api.minimax.io/v1, OPENAI_API_KEY: ${MINIMAX_API_KEY} }
  # Coding Plan subscription keys require `wire_api = "responses"` in config_toml:
  # kwargs:
  #   config_toml: |
  #     [model_providers.minimax_coding_plan]
  #     name = "MiniMax Coding Plan"
  #     base_url = "https://api.minimax.io/anthropic"
  #     wire_api = "responses"
  #     env_key = "MINIMAX_API_KEY"
```

`gemini-cli` and `cursor-cli` are not supported — their SDKs do not accept a foreign provider URL.
