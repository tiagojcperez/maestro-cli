# Model & Reasoning Reference

Complete model alias tables for all supported engines.

> Back to [README](../README.md)

---

## Claude

| Alias | Best For | Cost |
|-------|----------|------|
| `haiku` | Simple tasks, quick checks (Haiku 4.5) | $ |
| `sonnet` | Daily coding, implementation (Sonnet 4.6) | $$ |
| `opus` | Complex reasoning, architecture, long-horizon agentic (resolves to Opus 4.8 since 2026-06) | $$$ |
| `opusplan` | Plan with Opus, execute with Sonnet | $$-$$$ |

**Pricing per million tokens** (Anthropic API, Apr 2026):

| Model | Input | Output |
|-------|-------|--------|
| Haiku 4.5 | $1.00 | $5.00 |
| Sonnet 4.6 | $3.00 | $15.00 |
| Opus 4.8 (and 4.7 / 4.6) | $5.00 | $25.00 |

**Reasoning effort** (Opus 4.6 / 4.7 / 4.8 / Sonnet 4.6; ignored on Haiku):
`low`, `medium`, `high` (default), `xhigh` *(Opus 4.7 / 4.8)*, `max` *(Opus 4.6 / 4.7 / 4.8 / Sonnet 4.6)*.

Per-model defaults: Opus 4.8 = `high`, Opus 4.7 = `xhigh`, Sonnet 4.6 = `high`; Haiku
ignores effort. For Opus 4.8, `xhigh` is recommended for hard coding / agentic work,
with `max` reserved for genuinely frontier problems. Adaptive
thinking is the only thinking-on mode; extended thinking with explicit
`budget_tokens` is rejected. Sampling parameters (`temperature`, `top_p`,
`top_k`) at non-default values return 400.

Mechanism: `CLAUDE_CODE_EFFORT_LEVEL=<level>` env var (injected automatically by Maestro).

---

## Codex

| Alias | Full Name | Cost |
|-------|-----------|------|
| `5.5` | gpt-5.5 (latest, since 2026-04-23) | $$$$ |
| `5.4` | gpt-5.4-codex *(alias kept for back-compat; canonical OpenAI ID is now `gpt-5.4`)* | $$$ |
| `5.4-mini` | gpt-5.4-mini | $ |
| `5.3` | gpt-5.3-codex | $$ |
| `5.2` | gpt-5.2-codex | $$ |
| `5.1` | gpt-5.1-codex | $$ |
| `5` | gpt-5-codex | $$ |
| `5-mini` | gpt-5-codex-mini | $ |

> **Naming change**: Starting with `gpt-5.4`, OpenAI dropped the `-codex`
> suffix from canonical model IDs — the standard model now powers Codex CLI
> workloads directly. The `5.5` and `5.4-mini` aliases target the unsuffixed
> names; the older `5.4` / `5.3` / `5.2` / `5.1` / `5` aliases keep the
> historical `-codex` suffix for backward compatibility.

**Pricing per million tokens** (OpenAI API, Apr 2026):

| Model | Input | Cached Input | Output |
|-------|-------|--------------|--------|
| `gpt-5.5` | $5.00 | $0.50 | $30.00 |
| `gpt-5.4` (and `gpt-5.4-codex` via alias) | $2.50 | $0.25 | $15.00 |
| `gpt-5.4-mini` | $0.75 | $0.075 | $4.50 |
| `gpt-5.3-codex` | $1.75 | $0.175 | $14.00 |
| `default` (fallback) | $2.00 | $0.50 | $8.00 |

> Prompts > 272K input tokens are billed at 2× input / 1.5× output by OpenAI
> for the full session — applies to standard, batch, and flex tiers. Override
> via `MAESTRO_CODEX_PRICING_JSON` if your workloads regularly exceed that.

**Reasoning effort** (all models): `none` *(GPT-5.5+)*, `minimal`, `low`, `medium`, `high`, `xhigh`.

Mechanism: `-c model_reasoning_effort=<level>` config flag (injected automatically by Maestro).

---

## Gemini

| Alias | Full Name | Cost |
|-------|-----------|------|
| `flash` | gemini-2.5-flash | $ |
| `flash-lite` | gemini-2.5-flash-lite | $ |
| `pro` | gemini-2.5-pro | $$ |
| `flash-3` | gemini-3-flash-preview | $$ |
| `pro-3` | gemini-3.1-pro-preview *(3-pro-preview retired)* | $$$ |
| `pro-3.1` | gemini-3.1-pro-preview | $$$ |
| `auto` | (system routes) | varies |

Gemini does not expose reasoning effort control. Use model selection (Pro vs Flash) instead.

Mechanism: model set via `-m <model>` flag (injected automatically by Maestro).

---

## Copilot (GitHub Copilot CLI)

Access multiple model families via a single GitHub Copilot subscription (premium requests).

### Claude models

| Alias | Full Name |
|-------|-----------|
| `opus` | Claude Opus 4.6 |
| `opus-fast` | Claude Opus 4.6 (fast mode) |
| `opus-4.5` | Claude Opus 4.5 |
| `sonnet` | Claude Sonnet 4.6 |
| `sonnet-4.5` | Claude Sonnet 4.5 |
| `sonnet-4` | Claude Sonnet 4 |
| `haiku` | Claude Haiku 4.5 |

### GPT models

| Alias | Full Name |
|-------|-----------|
| `gpt-5.4-codex` | GPT-5.4-Codex |
| `gpt-5.3-codex` | GPT-5.3-Codex |
| `gpt-5.2-codex` | GPT-5.2-Codex |
| `gpt-5.1-codex` | GPT-5.1-Codex |
| `gpt-5.1-codex-mini` | GPT-5.1-Codex-Mini |
| `gpt-5.1-codex-max` | GPT-5.1-Codex-Max |
| `gpt-5.2` | GPT-5.2 |
| `gpt-5.1` | GPT-5.1 |
| `gpt-5-mini` | GPT-5 mini |
| `gpt-4.1` | GPT-4.1 |

### Gemini & other models

| Alias | Full Name |
|-------|-----------|
| `gemini-pro` | Gemini 2.5 Pro |
| `gemini-3-pro` | Gemini 3 Pro Preview |

(`grok`/`grok-code-fast-1` was removed — retired from Copilot on 2026-05-15.)

Copilot does not expose reasoning effort control. Use model routing for capability tiers. Cost is subscription-based (premium requests), not per-token.

Mechanism: `copilot --autopilot --silent --no-color --model <model> -p <prompt>`.

Environment variables: `COPILOT_GITHUB_TOKEN`, `GH_TOKEN`, `GITHUB_TOKEN` (auth), `COPILOT_MODEL` (default model).

---

## Qwen

| Alias | Full Name | Cost |
|-------|-----------|------|
| `coder` | qwen-coder-plus | $$ |
| `coder-turbo` | qwen-coder-turbo | $ |
| `max` | qwen-max | $$$ |
| `plus` | qwen-plus | $$ |
| `qwq` | qwq-plus | $$ |

Qwen does not expose reasoning effort control. Use model selection instead.

Mechanism: `qwen --model <model> --prompt "<prompt>"`.

Environment variable: `DASHSCOPE_API_KEY` (auth).

---

## Ollama (local models)

| Alias | Full Name | Cost |
|-------|-----------|------|
| `llama3` | llama3 | Free |
| `llama3.1` | llama3.1 | Free |
| `llama3.2` | llama3.2 | Free |
| `codellama` | codellama | Free |
| `mistral` | mistral | Free |
| `mixtral` | mixtral | Free |
| `phi3` | phi3 | Free |
| `qwen2` | qwen2 | Free |
| `qwen2.5-coder` | qwen2.5-coder | Free |
| `deepseek-coder` | deepseek-coder | Free |
| `deepseek-coder-v2` | deepseek-coder-v2 | Free |
| `starcoder2` | starcoder2 | Free |
| `llama4` | llama4 | Free |
| `qwen3` | qwen3 | Free |
| `qwen3-coder` | qwen3-coder | Free |
| `deepseek-r1` | deepseek-r1 | Free |
| `deepseek-v3` | deepseek-v3 | Free |
| `gemma3` | gemma3 | Free |
| `phi4` | phi4 | Free |
| `gpt-oss` | gpt-oss | Free |

All Ollama models run locally -- zero API cost. Unknown model names are passed through as-is (supports any model available via `ollama pull`).

Requires `ollama` CLI on PATH with models pulled. Set `OLLAMA_HOST` to override the default server address (`http://localhost:11434`).

---

## Llama (llama.cpp / llama-cli)

| Alias | Full Name | Cost |
|-------|-----------|------|
| `llama3` | llama-3-8b | Free |
| `llama3.1` | llama-3.1-8b | Free |
| `llama3.2` | llama-3.2-3b | Free |
| `codellama` | codellama-13b | Free |
| `phi3` | phi-3-mini | Free |
| `mistral` | mistral-7b | Free |
| `qwen2.5-coder` | qwen2.5-coder-7b | Free |
| `llama4-scout` | llama-4-scout-17b-16e | Free |
| `llama4-maverick` | llama-4-maverick-17b-128e | Free |

All Llama models run locally via llama.cpp -- zero API cost. Unknown model names are passed through as-is.

Llama does not expose reasoning effort control. Use model selection instead.

Requires `llama-cli` on PATH. Set `LLAMA_MODEL_DIR` to the directory containing model files; relative model names are resolved against it.

Mechanism: `llama-cli -m <model> -p "<prompt>" --no-display-prompt`.

Environment variable: `LLAMA_MODEL_DIR` (directory containing model files).

---

## Automatic Model Routing

Set `model: auto` on any engine task to let Maestro select the best model based on task complexity.

Complexity signals:
- **Tags**: `security`/`architecture`/`critical`/`audit` → high tier; `trivial`/`typo`/`config`/`docs` → low tier
- **Prompt length**: longer prompts suggest higher complexity
- **Dependencies**: more dependencies increase complexity score
- **Context mode**: `recursive`/`map_reduce` boost complexity
- **Judge presence**: tasks with judges score higher

### Routing tiers per engine

| Engine | Low | Medium | High |
|--------|-----|--------|------|
| Claude | haiku | sonnet | opus *(Opus 4.8 since 2026-06)* |
| Codex | 5-mini | 5.4 | 5.5 *(bumped from 5.4 on 2026-04-27)* |
| Gemini | flash-lite | flash | pro |
| Copilot | haiku | sonnet | opus |
| Qwen | coder-turbo | coder | max |
| Ollama | phi3 | llama3 | mixtral |
| Llama | llama-3.2-3b | llama-3-8b | codellama-13b |

### Routing strategies

Set `routing_strategy:` at plan level to bias the routing:

| Strategy | Effect |
|----------|--------|
| `cost_optimized` | Pushes towards cheaper models |
| `quality_first` | Pushes towards more capable models |
| `balanced` | Default -- no bias |

DAG structural signals (fan-out > 3, depth > 4, upstream failure rate > 0.3) also influence routing decisions.

---

## Pricing Table Overrides

Per-engine pricing tables can be overridden via environment variables:

| Env Var | Engine |
|---------|--------|
| `MAESTRO_CODEX_PRICING_JSON` | Codex |
| `MAESTRO_CLAUDE_PRICING_JSON` | Claude |
| `MAESTRO_GEMINI_PRICING_JSON` | Gemini |
| `MAESTRO_QWEN_PRICING_JSON` | Qwen |

Override format: `'{"model":{"input_per_million":X,"cached_input_per_million":Y,"output_per_million":Z}}'`

Copilot uses subscription-based pricing (no per-token cost). Ollama is zero-cost (local).

---

## Judge Presets

### Named Presets

Available via `judge.preset`:

| Preset | Focus | Threshold | Aggregation |
|--------|-------|-----------|-------------|
| `code_quality` | Correctness, code style, error handling | 0.6 | weighted_mean |
| `security_audit` | Input validation, authentication, data protection | 0.7 | min |
| `ai_slop_detection` | Filler preamble, hedging, repetition, specificity, trailing summary | 0.6 | weighted_mean |

### CWE Security Profiles

4 judge presets mapped to CWE vulnerability categories, available via `judge.preset`:

| Preset | CWE Coverage | Criteria | Threshold | Aggregation |
|--------|-------------|----------|-----------|-------------|
| `cwe_injection` | CWE-89 (SQL), CWE-78 (Command), CWE-79 (XSS), CWE-22 (Path Traversal) | 4 rubrics | 0.8 | min |
| `cwe_auth` | CWE-287 (Auth), CWE-284 (Access Control), CWE-256 (Credentials), CWE-384 (Sessions) | 4 rubrics | 0.8 | min |
| `cwe_data_exposure` | CWE-200 (Data Exposure), CWE-327 (Crypto), CWE-209 (Error Leakage) | 3 rubrics | 0.7 | min |
| `cwe_top_25` | Injection + Access Control + Data + Resources + Config (OWASP Top 25) | 5 rubrics | 0.75 | min |

All CWE presets use `aggregation: min` -- every criterion must individually pass. Use these for targeted vulnerability scanning instead of the general `security_audit` preset.

Example:

```yaml
judge:
  preset: cwe_injection
  on_fail: retry
```
