# Harness providers

`app.harness()` hands a task to a coding agent — a multi-turn worker that reads,
writes, and edits files, then reports back through the same structured-output
contract as `app.ai()`. AgentField ships its own harness, **AForge**, and it is
the default: a call with no provider set runs `aforge`. Naming a different
provider swaps the worker without changing the surrounding loop, which is how
you orchestrate Claude Code, Codex, Gemini CLI, or OpenCode from a reasoner.

## Default: AForge

The `aforge` binary is provisioned alongside the `af` CLI — the curl installer,
the desktop app, and the published Docker images all ship it. To install or
repair it on demand:

```bash
af aforge ensure
```

Set `OPENROUTER_API_KEY`, then call the harness with nothing else configured:

```python
result = await app.harness("Fix the failing test in tests/test_auth.py", schema=Report)
```

```go
result, err := agent.Harness(ctx, task, schema, &dest, harness.Options{Cwd: repoRoot})
```

```ts
const result = await app.harness(task, { schema });
```

The model defaults to AForge's own default. Set `AFORGE_MODEL` to change it
process-wide, or pass `model=` per call.

Verify the runtime before a paid run:

```bash
af harness doctor --provider aforge
```

## Choosing a different provider

Provider selection follows one precedence chain:

| Order | Source | Example |
| --- | --- | --- |
| 1 | Explicit value on the call or in the agent's harness config | `app.harness(task, provider="codex")` |
| 2 | `AGENTFIELD_HARNESS_PROVIDER` environment variable | `AGENTFIELD_HARNESS_PROVIDER=claude-code` |
| 3 | Default | `aforge` |

Same loop code, different worker:

```python
# AForge — nothing to configure
report = await app.harness(task, schema=Report)

# Orchestrate Claude Code instead
report = await app.harness(task, schema=Report, provider="claude-code")

# ...or Codex, Gemini CLI, OpenCode
report = await app.harness(task, schema=Report, provider="codex")
```

The same override exists in every SDK — `harness.Options{Provider: harness.ProviderCodex}`
in Go, `{ provider: 'codex' }` in TypeScript — and an agent-wide default can be
set once on the agent's harness config (`HarnessConfig(provider="codex")` in
Python, `agent.HarnessConfig{Provider: "codex"}` in Go).

## Install

| Provider | Install | Python extra | Required CLI | Authentication |
| --- | --- | --- | --- | --- |
| `aforge` (default) | `af aforge ensure` (shipped with `af`) | None | `aforge` | `OPENROUTER_API_KEY` |
| `claude-code` | `pip install 'agentfield[harness-claude]'` | `agentfield[harness-claude]` | Bundled by `claude-agent-sdk` | Claude login or `ANTHROPIC_API_KEY` |
| `codex` | `npm install -g @openai/codex` | `agentfield[harness-codex]` | `codex` | Codex login or `OPENAI_API_KEY` |
| `gemini` | `npm install -g @google/gemini-cli` | None | `gemini` | Gemini login, `GEMINI_API_KEY`, or `GOOGLE_API_KEY` |
| `opencode` | `curl -fsSL https://opencode.ai/install \| bash` | `agentfield[harness-opencode]` | `opencode` | Provider credentials configured in OpenCode |
| `grok` | Install the Grok Build CLI, then `grok login` | None | `grok` | `XAI_API_KEY` |

Install every Python wrapper with:

```bash
pip install 'agentfield[harness-all]'
```

`aforge` is the one CLI AgentField distributes itself. Every install surface
provisions it beside `af` in `~/.agentfield/bin` — the curl installer, the
desktop app on launch, and the `python-agent` / `go-agent` / cloud control-plane
images. To install or repair it by hand:

```bash
af aforge ensure          # --force re-downloads even when already current
```

The pinned build, its download host and the opt-out are documented under
`AGENTFIELD_AFORGE_BASE_URL` / `AGENTFIELD_SKIP_AFORGE` in
[docs/ENVIRONMENT_VARIABLES.md](ENVIRONMENT_VARIABLES.md).

The extras install Python wrappers. They do not replace the runtime preflight:
AForge and Gemini are CLI-only, and Codex or OpenCode may still require a
separately available executable depending on the wrapper and platform. `grok`
is available in the Python SDK only.

### AForge adapter contract

AForge is registered as `aforge` in the Python, Go, and TypeScript SDKs. The
adapters default to the direct non-interactive contract, `aforge exec --json`,
send the task over stdin, and map AForge's usage ledger into AgentField turns,
token counts, and cost metrics. Set `AGENTFIELD_AFORGE_COMMAND=do` to opt into
the routed `aforge do --json --yes-spend` workflow instead.

Set `AFORGE_MAX_CONCURRENT` to cap simultaneous AForge subprocesses. The
default is 8. `AGENTFIELD_HARNESS_TIMEOUT_SECONDS` is the outer watchdog; each
adapter gives AForge a five-second landing window to emit its exit-2 timeout
envelope. Schema runs use a unique output directory per invocation so parallel
jobs can safely share a checkout. Set `AFORGE_BIN` to an absolute path when the
binary is installed somewhere off `PATH`.

## Model selection and reasoning-effort variants

Every provider accepts a `model` option on `.harness()` calls. Leaving it unset
uses the provider's own default — AForge picks its own model, `claude-code`
keeps using `sonnet`. The model string may carry a reasoning-effort variant
after a `#` separator:

```python
result = await app.harness(
    task,
    provider="opencode",
    model="openrouter/z-ai/glm-5.2#high",
)
```

An explicit `variant="high"` keyword wins over the suffix. Per provider:

| Provider | Model flag | Variant handling |
| --- | --- | --- |
| `aforge` | `exec`: `--model` and `--plan-model`; `do`: `AFORGE_MODEL` (a leading `openrouter/` is stripped) | `AFORGE_EXEC_REASONING` (`off`, `low`, `medium`, or `high`) |
| OpenCode | `-m <model>` | `--variant <v>` (provider-specific effort, e.g. `high`, `max`, `minimal`) |
| Codex | `-m <model>` | `-c model_reasoning_effort=<v>` |
| Claude Code | SDK `model` option | No effort control — variant is dropped with a debug log |
| Gemini | `-m <model>` | No effort control — variant is dropped |

The `#` separator is safe in model ids: `:` belongs to OpenRouter suffixes like
`:free`, and `@` to Vertex-style ids, but no provider uses `#`.

## Verify

Check selected providers in a container or CI job before any paid run:

```bash
af harness doctor --provider codex,opencode --json
```

The command exits non-zero if a requested provider is missing, its version
cannot be read, or it is otherwise unusable. JSON is still written to stdout so
CI can archive the report when the command fails.

Python applications can use the same preflight data:

```python
reports = await app.harness_doctor(providers=["codex", "opencode"])
for report in reports:
    print(report.provider, report.usable, report.issues)
```

The preflight currently ships in the Python SDK and the `af` CLI. Equivalent
TypeScript and Go SDK APIs are planned follow-ups (see #685) and are not
available yet.

Each report includes the provider name, resolved binary, installed state,
version, auth state, usability, installation command, recognized auth variables,
and machine-readable issues.

The static preflight never performs a paid model request. `auth="configured"`
means a recognized environment variable is present. `auth="unknown"` does not
mean authentication failed: the provider may use a local CLI login that an
offline environment check cannot safely prove. A future explicit liveness probe
can validate provider login without changing the static default.

If a dependency disappears between preflight and execution, providers raise
`HarnessProviderUnavailable` before retrying the task. The exception includes
the provider, missing dependency, and an installation command.
