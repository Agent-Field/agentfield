# Harness providers

`app.harness()` hands a task to a coding agent — a multi-turn worker that reads,
writes, and edits files, then reports back through the same structured-output
contract as `app.ai()`. AgentField ships its own harness, **AForge**, and it is
the default: a call with no provider set runs `aforge`. Naming a different
provider swaps the worker without changing the surrounding loop, which is how
you orchestrate Claude Code, Codex, Gemini CLI, OpenCode, Pi, or OMP from a
reasoner.

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

# ...or Codex, Gemini CLI, OpenCode, Pi, OMP
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
| `pi` | `npm install -g --ignore-scripts @earendil-works/pi-coding-agent` | None | `pi` | Provider login or API key such as `OPENROUTER_API_KEY` |
| `omp` | `curl -fsSL https://omp.sh/install \| sh` | None | `omp` | Provider login or API key such as `OPENROUTER_API_KEY` |

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
is available in the Python SDK only. Pi and OMP are CLI-only: install their
upstream binaries as shown below.

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

### OpenCode concurrency

The OpenCode adapters cap how many `opencode run` subprocesses may be in flight
at once, so a wide fan-out does not overwhelm the upstream provider with
parallel requests. Set `OPENCODE_MAX_CONCURRENT` to a positive integer to change
the cap. The defaults differ per SDK:

| SDK | Default | Source |
| --- | --- | --- |
| Go | 4 | `sdk/go/harness/opencode.go` |
| Python | 10 | `sdk/python/agentfield/harness/providers/opencode.py` |

The value is read once per process — Go reads it the first time the limiter is
used, Python reads it at import time — so export it before starting the agent
rather than mutating the environment mid-run. The TypeScript OpenCode provider
has no limiter and ignores the variable.

Install Pi or OMP directly from their official distributions:

```bash
npm install -g --ignore-scripts @earendil-works/pi-coding-agent
curl -fsSL https://omp.sh/install | sh
```

For reproducible containers and CI, run the chosen upstream installer in the
image build, then gate startup with `af harness doctor`. If the executable is
missing at dispatch time, all three SDKs return an actionable provider error
containing the same upstream install command instead of attempting a mutation.

## Provider parity

Pi and OMP implement the same provider-neutral harness surface as OpenCode.
The SDK translates that surface to each CLI's native flags rather than exposing
CLI-specific command construction to application code.

| Capability | OpenCode | Pi | OMP |
| --- | --- | --- | --- |
| Model and `#variant` | `-m`, `--variant` | `--model`, `--thinking` | `--model`, `--thinking` |
| Project root | `--dir` | process working directory | `--cwd` plus process working directory |
| One-shot machine output | JSON output | stdin + JSON event stream | stdin + JSON event stream |
| System prompt | Native prompt option | Native prompt option | Native prompt option |
| Tool allowlist | Ignored today | Normalized Pi tool names | Normalized OMP tool names |
| Plan / auto permissions | Ignored today | Read-only tools / no approval flag | Read-only tools / `--auto-approve` |
| Session resume | Native session option | `--session` | `--resume` |
| Structured output | Isolated schema file protocol | Same protocol | Same protocol |
| Metrics | Sessions, turns, tokens, cost, duration | Same normalized fields | Same normalized fields |
| Runtime controls | Env, timeout, retries, binary override | Same | Same |

The contract is equivalent, not flag-identical. Pi calls its filesystem search
tool `find`, OMP calls it `glob`, and each CLI has its own resume flag. Only OMP
has an approval flag: `--tools` is Pi's documented read-only mechanism and it has
no approval flag at all, so `permission_mode="auto"` adds nothing for Pi. These
differences stay inside the provider adapters. Unsupported native concepts are
handled consistently: plan mode removes mutating tools, explicit model variants
override `#variant`, and provider-reported metrics are normalized into the shared
result type.

OpenCode currently receives only the selected model, project directory, and
prompt. Its adapters ignore `tools` and `permission_mode`; they do not translate
either option to native OpenCode flags today.

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

Pi and OMP accept the same OpenRouter model strings in every SDK, for example
`openrouter/minimax/minimax-m2.7` or
`openrouter/google/gemini-2.5-flash#low`.

| Provider | Model flag | Variant handling |
| --- | --- | --- |
| `aforge` | `exec`: `--model` and `--plan-model`; `do`: `AFORGE_MODEL` (a leading `openrouter/` is stripped) | `AFORGE_EXEC_REASONING` (`off`, `low`, `medium`, or `high`) |
| OpenCode | `-m <model>` | `--variant <v>` (provider-specific effort, e.g. `high`, `max`, `minimal`) |
| Codex | `-m <model>` | `-c model_reasoning_effort=<v>` |
| Claude Code | SDK `model` option | No effort control — variant is dropped with a debug log |
| Gemini | `-m <model>` | No effort control — variant is dropped |
| Pi | `--model <model>` | `--thinking <v>` |
| OMP | `--model <model>` | `--thinking <v>` |

The `#` separator is safe in model ids: `:` belongs to OpenRouter suffixes like
`:free`, and `@` to Vertex-style ids, but no provider uses `#`.

## Verify

Check selected providers in a container or CI job before any paid run:

```bash
af harness doctor --provider codex,opencode,pi,omp --json
```

The command exits non-zero if a requested provider is missing, its version
cannot be read, or it is otherwise unusable. JSON is still written to stdout so
CI can archive the report when the command fails.

Python applications can use the same preflight data:

```python
reports = await app.harness_doctor(providers=["codex", "opencode", "pi", "omp"])
for report in reports:
    print(report.provider, report.usable, report.issues)
```

The preflight currently ships in the Python SDK and the `af` CLI. Equivalent
TypeScript and Go SDK APIs are planned follow-ups (see #685) and are not
available yet.

For a complete Go workflow that fans one task out to Pi and OMP concurrently,
see `examples/go_agent_nodes/cmd/harness_duo`.

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
