# Harness providers

AgentField harness providers run external coding agents behind one SDK contract.
OMP is the default in Python, TypeScript, and Go; an explicit provider always
wins. Install the provider wrapper you need, install its CLI when required, and
verify the runtime before starting a workflow.

## Default selection

Provider resolution is identical in every SDK:

1. The provider passed to the individual harness call.
2. The provider in the agent's harness configuration.
3. OMP.

Model resolution follows the same first two layers, then defers to the selected
CLI's configured default. AgentField does not silently force a Claude model onto
OMP or another provider.

```python
# Python: provider omitted, so this runs OMP.
result = await app.harness("Fix the failing test")
```

```typescript
// TypeScript: provider omitted, so this runs OMP.
const result = await app.harness('Fix the failing test');
```

```go
// Go: the zero-value Options use OMP.
result, err := app.Harness(ctx, "Fix the failing test", nil, nil, harness.Options{})
```

## Install

| Provider | Python extra | Required CLI | Authentication |
| --- | --- | --- | --- |
| `aforge` | None | `aforge` | `OPENROUTER_API_KEY` |
| Claude Code | `agentfield[harness-claude]` | Bundled by `claude-agent-sdk` | Claude login or `ANTHROPIC_API_KEY` |
| Codex | `agentfield[harness-codex]` | `codex` | Codex login or `OPENAI_API_KEY` |
| Gemini | None | `gemini` | Gemini login, `GEMINI_API_KEY`, or `GOOGLE_API_KEY` |
| OpenCode | `agentfield[harness-opencode]` | `opencode` | Provider credentials configured in OpenCode |
| Pi | None | `pi` | Provider login or API key such as `OPENROUTER_API_KEY` |
| OMP (Oh My Pi, default) | None | `omp` | Provider login or API key such as `OPENROUTER_API_KEY` |

Install every Python wrapper with:

```bash
pip install 'agentfield[harness-all]'
```

The extras install Python wrappers. AgentField does not install or upgrade a
coding-agent executable at application startup. This is the same lifecycle used
for Codex, Gemini, and OpenCode: operators choose and pin the CLI version, while
the SDK locates it on `PATH` (or through a provider-specific binary override).
Aforge and Gemini are CLI-only, and Codex or OpenCode may still require a
separately available executable depending on the wrapper and platform.

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
| Tool allowlist | Native tool flags | Normalized Pi tool names | Normalized OMP tool names |
| Plan / auto permissions | Native permission flags | Read-only tools / `--approve` | Read-only tools / `--auto-approve` |
| Session resume | Native session option | `--session` | `--resume` |
| Structured output | Isolated schema file protocol | Same protocol | Same protocol |
| Metrics | Sessions, turns, tokens, cost, duration | Same normalized fields | Same normalized fields |
| Runtime controls | Env, timeout, retries, binary override | Same | Same |

The contract is equivalent, not flag-identical. Pi calls its filesystem search
tool `find`, OMP calls it `glob`, and each CLI has its own resume and approval
flags. These differences stay inside the provider adapters. Unsupported native
concepts are handled consistently: plan mode removes mutating tools, explicit
model variants override `#variant`, and provider-reported metrics are normalized
into the shared result type.

## Model selection and reasoning-effort variants

Every provider accepts a `model` option on `.harness()` calls. The model string
may carry a reasoning-effort variant after a `#` separator:

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
| `aforge` | `AFORGE_MODEL` env var with a bare OpenRouter slug (a leading `openrouter/` is stripped) | `AFORGE_EXEC_REASONING` (`off`, `low`, `medium`, or `high`) |
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
