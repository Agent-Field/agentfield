# Harness providers

`app.harness()` hands a task to a coding agent — a multi-turn worker that reads,
writes, and edits files, then reports back through the same structured-output
contract as `app.ai()`. AgentField ships its own harness, **CodeAF**, and it is
the default: a call with no provider or legacy binary pin runs `codeaf`. Naming a different
provider swaps the worker without changing the surrounding loop, which is how
you orchestrate Claude Code, Codex, Gemini CLI, OpenCode, Pi, or OMP from a
reasoner.

## Default: CodeAF

AgentField provisions CodeAF **v0.8.2** in its own managed directory,
`$AGENTFIELD_HOME/bin/codeaf` (`~/.agentfield/bin/codeaf` by default, `.exe` on
Windows). The curl installer, `af skill install --all`, successful `af install`,
and Desktop launch ensure it best-effort. Install and skill hooks honor the
command context and wait at most 30 seconds for an installer lock, warning once
if that wait expires.
A progress line appears only when a download starts. Explicit `af codeaf ensure`
can wait for another installer until canceled; `--lock-wait 30s` bounds that wait.
The Python-agent,
Go-agent, and cloud control-plane Docker images ship `/usr/local/bin/codeaf`;
a failed download or hash verification fails those image builds.

```bash
af codeaf ensure          # install or repair the pinned release
af codeaf ensure --force  # re-download even when already current
```

Installers resolve one managed home for all child commands: staging uses
`~/.agentfield-staging`, otherwise an explicit `AGENTFIELD_HOME` wins, followed
by `~/.agentfield`. `AGENTFIELD_INSTALL_DIR` selects the directory for `af`.
CodeAF always installs into `<managed home>/bin`, so skill provisioning and
the installer's explicit ensure share one download and destination.

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

Without a model credential, Python, Go and TypeScript harness calls fail before
resolving, downloading or launching CodeAF, including with an explicit binary:

```text
Harness provider 'codeaf' needs a model key and none is configured. Set OPENROUTER_API_KEY (https://openrouter.ai/keys), or choose another harness provider with the provider option or AGENTFIELD_HARNESS_PROVIDER.
```

The error remains catchable as the SDK's unavailable-provider error (Python and
TypeScript expose `HarnessProviderNotConfigured`; Go uses
`ProviderUnavailableError` with `NotConfigured: true`). It identifies `codeaf`
and lists `OPENROUTER_API_KEY` as missing. The warning is logged once per process —
at construction for an Agent with a CodeAF harness config and no key (counting
the config's `env`), otherwise at the first refused call. Construction continues
normally; every refused call still fails with the not-configured error.

The check uses the effective child environment, including per-call `env`
overrides and the SDK's default `CODEAF_HOME`. It accepts `OPENROUTER_API_KEY`,
an `OPENAI_API_KEY` starting with `sk-or-`, or another `OPENAI_API_KEY` with a
custom `CODEAF_BASE_URL` (falling back to `AFORGE_BASE_URL`). Ordinary OpenAI keys
are not accepted for the OpenRouter host. Whitespace-only values are empty.
The profile directory is the first non-empty `CODEAF_PROFILE_DIR`,
`AFORGE_PROFILE_DIR`, child `CODEAF_HOME`, or `AFORGE_HOME`; when none is set, or
one is whitespace only, the call is allowed and CodeAF decides.
A saved `api_key` or non-empty `model_sources` array/object in its `config.json`
also permits execution. Unreadable or malformed profile configs and existing
`credentials.json` or `codex.json` permit execution conservatively, so connected
providers and sign-ins can supply their credentials. Legacy aforge pins keep
their existing behavior.

The default model is `~deepseek/deepseek-v4-flash-latest`. Pass `model=` to change
it; the SDK strips one leading `openrouter/` before invoking CodeAF.

Verify the runtime without starting a model task:

```bash
af harness doctor --provider codeaf
```

`aforge` remains an accepted provider alias for CodeAF. To keep the legacy
adapter and its original protocol, select `provider="aforge"` and set
`AFORGE_BIN` to the legacy executable (or supply the SDK's explicit legacy
binary option). `af aforge ensure` still installs the original pinned AForge
binary; it is a separate command, installation and marker.

## Choosing a different provider

Provider selection follows one precedence chain:

| Order | Source | Example |
| --- | --- | --- |
| 1 | Explicit value on the call or in the agent's harness config | `app.harness(task, provider="codex")` |
| 2 | `AGENTFIELD_HARNESS_PROVIDER` environment variable | `AGENTFIELD_HARNESS_PROVIDER=claude-code` |
| 3 | Default | `codeaf` |

Same loop code, different worker:

```python
# CodeAF — nothing to configure
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
| `codeaf` (default; `aforge` alias) | `af codeaf ensure` (shipped with `af`) | None | `codeaf` | `OPENROUTER_API_KEY` |
| `aforge` with a legacy binary pin | `af aforge ensure` | None | `aforge` | `OPENROUTER_API_KEY` |
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

### Managed installation and resolution

CodeAF release assets are raw executables, verified against embedded SHA-256
pins, capped at 128 MiB, and installed atomically. The version marker
`$AGENTFIELD_HOME/bin/.codeaf.version` is written last. Supported platforms are
Linux, macOS and Windows on amd64/arm64. A user-managed `~/.codeaf/bin/codeaf`
is never modified.

The SDK resolves the binary lazily at the first execution: explicit CodeAF
binary option, then `CODEAF_BIN`, then the managed installation, then a managed
download, then `codeaf` on PATH. An unwritable home bin directory is reported
as a download failure. If CodeAF still cannot be found, a managed
or PATH `aforge` binary is a warned legacy
fallback. Otherwise the unavailable error includes `af codeaf ensure` and any
download failure. Results and usage identify the adapter that actually ran.
Both CLI doctors use the same read-only order: `CODEAF_BIN`, home-managed,
then PATH. Managed binaries require the matching version marker.
Doctors never download CodeAF.

Set `AGENTFIELD_CODEAF_BASE_URL` to a full release directory URL (including the
version) to use a mirror. Embedded pins still apply; no downloaded checksum
file is trusted. CLI downloads follow redirects and honor `HTTP_PROXY`,
`HTTPS_PROXY`, `NO_PROXY` and their lowercase forms.
The TypeScript SDK skips downloads when `HTTPS_PROXY` or `https_proxy` is set; run `af codeaf ensure` to install through a proxy.
`AGENTFIELD_SKIP_CODEAF` or the legacy `AGENTFIELD_SKIP_AFORGE`
opts out of every CodeAF download, including explicit ensure and SDK lazy
provisioning. Truthy values are `1`, `true`, `yes`, and `on`, case-insensitive.
The shell installers accept `--no-codeaf` / `CODEAF_MODE=none` and PowerShell
accepts `-NoCodeaf`; old `--no-aforge`, `-NoAforge` and `AFORGE_MODE=none` remain
opt-out aliases. Both skip variables reach all subsequent installer children.
See [environment variables](ENVIRONMENT_VARIABLES.md#coding-harness-codeaf).

The Python extras install wrappers, rather than these managed binaries.
CodeAF, Gemini, Pi and OMP are CLI-only; other providers may require a
separately available executable or wrapper, as listed above.

With no explicit provider (including no `AGENTFIELD_HARNESS_PROVIDER`), an
existing `AFORGE_BIN` or SDK legacy binary option keeps the legacy aforge adapter.
Select `provider="codeaf"` explicitly to use CodeAF despite that pin. Doctor checks
for unpinned `aforge` look for CodeAF without downloading, report its version,
and fall back to checking legacy aforge; a pinned alias checks only that pin.

### CodeAF adapter contract

By default the SDK runs `codeaf exec --json --dir <cwd> --model <model>` with
the task on stdin, an explicit timeout, optional `--max-turns`, and optional
`--system`. Set `AGENTFIELD_CODEAF_COMMAND=do` to use the routed
`codeaf do --json --yes-spend` workflow instead; it pins worker, plan and check
models to the selected model and combines the system prompt and task on stdin.
When `AGENTFIELD_CODEAF_COMMAND` is blank or unset, the process environment's
`AGENTFIELD_AFORGE_COMMAND` is honored before defaulting to `exec`. Models resolve
from the explicit option, then `CODEAF_MODEL`, then `AFORGE_MODEL`, then
`~deepseek/deepseek-v4-flash-latest`; the per-call environment overlays the process
environment, blank values are unset, and one leading `openrouter/` is stripped.
Only `exec` and `do` are accepted; other values warn and use `exec`. CodeAF's
own task spending limit still applies to `do`.

`CODEAF_MAX_CONCURRENT` caps subprocesses at 8 by default, falling back to
`AFORGE_MAX_CONCURRENT` when unset. Invalid or non-positive values use 8.
`AGENTFIELD_HARNESS_TIMEOUT_SECONDS` keeps the existing 1800-second outer
budget, including first-use provisioning and waiting on a shared download.
After provisioning, the subprocess gets the remaining budget and the inner
timeout is five seconds shorter, with a one-second minimum. A call that times
out while provisioning leaves the shared download running for later calls.
An idle watchdog is disabled because CodeAF can stay quiet for minutes.

Each child defaults `CODEAF_HOME` to `$AGENTFIELD_HOME/codeaf`, isolating state
from the user's `~/.codeaf` (directory creation is best effort), and defaults
`CODEAF_NO_UPDATE_CHECK=1`.
`CODEAF_TELEMETRY=off` is supplied when `CI` or `DO_NOT_TRACK` is truthy, or
`AGENTFIELD_TELEMETRY_ENABLED` is explicitly false (`0`, `false`, `no`, `off`).
Explicit caller and per-call environment values win over these defaults.
Inherited `CODEAF_MODELS` and `AFORGE_MODELS` panel settings are removed unless
set in the per-call overlay; the adapter does not set legacy `AFORGE_*` values.

The CodeAF adapter reads the `ok` / `stop` result envelope, using `answer`,
`files`, `spend_usd`, `tokens.in`, `tokens.out`, and `steps`. It accepts a useful
partial answer at budget/turn limits, reports deadlines as timeouts, and treats
blocked questions and incomplete/error/price stops as failures. Legacy-only
AForge envelopes are handled only by the retained legacy adapter. Structured
output uses the existing isolated per-invocation schema protocol.

### Legacy AForge adapter

The original adapter is retained for explicit legacy pins and the warned
fallback. It keeps `AGENTFIELD_AFORGE_COMMAND`, `AFORGE_MAX_CONCURRENT`, its
original JSON/exit contract, and its existing model/variant handling.

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
| System prompt | Per-run agent overlay (inline rollback available) | Native prompt option | Native prompt option |
| Tool allowlist | Ignored; fixed headless baseline | Normalized Pi tool names | Normalized OMP tool names |
| Plan / auto permissions | Ignored; fixed headless baseline | Read-only tools / no approval flag | Read-only tools / `--auto-approve` |
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

OpenCode receives the selected model, project directory, and task prompt, plus
an explicit generated per-run agent configuration containing the system prompt
and fixed headless permissions. Its adapters accept `tools` and
`permission_mode` for provider-neutral API compatibility but intentionally do
not translate either option to OpenCode permissions: the wildcard baseline is
already permissive, so tool entries would only add redundant allow rules, and
`permission_mode` does not select an OpenCode permission mode.

## Model selection and reasoning-effort variants

Every provider accepts a `model` option on `.harness()` calls. Leaving it unset
uses the provider's default — CodeAF uses
`~deepseek/deepseek-v4-flash-latest`, while `claude-code`
keeps using `sonnet`. The model string may carry a reasoning-effort variant
after a `#` separator:

```python
result = await app.harness(
    task,
    provider="opencode",
    model="openrouter/z-ai/glm-5.2#high",
)
```

An explicit `variant="high"` keyword wins over the suffix. In Python,
`variant` is also available on `HarnessConfig` and `HarnessRunner.run`; a
per-call `Agent.harness(..., variant=...)` value overrides the configured
default. Per provider:

Pi and OMP accept the same OpenRouter model strings in every SDK, for example
`openrouter/minimax/minimax-m2.7` or
`openrouter/google/gemini-2.5-flash#low`.

| Provider | Model flag | Variant handling |
| --- | --- | --- |
| `codeaf` (and unpinned `aforge` alias) | `--model` (one leading `openrouter/` stripped); `do` also pins `--plan-model` / `--check-model` | `CODEAF_EXEC_REASONING` (`off`, `low`, `medium`, or `high`) |
| Legacy pinned `aforge` | `exec`: `--model` and `--plan-model`; `do`: `AFORGE_MODEL` (a leading `openrouter/` is stripped) | `AFORGE_EXEC_REASONING` (`off`, `low`, `medium`, or `high`) |
| OpenCode | `-m <model>` | `--variant <v>` (provider-specific effort, e.g. `high`, `max`, `minimal`) |
| Codex | `-m <model>` | `-c model_reasoning_effort=<v>` |
| Claude Code | SDK `model` option | No effort control — variant is dropped with a debug log |
| Gemini | `-m <model>` | No effort control — variant is dropped |
| Pi | `--model <model>` | `--thinking <v>` |
| OMP | `--model <model>` | `--thinking <v>` |

The `#` separator is safe in model ids: `:` belongs to OpenRouter suffixes like
`:free`, and `@` to Vertex-style ids, but no provider uses `#`.

### OpenCode standalone runs

The Python, Go, and TypeScript OpenCode adapters use `opencode run` for each
call; they do not use `opencode serve` or attach to a session. By default, each
adapter selects the fixed
`agentfield-harness` agent with `--agent` and supplies that agent through the
child process's `OPENCODE_CONFIG_CONTENT` environment variable. The generated
overlay sets `$schema` to `https://opencode.ai/config.json`, selects
`agentfield-harness` as the default agent, and fixes its mode to `primary` with
`steps` set to `500`. Set `AGENTFIELD_OPENCODE_STEPS` to a positive integer in
the per-call environment or ambient process environment to override that
default; a per-call value takes precedence, and invalid or non-positive values
fall back to `500`. The adapters do not modify shared OpenCode configuration
files.

The generated overlay is deep-merged into a caller-provided
`OPENCODE_CONFIG_CONTENT` value, or into the ambient value when no per-call
value is supplied. Unrelated providers, agents, MCP servers, and other
configuration remain intact, while AgentField's generated harness fields take
precedence where they overlap. The overlay contains no provider credentials or
API keys. The supplied configuration must be a JSON object; malformed content
is reported instead of being silently discarded.

A fixed worker instruction is always included, in the default agent overlay
or in the inline prompt when the rollback is enabled. When a caller supplies a
non-blank `system_prompt`, it precedes that instruction. The resolved base
`model` and `reasoningEffort` are included only when available.
The base model is also passed with `-m`; a resolved `#variant` suffix is
passed exactly once with `--variant`. AgentField `max_turns` remains a runner
limit and is not serialized as OpenCode `steps`.

OpenCode's initial permission baseline is headless and compatibility-oriented:
the wildcard action is allowed. It is followed by a resource-specific `skill`
denial for `agentfield*`, which prevents the child from loading AgentField
orchestration skills while leaving unrelated skills available. OpenCode
evaluates the last matching permission rule, so this ordering is intentional.
`question` and `task` are also explicitly denied, and no `ask` permission is
generated. All three OpenCode providers accept the common `tools` and
`permission_mode` options but ignore them; they do not translate tool names
into redundant permission entries, and `permission_mode` does not select an
OpenCode permission mode. The wildcard behavior is not per-role authorization.

For an opt-in rollback or compatibility test, set
`AGENTFIELD_OPENCODE_INLINE_SYSTEM_PROMPT=1` (in the per-call environment or
the ambient environment). That path keeps the generated agent selection and
targeted permissions, removes the selected agent's configured system prompt
from the merged configuration, and places the caller system prompt, fixed
worker instruction, and task into one prompt. POSIX sends that prompt as the
positional argument; on Windows, the Python and Go adapters send it over stdin,
while the TypeScript adapter always passes it as the positional argument. The
default path keeps the task prompt (including the runner's schema instructions,
when applicable) as the only user-facing prompt and configures the system
prompt on the generated agent.

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

TypeScript applications can run the same offline check before dispatch:

```typescript
const reports = await agent.harnessDoctor(['codex', 'opencode', 'pi', 'omp']);
for (const report of reports) {
  console.log(report.provider, report.usable, report.issues);
}
```

The preflight ships in the Python and TypeScript SDKs and the `af` CLI. The
equivalent Go SDK API remains a planned follow-up (see #685).

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
