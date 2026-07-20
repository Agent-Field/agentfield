# Harness providers

AgentField harness providers run external coding agents. Install the provider
wrapper you need, install its CLI when required, and verify the runtime before
starting a workflow.

## Install

| Provider | Python extra | Required CLI | Authentication |
| --- | --- | --- | --- |
| Claude Code | `agentfield[harness-claude]` | Bundled by `claude-agent-sdk` | Claude login or `ANTHROPIC_API_KEY` |
| Codex | `agentfield[harness-codex]` | `codex` | Codex login or `OPENAI_API_KEY` |
| Gemini | None | `gemini` | Gemini login, `GEMINI_API_KEY`, or `GOOGLE_API_KEY` |
| OpenCode | `agentfield[harness-opencode]` | `opencode` | Provider credentials configured in OpenCode |

Install every Python wrapper with:

```bash
pip install 'agentfield[harness-all]'
```

The extras install Python wrappers. They do not replace the runtime preflight:
Gemini is CLI-only, and Codex or OpenCode may still require a separately
available executable depending on the wrapper and platform.

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
