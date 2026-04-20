# Copilot Agent — GitHub Copilot CLI as an AgentField node

This example exposes **GitHub Copilot CLI** as a first-class AgentField agent
via the official [`github-copilot-sdk`](https://github.com/github/copilot-sdk)
(Public Preview). Other agents on the field can call Copilot the same way
they call any other AgentField reasoner — no subprocess parsing, no bespoke
protocol, full access to Copilot's streaming event model, permission hooks,
and skill discovery.

## Reasoners

| Reasoner | Tools | Use it for |
| --- | --- | --- |
| `ask(prompt, model?, cwd?, isolate?, continue_session?, timeout?)` | **None** | One-shot Q&A, text-only reply. |
| `plan(task, cwd?, model?, isolate?, timeout?)` | **None** | Step-by-step plan without executing anything. |
| `review(diff?, files?, cwd?, model?, isolate?, timeout?)` | Inline diff → **none**; file list → read-only allow-list (`read_file`, `list_directory`, `grep`, `git_diff`). | Code review. |
| `run_task(task, cwd?, model?, allow_tools=False, allow_list?, deny_list?, isolate?, continue_session?, timeout?)` | **Opt-in.** `allow_list` beats `deny_list`. | Full agent mode — Copilot plans *and* executes. |

All reasoners return a stable, structured dict:

```jsonc
{
  "af_session_id": "…",            // AgentField session
  "copilot_session_id": "…",       // Copilot session (mapped via memory)
  "model": "gpt-5",
  "answer": "…",
  "transcript": [{ "role": "assistant", "content": "…" }, …],
  "tool_calls": [{ "tool_name": "read_file", "tool_call_id": "…", "status": "complete" }],
  "usage":   { "input_tokens": 0, "output_tokens": 0 },
  "finished_reason": "idle|error|timeout|aborted",
  "error": null
}
```

## Quickstart

```bash
cd examples/python_agent_nodes/copilot_agent
python -m venv .venv && source .venv/bin/activate
pip install -e .[test]

cp .env.example .env         # edit if needed
export $(grep -v '^#' .env | xargs)

# Optional: one-time Copilot login (the SDK will reuse this).
copilot --login

# Start the control plane separately, then run the agent:
python main.py
```

`af run` also works; the example registers as node id `copilot` by default.

### Calling from another agent

```python
from agentfield import Client

client = Client("http://localhost:8080")
resp = await client.call("copilot.ask", {"prompt": "Explain CRDTs in one paragraph."})
print(resp["answer"])

# Full agent mode — must opt in to tools:
resp = await client.call("copilot.run_task", {
    "task": "Run the unit tests and summarize failures.",
    "cwd": "/workspace",
    "allow_tools": True,
    "allow_list": ["read_file", "list_directory", "grep", "run_tests"],
    "timeout": 600,
})
```

## Authentication

The Copilot SDK tries the following in order:

1. `COPILOT_GITHUB_TOKEN`
2. `GH_TOKEN`
3. `GITHUB_TOKEN`
4. Logged-in user from `~/.copilot` (run `copilot --login` once)

**Token type matters.** Copilot **rejects classic `ghp_*` PATs** — use a
**fine-grained PAT with the "Copilot Requests" permission** or the
interactive login. `af doctor` reports which *sources* are present but
never claims the session is actually authenticated; only `copilot --login`
(or a real request) confirms that.

AgentField has no secrets store by design. Tokens live in the agent
process environment and nowhere else.

## Isolation

By default the agent uses your real `~/.copilot/` directory so that:

- skills installed by `af skill install` (including via this repo's
  `copilot` skillkit target) are visible to the SDK;
- auth from `copilot --login` is reused;
- session resume works across restarts.

Set `AGENTFIELD_COPILOT_ISOLATE=1` (env) or pass `isolate=True` to any
reasoner to use a per-node sandbox under `$AGENTFIELD_HOME/copilot-home/<node_id>`
instead. Isolation is **per-node**, not per-run, so concurrent reasoner
calls on the same agent share one sandbox rather than each bootstrapping
an empty one.

## Continuing a conversation

`ask` and `run_task` accept `continue_session=True`. The AgentField session
id is mapped to a Copilot session id via session-scoped memory (key
`copilot_session_id`). Different AgentField sessions always get
independent Copilot sessions.

## Testing

```bash
pip install -e .[test]
pytest
```

The 21-test suite mocks `copilot.CopilotClient` so no Copilot subscription
is required in CI. `tests/test_sdk_compat.py` is a compatibility smoke
test that imports every SDK symbol this example depends on and fails
loudly if the preview SDK drifts — pin `github-copilot-sdk==0.2.2` in
`pyproject.toml`.

## See also

- `research/docs/2026-04-20-copilot-cli-support.md` — architecture overview.
- `control-plane/internal/skillkit/target_copilot.go` — how `af skill install`
  ships skills into `~/.copilot/skills/` for Copilot to discover.
- `af doctor --json | jq .copilot` — inspect Copilot auth/config on the
  local machine.
