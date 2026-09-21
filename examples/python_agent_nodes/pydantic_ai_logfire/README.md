# Pydantic AI + Logfire

A Pydantic AI agent doing the prompting and validation for one step, inside an
AgentField reasoner that keeps the run DAG, cross-agent calls and replay.
Logfire traces both halves — the Pydantic AI run and the `app.ai` completion —
in a single trace tagged with the AgentField run.

Full write-up: [docs/pydantic-ai.md](../../../docs/pydantic-ai.md).

## Run it

```bash
pip install -r requirements.txt

export OPENROUTER_API_KEY=sk-or-...
export LOGFIRE_TOKEN=...            # or: export LOGFIRE_SEND_TO_LOGFIRE=false (console only)
export AGENTFIELD_URL=http://localhost:8080

python main.py
```

Then, with a control plane running:

```bash
curl -X POST http://localhost:8080/api/v1/execute/ticket-triage.triage_ticket \
  -H 'Content-Type: application/json' \
  -d '{"input": {"ticket": "Checkout returns HTTP 500 for every card payment since the 14:00 deploy."}}'
```

Each execution produces one trace:

```
POST /reasoners/triage_ticket            (fastapi)
├── invoke_agent triager                 (pydantic-ai)
│   └── chat openai/gpt-4o-mini          (pydantic-ai)
└── acompletion                          (litellm, from app.ai)
```

Every span under the root carries `agentfield.run_id`, `agentfield.execution_id`,
`agentfield.node_id` and `agentfield.reasoner`.
