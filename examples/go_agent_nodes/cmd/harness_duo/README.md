# Pi + OMP Go workflow

This example registers one AgentField workflow with three Go reasoners:

```text
compare
├── pi_worker   (Pi harness)
└── omp_worker  (Oh My Pi harness)
```

`compare` starts both child reasoners concurrently and joins their structured
results. Both branches name their provider explicitly — Pi and OMP are
additional providers, while a call that omits `Provider` runs the SDK default,
`aforge`. The default model is
`openrouter/minimax/minimax-m2.7`; set
`HARNESS_MODEL=openrouter/google/gemini-2.5-flash` for the Gemini Flash path.

## Run

Start the AgentField control plane, make sure `OPENROUTER_API_KEY` is set, and
install the CLIs shown by `af harness doctor --provider pi` and
`af harness doctor --provider omp`. Then run:

```bash
cd examples/go_agent_nodes
HARNESS_PROJECT_DIR="$(git rev-parse --show-toplevel)" \
  go run ./cmd/harness_duo serve
```

In another terminal, submit the workflow:

```bash
curl -sS -X POST \
  http://localhost:8080/api/v1/execute/async/harness-duo-go.compare \
  -H 'content-type: application/json' \
  -d '{"input":{"task":"Read README.md and summarize the project with file-path evidence."}}'
```

The returned execution appears in AgentField Desktop as a fan-out from
`compare` to `pi_worker` and `omp_worker`. You can override `model`, `task`, and
`project_dir` in the JSON input. `PI_BIN` and `OMP_BIN` override CLI locations
when the binaries are not on `PATH`.
