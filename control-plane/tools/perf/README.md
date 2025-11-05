# Durable Orchestration Load Tests

This folder contains lightweight tooling for stress testing the durable execution
path introduced in the orchestration refactor.

## Quick start

1. Build the gateway binary (or run `go run ./cmd/agentfield-server`). The production
   build script is still the source of truth:

   ```bash
   cd agentfield/apps/platform/agentfield
   ./build-single-binary.sh
   ./dist/releases/agentfield-darwin-arm64 --config ./configs/local.yaml
   ```

2. Install the load-test dependencies in a virtual environment (or build the
   optional Docker image described below):

   ```bash
   cd agentfield/apps/platform/agentfield/tools/perf
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Target the `synthetic_nested` reasoner (or any workflow that fans out) with the
   stress driver. The script works against both sync and async handlers.

   ```bash
   python nested_workflow_stress.py \
     --base-url http://localhost:8080 \
     --target demo-agent.synthetic_nested \
     --mode async \
     --requests 500 \
     --concurrency 32 \
     --depth 5 --width 3 \
     --payload-bytes 32768 \
     --print-failures
   ```

   The invocation prints aggregate latency, throughput, HTTP-code distribution,
   and execution statuses. Use `--save-metrics` to persist the full JSON snapshot
   for later comparison.

## Containerised runner

Build the utility image from the repo root:

```bash
docker build -t agentfield-perf ./apps/platform/agentfield/tools/perf
```

Run it by either supplying CLI arguments or setting environment variables.
Examples (Linux users may need `--add-host host.docker.internal:host-gateway`):

```bash
# Pass arguments directly
docker run --rm --network host agentfield-perf \
  --base-url http://host.docker.internal:8080 \
  --target demo-agent.synthetic_nested \
  --mode async \
  --requests 300 \
  --concurrency 24 \
  --payload-bytes 65536

# Or rely on environment variables (defaults target to demo-agent.synthetic_nested)
docker run --rm --network host \
  -e TARGET=demo-agent.synthetic_nested \
  -e MODE=async \
  -e REQUESTS=500 \
  -e CONCURRENCY=32 \
  -e PAYLOAD_BYTES=32768 \
  -e PRINT_FAILURES=true \
  -e METRICS_URL=http://host.docker.internal:8080/metrics \
  -e METRICS=process_resident_memory_bytes,go_goroutines \
  agentfield-perf
```

Supported environment variables when no CLI arguments are provided:

| Variable | Default | Description |
| --- | --- | --- |
| `BASE_URL` | `http://host.docker.internal:8080` | Gateway address. |
| `TARGET` | `demo-agent.synthetic_nested` | `node.reasoner` or `node.skill`. |
| `MODE` | `async` | `sync` or `async`. |
| `REQUESTS` | `200` | Total requests to issue. |
| `CONCURRENCY` | `16` | Max in-flight requests. |
| `DEPTH` / `WIDTH` | `0` | Nested fan-out hints. |
| `PAYLOAD_BYTES` | `1024` | Random payload size in bytes. |
| `HEADERS` | _empty_ | Comma separated list of `KEY:VALUE` pairs. |
| `PRINT_FAILURES` | `false` | Emit sample failures to stdout. |
| `SAVE_METRICS` | _empty_ | Path to write JSON results inside container. |
| `METRICS_URL` | _empty_ | Prometheus endpoint to scrape before/after runs. |
| `METRICS` | _empty_ | Comma-separated metric names (defaults applied in code). |
| `METRICS_TIMEOUT` | `5` | Timeout (seconds) for metrics scraping requests. |
| `SCENARIO_FILE` | _empty_ | Path (inside container) to JSON scenario definitions. |

The entrypoint forwards any additional CLI flags to `nested_workflow_stress.py`,
so you can mix and match as needed.

### Local gateway stack (SQLite/GORM)

To exercise the harness against a containerised AgentField server backed by the new
SQLite + GORM storage layer, reuse the Docker image from
`apps/platform/agentfield/docker`:

```bash
# 1) Build and run the AgentField server (see docker/README.md for details)
docker build -t agentfield-local apps/platform/agentfield/docker
docker run --rm -d --name agentfield-local \
  -p 8080:8080 \
  -v "$(pwd)/apps/platform/agentfield/data:/app/data" \
  agentfield-local

# 2) Drive load with the harness (inside this directory)
python nested_workflow_stress.py --base-url http://localhost:8080 \
  --target demo-agent.synthetic_nested --requests 200 --concurrency 16

# 3) Stop the AgentField container when finished
docker stop agentfield-local
```

Because the server persists all durable state in SQLite, no external services or
additional database containers are required. Adjust the harness flags as needed to simulate
different workflows or payload sizes.

## Script capabilities

- Supports sync (`/execute`) and async (`/execute/async`) flows with the same
  CLI by flipping `--mode`.
- Streams large payloads without buffering in Python by generating the payload
  once per request; adjust via `--payload-bytes` or `--body-template` for fully
  custom envelopes.
- Polls async executions with adaptive backoff, jitter, and a configurable
  timeout so nested workflows have room to complete.
- Captures failure examples (including HTTP 429 backpressure responses) when
  `--print-failures` is supplied.
- Dumps metrics (latency p50/p95/p99, throughput, HTTP/status histograms, and
  exceptions) to disk when `--save-metrics <path>` is provided.
- Optional Prometheus scraping (`--metrics-url`) captures pre/post samples for
  memory (`process_resident_memory_bytes`), goroutines, queue depth, and any
  additional metrics you specify, plus deltas for quick comparison.
- Scenario runner (`--scenario-file scenarios.json`) executes named workloads
  back-to-back so you can sweep depth/width, payload size, or concurrency in a
  single container invocation.

Example `scenarios.json`:

```json
[
  {"name": "baseline", "mode": "sync", "requests": 150, "concurrency": 12},
  {"name": "deep-nesting", "mode": "async", "requests": 400, "concurrency": 32, "depth": 6, "width": 3, "payload_bytes": 65536}
]
```

## Integrating with benchmarking runs

- **Nested fan-out sweeps**: Script parameters mirror depth/width knobs in the
  synthetic nested reasoner. Automate sweeps with `for depth in ...` loops and
  keep JSON outputs side by side to compare queue depth, goroutine count, and
  latency after each change.
- **Backpressure verification**: Drive queue overload by setting
  `--requests` well above steady-state capacity. Watch for HTTP 429 responses
  (`status_counts` will include `queue_full`) and confirm Prometheus metrics
  (`agentfield_gateway_queue_depth`, `agentfield_gateway_backpressure_total`) move as
  expected.
- **Memory/CPU sampling**: While the stress harness is running, capture runtime
  stats (`go tool pprof`, `top`, `ps`, or `gops stats`). Persist the JSON metrics
  alongside any pprof artifacts for post-run analysis.

## Locust (optional)

If you prefer browser-based dashboards, you can drop a `locustfile.py` in this
folder that reuses the request helpers above. For now the standalone script keeps
setup minimal and plays nicely with CI; extend as needed.

## Next steps

- Wire the script into the CI/perf pipeline once we settle on baseline thresholds.
- Add scenario presets (YAML/JSON) for the most important workloads (deep
  hierarchical plans, large streaming outputs, webhook-heavy runs).
- Combine metrics with `promtool` snapshots so we keep a single artefact per
  campaign (JSON metrics + Prometheus samples + pprof captures).
