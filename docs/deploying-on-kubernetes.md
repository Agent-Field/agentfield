# Deploying AgentField on Kubernetes

AgentField's control plane must be able to reach every agent at its registered callback URL. In Kubernetes, expose each agent with a `Service` and register its service DNS name. Mount `AGENTFIELD_HOME` on persistent storage when using the `local` storage mode.

## Control-plane shutdown

`AGENTFIELD_SHUTDOWN_MIN_DELAY` is a control-plane-only delay between the shutdown signal and the start of listener shutdown. Its default is `0`, which reproduces the previous timing exactly. It accepts bare seconds (`5`) or Go durations (`5s`, `500ms`); an unparseable or negative value logs one warning and keeps the current value. Equivalent YAML: `agentfield.shutdown_min_delay`.

During the delay the process keeps serving normally — new requests are accepted and completed, and no execution is failed because of the delay. What changes is readiness:

| Route | During the min-delay window | Purpose |
| --- | --- | --- |
| `GET /readyz` | `503` | readiness |
| `GET /api/v1/health/ready` | `503` | readiness |
| `GET /health` | `200` while storage is healthy | liveness |
| `GET /api/v1/health` | `200` while storage is healthy | liveness |

That split is the point of the knob: the readiness routes fail first so kube-proxy removes the pod from the `Service` endpoints while the listener is still open, and liveness keeps passing so the kubelet does not kill a pod that is deliberately draining. All four routes are unauthenticated. Once the delay elapses and shutdown starts, the listener closes and probes get a connection refusal, which the kubelet also treats as a failure — do not expect `503` right up to process exit.

A second SIGTERM or SIGINT during the window terminates the process immediately.

The chart ships `controlPlane.readinessProbe.path: /api/v1/health` on purpose. `controlPlane.image.tag` defaults to `latest` with `pullPolicy: IfNotPresent` and `replicaCount: 1`, so a chart upgrade can land on a node that still has an older cached image; pointing the probe at a path that image does not serve would 404 every probe and leave the `Service` with zero endpoints. Switch to the shutdown-aware path only once every control-plane pod runs an image newer than v0.1.137:

```yaml
controlPlane:
  readinessProbe:
    path: /api/v1/health/ready
```

Leave the liveness probe on `/api/v1/health`.

The chart's readiness probe uses `periodSeconds: 2` and `failureThreshold: 1`. Once the path is switched, the first probe after `BeginDrain` therefore marks the pod Unready within two seconds, leaving about three seconds of the shipped five-second minimum delay for the EndpointSlice change to propagate before the listener closes. If you customize the probe, keep `periodSeconds * failureThreshold` below `AGENTFIELD_SHUTDOWN_MIN_DELAY`; otherwise the listener can close while Kubernetes still considers the pod Ready.

The chart already sets `controlPlane.shutdownMinDelay: 5s` and `controlPlane.terminationGracePeriodSeconds: 60`. A `preStop` hook is a useful complement, because it delays SIGTERM itself while endpoint removal propagates. The shipped control-plane image is distroless and has no `sh` or `sleep` executable, so use Kubernetes' native `sleep` lifecycle action rather than an `exec` hook:

```yaml
spec:
  template:
    spec:
      containers:
        - name: control-plane
          lifecycle:
            preStop:
              sleep:
                seconds: 5
```

The native sleep action was introduced behind the `PodLifecycleSleepAction` feature gate in Kubernetes 1.29, is enabled by default from 1.30, and is GA from 1.34. On 1.29 the cluster administrator must enable that feature gate. On older clusters, or clusters that explicitly disable it, the distroless image cannot run an exec-based sleep; rely on `AGENTFIELD_SHUTDOWN_MIN_DELAY` alone or build a derived image containing a sleep executable and point an `exec` hook directly at that executable.

Size the control-plane pod grace as the `preStop` sleep (if any) + `AGENTFIELD_SHUTDOWN_MIN_DELAY` + `AGENTFIELD_SHUTDOWN_TIMEOUT` + roughly 20 seconds of shutdown tail. The tail is a fresh asynchronous-pool drain budget of at least 5s, plus 5s each for package maintenance, the observability forwarder and the OpenTelemetry tracer, plus an unbounded `adminGRPCServer.GracefulStop()`. Without `preStop`, the shipped `5s` minimum delay and default 30-second shutdown timeout total about 55 seconds, so the chart's `60` leaves about five seconds of headroom. A five-second `preStop` raises the bounded sum to 60 seconds; set `terminationGracePeriodSeconds: 65` to retain that headroom.

The control plane uses `AGENTFIELD_SHUTDOWN_TIMEOUT` (default `30s`) to drain its own HTTP server and its asynchronous execution pool. During shutdown it rejects new queued work, drains active work, and marks work still queued as `failed` with status reason `control_plane_shutdown`.

`AGENTFIELD_SHUTDOWN_MIN_DELAY` is deliberately control-plane-only, and the name is reserved for the control plane so a future SDK does not adopt it and repeat the dual meaning `AGENTFIELD_SHUTDOWN_TIMEOUT` already carries. Note that it applies to `af server` too — the same binary and code path the desktop app launches — so `agentfield.shutdown_min_delay` in `~/.agentfield/agentfield.yaml` also slows a local Ctrl+C.

## Agent shutdown

Python, Go, and TypeScript agents handle SIGTERM by draining in-flight work for `AGENTFIELD_SHUTDOWN_TIMEOUT` (default `30s`). The value accepts bare seconds (`30`) or a duration (`30s`, `5m`). At the deadline, remaining work is cancelled, allowed up to five more seconds to settle, and reported with terminal status `cancelled`. Python and Go also expose `POST /shutdown`; TypeScript uses the process signal. Kubernetes should use SIGTERM directly: do not add an HTTP `preStop` hook, because Kubernetes `httpGet` hooks issue GET while `/shutdown` requires POST.

Set an agent pod's `terminationGracePeriodSeconds` at least 15 seconds above its `AGENTFIELD_SHUTDOWN_TIMEOUT` — five seconds of post-cancel settlement plus headroom for the terminal callback and process exit. The 45 seconds used by the shipped agent manifests is the 30-second default drain plus that 15.

## Rolling deployments and long reasoners

Agent registration is keyed by agent node ID and agent `version`. Keep `version` stable during an ordinary rollout: changing it creates a distinct registered version and bypasses the replacement-instance drain behavior described below.

Agent Deployments must run `replicas: 1` today. Executions are stamped with the node row's current `InstanceID` — the last registrant — and the callback URL is a single field per node ID and version. The replacement path is triggered by *any* re-registration carrying a non-empty `instance_id` that differs from the stored one, not by rollouts specifically, so horizontal replicas of one node ID are indistinguishable from a replacement. The reap additionally sweeps non-terminal rows whose `instance_id` is empty (rows written by SDKs that predate instance stamping). Every shipped agent manifest already sets `replicas: 1`.

When a replacement instance registers with the same node ID and version, the control plane stops routing new work to the departing instance and gives its in-flight executions `AGENTFIELD_AGENT_DRAIN_GRACE` (default `60s`) to finish. This grace is implemented by an in-memory timer, so a control-plane restart loses the timer; the stale sweep controlled by `AGENTFIELD_EXECUTION_STALE_TIMEOUT` (default `30m`) remains the backstop. Executions are scoped by `instance_id`, so only work belonging to the departing instance is reaped.

Dispatch to a node whose last heartbeat falls within the drain window is held for `AGENTFIELD_AGENT_RESTART_GRACE` (default `15s`) while a replacement can register. If none does, the request returns HTTP `503` with `Retry-After: 1`.

### Sizing the drain grace

The deferred reap fires `AGENTFIELD_AGENT_DRAIN_GRACE` after the **replacement registers**, regardless of how much drain budget the departing pod still has. Under a default rolling update the replacement becomes Ready and registers *before* Kubernetes signals the old pod, so that lag counts against the grace:

```text
AGENTFIELD_AGENT_DRAIN_GRACE >=
    (replacement registration -> old-pod SIGTERM lag)
  + AGENTFIELD_SHUTDOWN_TIMEOUT
  + 5s post-cancel settlement
  + terminal-callback latency
  + explicit headroom
```

Set `strategy.rollingUpdate.maxSurge: 0` to make the lag term zero, or measure it and budget for it. Size `AGENTFIELD_AGENT_DRAIN_GRACE` at or above the longest reasoner you are willing to protect.

For an agent whose longest reasoner runs about 10 minutes:

- `strategy.rollingUpdate.maxSurge: 0` on the agent Deployment, so the old pod is signalled before the replacement registers;
- agent env `AGENTFIELD_SHUTDOWN_TIMEOUT: "11m"` — the drain budget must outlast the reasoner;
- agent pod `terminationGracePeriodSeconds: 675` — `11m` plus the 15-second floor above;
- control-plane env `AGENTFIELD_AGENT_DRAIN_GRACE: "12m"` — `11m` drain + 5s settlement + callback latency + headroom;
- control-plane env `AGENTFIELD_AGENT_RESTART_GRACE: "15s"` — leave it at the default; see the warning below.

Write these with unit suffixes. `AGENTFIELD_AGENT_DRAIN_GRACE` is parsed with plain Go duration syntax and has no bare-seconds fallback and no warning on failure, so a bare `660` is silently ignored and the `60s` default survives — unlike `AGENTFIELD_SHUTDOWN_TIMEOUT`, which does accept bare seconds and at least warns. `AGENTFIELD_AGENT_DRAIN_GRACE=0s` does **not** disable the reap: a zero value keeps the `60s` default. A negative duration makes the reap fire immediately. There is no opt-out.

### What raising the drain grace costs

`AGENTFIELD_AGENT_DRAIN_GRACE` is a single **global** control-plane setting, not a per-agent one, and the same window also decides whether an offline node counts as draining. Raising it to `12m` therefore means every crashed agent's in-flight rows sit in `running` for 12 minutes, and a dispatch to any dead node is held for `AGENTFIELD_AGENT_RESTART_GRACE` and then answered `503` for as long as 12 minutes after that node's last heartbeat. Keep `AGENTFIELD_AGENT_RESTART_GRACE` small so those dispatches fail fast rather than blocking a caller.

`AGENTFIELD_EXECUTION_STALE_TIMEOUT` (default `30m`) is the second ceiling: it reaps any non-terminal **leaf** execution regardless of how the drain knobs are set. A leaf that runs longer than 30 minutes cannot be protected by tuning the drain grace alone.

### Why the stock defaults are safe

With the defaults, the agent's 30-second shutdown timeout is well inside the control plane's 60-second drain grace, so a cancelled reasoner reports its terminal `cancelled` status long before the reap could fire. The dangerous configuration is created by raising `AGENTFIELD_SHUTDOWN_TIMEOUT` on its own — increase `AGENTFIELD_AGENT_DRAIN_GRACE` with it, not just `terminationGracePeriodSeconds`.

If the reap does win the race, the row is marked `failed` and its `status_reason` is the string to grep for:

```text
agent_restart_orphaned: previous instance <instance-id> is gone and the execution did not complete within the drain window
```

A late callback against that row behaves as follows: a `succeeded` callback arriving after the reap is rejected with `409` (a terminal status may not be replaced by a different terminal status); a re-delivery of the *same* terminal status is an idempotent `200` no-op; a late **non-terminal** write currently surfaces as `500` rather than a `409` — that is a rough edge to be fixed, not intended behaviour.

## Cleanup and retention

Execution cleanup is enabled by default, including when `AGENTFIELD_CONFIG_FILE=/dev/null` selects defaults plus environment variables. Defaults are:

| Setting | Default | Purpose |
| --- | --- | --- |
| `AGENTFIELD_EXECUTION_CLEANUP_INTERVAL` | `5m` | Time between cleanup passes |
| `AGENTFIELD_EXECUTION_STALE_TIMEOUT` | `30m` | Maximum inactive running-execution age |
| `AGENTFIELD_EXECUTION_CLEANUP_BATCH_SIZE` | `200` | Terminal rows removed per transaction |
| `AGENTFIELD_EXECUTION_PRESERVE_RECENT` | `1h` | Recent rows protected from retention |
| `AGENTFIELD_PAYLOAD_ORPHAN_GRACE` | `1h` | Minimum age for deleting unreferenced payloads |
| `AGENTFIELD_EXECUTION_RETENTION_PERIOD` | `0s` | Terminal-row retention; `0s` keeps rows forever |

Set `AGENTFIELD_EXECUTION_CLEANUP_ENABLED=false` to disable all cleanup. The same settings are available in YAML under `agentfield.execution_cleanup`; environment variables take precedence. Retention deletion is opt-in, so choose a nonzero `AGENTFIELD_EXECUTION_RETENTION_PERIOD` only after deciding how long execution history must remain available.

## Storage

The supported storage modes are exactly `local` and `postgres`; `postgresql` is not a valid mode. For PostgreSQL, set:

```yaml
env:
  - name: AGENTFIELD_CONFIG_FILE
    value: /dev/null
  - name: AGENTFIELD_STORAGE_MODE
    value: postgres
  - name: AGENTFIELD_STORAGE_POSTGRES_URL
    valueFrom:
      secretKeyRef:
        name: agentfield-postgres
        key: url
```

Using `/dev/null` does not disable defaults or cleanup; it avoids relying on the image's bundled YAML and lets environment variables provide deployment-specific configuration. See the [environment variable reference](ENVIRONMENT_VARIABLES.md) for all storage and cleanup settings.
