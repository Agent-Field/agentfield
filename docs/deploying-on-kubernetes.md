# Deploying AgentField on Kubernetes

AgentField's control plane must be able to reach every agent at its registered callback URL. In Kubernetes, expose each agent with a `Service` and register its service DNS name. Mount `AGENTFIELD_HOME` on persistent storage when using the `local` storage mode.

## Rolling deployments

Agent registration is keyed by agent node ID and agent `version`. Keep `version` stable during an ordinary rollout: changing it creates a distinct registered version and bypasses the replacement-instance drain behavior.

When a replacement instance registers with the same node ID and version, the control plane stops routing new work to the departing instance and gives its in-flight executions `AGENTFIELD_AGENT_DRAIN_GRACE` (default `60s`) to finish. This grace is implemented by an in-memory timer, so a control-plane restart loses the timer; the stale sweep controlled by `AGENTFIELD_EXECUTION_STALE_TIMEOUT` (default `30m`) remains the backstop. Executions are scoped by `instance_id`, so only work belonging to the departing instance is reaped.

Dispatch to a node whose last heartbeat falls within the drain window is held for `AGENTFIELD_AGENT_RESTART_GRACE` (default `15s`) while a replacement can register. If none does, the request returns HTTP 503.

Python, Go, and TypeScript agents handle SIGTERM by draining in-flight work for `AGENTFIELD_SHUTDOWN_TIMEOUT` (default `30s`). The value accepts bare seconds (`30`) or a duration (`30s`, `5m`). At the deadline, remaining work is cancelled, allowed up to five more seconds to settle, and reported with terminal status `cancelled`. Python and Go also expose `POST /shutdown`; TypeScript uses the process signal. Kubernetes should use SIGTERM directly: do not add an HTTP `preStop` hook, because Kubernetes `httpGet` hooks issue GET while `/shutdown` requires POST.

Use this baseline for every control-plane and agent pod:

```yaml
spec:
  template:
    spec:
      terminationGracePeriodSeconds: 45
```

The 45-second pod grace accommodates the default 30-second agent drain, five-second settlement, and process-exit headroom. If you increase `AGENTFIELD_SHUTDOWN_TIMEOUT`, increase the pod grace too. A practical rollout configuration is:

- control plane: `AGENTFIELD_AGENT_DRAIN_GRACE=60s` and `AGENTFIELD_AGENT_RESTART_GRACE=15s`;
- each agent: `AGENTFIELD_SHUTDOWN_TIMEOUT=30s`;
- each pod: `terminationGracePeriodSeconds: 45`;
- agent manifest: keep `version` unchanged across replica replacements.

The control plane independently uses `AGENTFIELD_SHUTDOWN_TIMEOUT` (default `30s`) to drain its asynchronous execution pool. During shutdown it rejects new queued work, drains active work, and marks work still queued as `failed` with reason `control_plane_shutdown`.

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
