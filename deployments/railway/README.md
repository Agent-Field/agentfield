# AgentField on Railway

AgentField Desktop can deploy a cloud control plane to your Railway account, or
you can create the service from the published Docker Hub image yourself. The
cloud image includes the control plane, the `af` CLI, Git, and the language
toolchains needed to install and run agent packages in the same container.

## Deploy from AgentField Desktop

In AgentField Desktop, open **Cloud**, choose **Railway**, sign in, select a
workspace, and click **Deploy control plane**. Desktop creates the Railway
project, a public domain, an API key, and a persistent volume mounted at
`/data`, then connects Desktop to the new control plane.

Desktop resolves the current stable release and deploys a concrete image such
as:

```text
agentfield/control-plane-cloud:vX.Y.Z
```

Pinning the semantic version makes the deployed release and rollback history
explicit. Do not replace the Desktop-managed volume: `/data` contains the
control-plane databases, credentials, package registry, and installed agents.

## Control-plane image updates

After the first successful deploy, Desktop enables Railway Image Auto Updates
with the **patch** policy and the **Nightly** window: every day from 02:00 to
06:00 UTC. The concrete `vX.Y.Z` image tag therefore follows patch releases
without jumping to a new minor release. On later deploys, Desktop keeps any
policy already present in Railway; it seeds the saved Desktop window only when
Railway has no policy. Desktop's Cloud settings map to Railway as follows:

- **Off** disables the auto-update policy and has no maintenance window.
- **Nightly** uses the patch policy every day from 02:00 to 06:00 UTC.
- **Weekends** uses the patch policy all day Saturday and Sunday UTC.
- **Anytime** uses the patch policy all day, every day.

If the service already uses Railway's **minor** policy, Desktop preserves it
when changing to an enabled window; Railway will then apply minor updates and
patches. Choosing **Off** replaces it with the disabled policy. Otherwise,
Desktop uses the patch policy described above.

Desktop shows Railway's current value whenever it can read it. While that read
is in progress, the control is disabled, retains the last known window for that
service, and says **Checking Railway…**; loading is never presented as **Not set**.
If the read fails, the select shows **Current window unknown — choose one to set
it** with **Last known window: Nightly** (or the applicable cached window) in the
note, not as the selected option. **Not set — choose a window** appears only
after Railway successfully reports that the service has no policy. A window set
to something else in Railway appears as **Custom**; choosing one of the Desktop
options replaces that custom value while preserving a live minor policy unless
you choose **Off**. The same setting is visible and editable directly in Railway:

1. Open the control-plane service.
2. Open **Settings**.
3. Under **Source**, select **Configure Auto Updates**.
4. Choose the update policy and maintenance window, or disable auto updates.

Railway checks the semantic image tag according to the selected policy and
redeploys the service during the selected window. Its maintenance schedules use
UTC. Because the service has an attached volume, allow for a brief interruption
during replacement. See [Railway Image Auto Updates](https://docs.railway.com/deployments/image-auto-updates)
for Railway's update, backup, and notification behavior.

## Deploy the image manually

To manage the Railway service yourself:

1. Create an empty Railway project.
2. Add a service with **Docker Image** as its source.
3. Use `agentfield/control-plane-cloud:latest`.
4. Attach a persistent volume mounted at `/data` before installing agents.
5. Set `AGENTFIELD_PORT=8080` and set `AGENTFIELD_API_KEY` to a securely
   generated value in the service's Variables tab.
6. Generate a public domain under **Settings** → **Networking**.
7. Set the health-check path to `/health`.

The mutable `latest` tag tracks the latest stable control-plane release. When
Railway Image Auto Updates are enabled with an update policy and window, the
service is redeployed after that tag's image digest changes. For repeatable
rollbacks, use a concrete `vX.Y.Z` tag instead.

The default local storage backend works when `/data` is persistent. A separate
PostgreSQL service is optional; configure it with the normal AgentField storage
variables if your deployment requires PostgreSQL. Add provider credentials such
as `OPENROUTER_API_KEY` only when installed agents need them.

Verify the deployment without exposing the API key:

```bash
curl https://your-control-plane.example/health
curl https://your-control-plane.example/api/v1/version \
  -H "X-API-Key: $AGENTFIELD_API_KEY"
```

`/health` is public for Railway health probes. `/api/v1/version` uses the same
authentication as the other `/api/v1` routes and reports the running build and
Railway hosting metadata; it never returns Railway tokens or credentials.

## Agent package updates and boot restore

The cloud image contains Git, so installed agent packages participate in
control-plane maintenance by default:

- Once the HTTP listener is ready (normally after a two-second settle), and
  then every six hours, the control plane checks unpinned Git-installed
  packages and updates eligible packages. If readiness is not observed, the
  boot pass uses a 20-second compatibility fallback.
- A source with an explicit `@ref` remains pinned. A package with
  `auto_update: false` in `installed.yaml` remains paused.
- Updates preserve the package's `.env`. A running execution defers an update,
  and the control plane retries it during the maintenance pass.
- A failed unattended update is recorded as `update.status: failed` with its
  error message and remote commit. That commit is not retried until remote HEAD
  moves; a manual update clears the failed memo.
- On container startup, packages recorded as `running` are restored when their
  old process is no longer alive. Their previous port is reused when available.
- Legacy runtime records that have `started_at` but no process `start_time`
  still receive PID-reuse protection: the observed process start must fall
  between 180 seconds before and 5 seconds after `started_at`. A process
  outside that window is treated as a different process and is never signalled.
- Before a live recorded process is declared unhealthy and restarted, the
  control plane confirms a silent health probe three times, about three seconds
  apart. Status/list reads remain non-blocking and keep an unverified live PID
  for a later lifecycle decision.
- On the first boot after upgrading a legacy registry, entries without
  `desired_state` migrate to `running` in Railway/Docker and are restored.
  Local installations keep the historical status-derived intent. Once an
  explicit stop writes `desired_state: stopped`, later boots do not resurrect it.
- Failed restores or deferred updates schedule the next pass after 1 minute,
  then 5 minutes, then 15 minutes, before returning to the configured interval.
  A clean pass resets this backoff.

Both features depend on the state under `/data`. Without a persistent volume,
the registry and installed packages disappear when Railway replaces the
container, so there is nothing to restore.

### Maintenance environment switches

| Variable | Default | Effect |
|---|---|---|
| `AGENTFIELD_PACKAGE_AUTO_UPDATE` | enabled | Set to `0`, `false`, or `off` to disable unattended package updates. Boot restore still runs. |
| `AGENTFIELD_PACKAGE_UPDATE_INTERVAL` | `6h` | Go duration between maintenance passes, for example `30m` or `12h`. The minimum is `15m`. |

Per-package control is stored as `auto_update: false` in `installed.yaml` and
can also be changed through the package API or AgentField Desktop. Explicit
source pins remain pinned regardless of the global interval.

A manual `POST /api/ui/v1/agents/packages/:id/update` returns HTTP 409 with
`code: executions_active` and `active_executions` when runs are in flight.
Clients may confirm the interruption and retry with `{"force": true}`.

`GET /api/ui/v1/agents/packages/maintenance` reports
`boot_restore_completed: true` as soon as the boot restore loop finishes, even
if update checks in that boot pass are still running. `boot_pass_completed`
only becomes true after the entire boot maintenance pass finishes.

## Updating and recovery

- To update immediately from Desktop, use **Update now** in the Cloud view.
- To update manually in Railway, change the image tag under the service's
  **Settings** → **Source**, or redeploy `latest` after a new digest is
  available.
- To roll back the control plane, use Railway's deployment history. The `/data`
  volume remains attached across image replacements.
- To inspect the running release, call `/api/v1/version` with the service API
  key or run `af version` from a Railway shell.

Agent package maintenance resumes after the replacement control plane starts,
and boot restore brings back packages that were recorded as running.
