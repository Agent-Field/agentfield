# Deploying workspace-demo and registering it with a control plane

This walks through building the container image, deploying it somewhere with a
public https URL (fly.io is used here purely as one example deployment target
-- any host that gives you a stable https URL works), and registering the
resulting node with a control plane via the `register-serverless` mechanism.

All commands below are run from the **repo root** unless noted otherwise,
because the Dockerfile needs the repo root as its build context (it installs
`sdk/python` from the tree, not a published package).

## 1. Build the image

```bash
docker build -f examples/workspace-demo/Dockerfile -t workspace-demo .
```

Sanity-check it locally before deploying anything:

```bash
docker run --rm -p 8001:8001 \
  -e AGENTFIELD_SERVERLESS=true \
  -e AGENT_NODE_ID=workspace-demo \
  workspace-demo
# in another shell:
curl -s http://localhost:8001/discover | head -c 400
```

`AGENTFIELD_SERVERLESS=true` is what switches `main.py` into the thin
`/discover` + `/execute` wrapper that `register-serverless` expects (no
heartbeat loop) -- see the module docstring in `main.py`.

## 2. Deploy to fly.io

Requires `flyctl` (https://fly.io/docs/flyctl/) already authenticated
(`flyctl auth login`).

Copy `examples/workspace-demo/fly.toml` and replace the placeholder `app`
name with a globally-unique one (fly app names are a shared namespace):

```bash
sed -i '' 's/workspace-demo-CHANGEME/workspace-demo-yourname-123/' examples/workspace-demo/fly.toml   # macOS sed
# or: sed -i 's/workspace-demo-CHANGEME/workspace-demo-yourname-123/' examples/workspace-demo/fly.toml # GNU sed

flyctl apps create workspace-demo-yourname-123   # one-time, matches the app name in fly.toml
```

Deploy, keeping the repo root as build context and pointing `--config` /
`--dockerfile` at the files under `examples/workspace-demo/`:

```bash
flyctl deploy \
  --config examples/workspace-demo/fly.toml \
  --dockerfile examples/workspace-demo/Dockerfile
```

Confirm it's up:

```bash
curl -s https://workspace-demo-yourname-123.fly.dev/discover | head -c 400
```

You should get back JSON with `"node_id":"workspace-demo"` and `report` /
`apply_note` listed under `reasoners`.

## 3. Start a local control plane

```bash
cd control-plane
go run ./cmd/af server
```

Runs in local mode (SQLite + BoltDB, no Postgres needed) at
`http://localhost:8080` by default. See the repo's `CLAUDE.md` for other
storage modes.

## 4. Allow the deployed host for server-side discovery

`register-serverless` makes the control plane fetch `GET <url>/discover`
itself, so it enforces an SSRF allowlist
(`control-plane/internal/handlers/nodes_register.go`,
`isServerlessDiscoveryHostAllowed`). `localhost`/loopback is always allowed;
any other host -- including your fly.dev hostname -- must be added, either in
`control-plane/config/agentfield.yaml`:

```yaml
agentfield:
  registration:
    serverless_discovery_allowed_hosts:
      - "workspace-demo-yourname-123.fly.dev"
      # or, to cover any app on the shared fly.dev domain: "*.fly.dev"
```

or via the environment variable (comma-separated), set before starting the
control plane in step 3:

```bash
export AGENTFIELD_REGISTRATION_SERVERLESS_DISCOVERY_ALLOWED_HOSTS="workspace-demo-yourname-123.fly.dev"
```

Skipping this step makes `register-serverless` fail with:
`invocation_url host "..." is not allowlisted for server-side discovery`.

## 5. Register the node

This is the exact command, verified against
`control-plane/internal/cli/nodes.go` (`af nodes register-serverless`) and the
handler it calls, `control-plane/internal/handlers/nodes_register.go`
(`RegisterServerlessAgentHandler`):

```bash
af nodes register-serverless \
  --server http://localhost:8080 \
  --url https://workspace-demo-yourname-123.fly.dev \
  --json
```

(`--server` can be omitted if `AGENTFIELD_SERVER` is already exported; add
`--token <bearer-token>` if the control plane requires
`AGENTFIELD_AUTHORIZATION_INTERNAL_TOKEN` auth.)

What this does, precisely:
1. Posts `{"invocation_url": "https://workspace-demo-yourname-123.fly.dev"}`
   to `POST http://localhost:8080/api/v1/nodes/register-serverless`.
2. The control plane validates/sanitizes that URL, then itself calls
   `GET https://workspace-demo-yourname-123.fly.dev/discover`.
3. It registers a node with `deployment_type: serverless` and
   `invocation_url` set to `https://workspace-demo-yourname-123.fly.dev/execute`
   -- that is the URL the control plane will `POST` to for every future
   execution of this node's reasoners. There is no heartbeat; the control
   plane only ever needs this one URL.

A successful call prints `Registered serverless agent: workspace-demo` (or
the full JSON envelope with `--json`).

## 6. Call it

```bash
af call workspace-demo.report --in '{}'
```

This runs `report()` against the container's own (ephemeral) filesystem --
useful as a smoke test. The point of this demo is the `--dir` workspace-
artifacts flow (see the top-level README.md and
`docs/design/workspace-artifacts.md`), which works the same way whether the
node is reached via a heartbeat registration or via `register-serverless`;
the node code in `main.py` does not know or care which mode dispatched it.

## Re-deploying

Re-run step 1-2 after any code change, then re-run step 5
(`register-serverless` is also how you refresh an existing node's discovered
reasoners/skills after a redeploy -- the control plane treats a matching
`node_id` as an update, not a duplicate).
