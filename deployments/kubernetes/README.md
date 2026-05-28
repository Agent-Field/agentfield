# Kubernetes (Kustomize)

Plain Kubernetes manifests for evaluating AgentField without Helm.

If you want a values-driven install, use `deployments/helm/agentfield`.

## Choose an overlay

- `deployments/kubernetes/overlays/python-demo`: control plane + a small Python agent (good default)
- `deployments/kubernetes/overlays/local-demo`: control plane + Go demo agent (requires a custom image)
- `deployments/kubernetes/overlays/postgres-demo`: PostgreSQL + Go demo agent (production-like storage)

If you use `local-demo` or `postgres-demo`, build and load the Go demo agent image first (example for Minikube):

```bash
docker build -t agentfield-demo-go-agent:local -f deployments/docker/Dockerfile.demo-go-agent .
minikube image load agentfield-demo-go-agent:local
```

## Quick start

### 1) Namespace

```bash
kubectl create namespace agentfield --dry-run=client -o yaml | kubectl apply -f -
kubectl config set-context --current --namespace=agentfield
```

### 2) Apply (recommended)

```bash
kubectl apply -k deployments/kubernetes/overlays/python-demo
```

Wait for the control plane and demo agent to start (first run installs Python deps):

```bash
kubectl -n agentfield wait --for=condition=Ready pod -l app.kubernetes.io/component=control-plane --timeout=300s
kubectl -n agentfield wait --for=condition=Ready pod -l app.kubernetes.io/component=demo-python-agent --timeout=600s
```

### 3) Port-forward UI/API

```bash
kubectl port-forward svc/agentfield-control-plane 8080:8080
```

Open:
- `http://localhost:8080/ui/`

Sanity check:
```bash
curl -s http://localhost:8080/api/v1/health
```

### 4) Execute an agent via the control plane

```bash
curl -X POST http://localhost:8080/api/v1/execute/demo-python-agent.hello \
  -H "Content-Type: application/json" \
  -d '{"input":{"name":"World"}}'
```

### 5) Check VCs (optional)

```bash
resp=$(curl -s -X POST http://localhost:8080/api/v1/execute/demo-python-agent.hello \
  -H "Content-Type: application/json" \
  -d '{"input":{"name":"VC"}}')
run_id=$(echo "$resp" | python3 -c 'import sys,json; print(json.load(sys.stdin)["run_id"])')
curl -s http://localhost:8080/api/v1/did/workflow/$run_id/vc-chain | head -c 1200
```

## Other overlays

Go demo agent:
```bash
kubectl apply -k deployments/kubernetes/overlays/local-demo
```

PostgreSQL + Go demo agent:
```bash
kubectl apply -k deployments/kubernetes/overlays/postgres-demo
```

## Notes

- The Python demo agent installs the SDK at startup; your cluster needs outbound network access.
- For agent nodes, always register a `Service` DNS name (`AGENT_PUBLIC_URL` / `AGENT_CALLBACK_URL`), not `localhost`.
