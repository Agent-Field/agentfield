---
date: "2026-04-20"
researcher: "codebase-analyzer (sdk-patterns)"
git_commit: "0f42e612"
branch: "copilot-cli-support"
repository: "agentfield"
topic: "AgentField SDK agent patterns (Python, Go, templates, af run)"
tags: [research, codebase, sdk, python-sdk, go-sdk, templates]
status: complete
last_updated: "2026-04-20"
last_updated_by: "codebase-analyzer"
---

## Analysis: Building an AgentField Agent — Python SDK, Go SDK, `af init` Templates, `af run`, and ExecutionContext

---

### Overview

An AgentField agent is a small HTTP server (FastAPI in Python, `net/http` in Go) that registers its callable functions ("reasoners" or "skills") with a central control-plane server. Reasoners are exposed as individual `POST` endpoints; the control plane routes execution requests to those endpoints and propagates a correlation-header bundle (`X-Run-ID`, `X-Execution-ID`, etc.) as the `ExecutionContext`. The `af init` scaffolding tool generates the minimal directory with one ready-to-run echo reasoner, and `af run` / `af dev` launch the resulting `main.py` or Go binary while injecting `PORT` and `AGENTFIELD_SERVER_URL` into the process environment.

---

### Part 1 — Python SDK

#### 1.1 The `Agent` Class

**File:** `sdk/python/agentfield/agent.py`

`Agent` subclasses `FastAPI` directly (`agent.py:419`). Construction signature (`agent.py:464–484`):

```python
Agent(
    node_id: str,                        # required — unique registration key
    agentfield_server: Optional[str],    # control-plane URL; falls back to
                                         # AGENTFIELD_SERVER / AGENTFIELD_SERVER_URL / "http://localhost:8080"
    version: str = "1.0.0",
    ai_config: Optional[AIConfig] = None,
    dev_mode: bool = False,
    callback_url: Optional[str] = None,  # explicit callback URL the CP uses to reach this agent
    vc_enabled: Optional[bool] = True,
    api_key: Optional[str] = None,
    enable_did: bool = True,
    ...
)
```

`agentfield_server` resolution order (`agent.py:555–558`):
1. Explicit parameter
2. `AGENTFIELD_SERVER` env var
3. `AGENTFIELD_SERVER_URL` env var
4. Literal `"http://localhost:8080"`

After `super().__init__()`, the constructor stores `node_id` and `agentfield_server` as instance attributes (`agent.py:563–564`), constructs an `AgentFieldClient` (`agent.py:606–607`), initialises in-memory `_reasoner_registry` and `_skill_registry` dicts (`agent.py:572–573`), and creates `workflow_handler`, `agentfield_handler`, and `agent_server` helpers.

#### 1.2 The `@app.reasoner()` Decorator

**File:** `sdk/python/agentfield/agent.py:1615–1829`

The decorator can be applied directly (`@app.reasoner`) or with keyword options:

| Option | Purpose |
|---|---|
| `path` | Override the default `/reasoners/<name>` endpoint path |
| `name` | Override the registration ID (defaults to `func.__name__`) |
| `tags` | List of organizational tags sent at registration |
| `vc_enabled` | Override per-reasoner VC generation |
| `require_realtime_validation` | Force CP-side verification instead of local |

What the decorator does step by step (`agent.py:1649–1829`):
1. Derives `reasoner_id` from `name` or `func.__name__` (`agent.py:1652`).
2. Derives `endpoint_path` as `/reasoners/<reasoner_id>` (or custom path) (`agent.py:1653–1660`).
3. Inspects parameter type hints to build `input_fields` dict mapping param-name → `(type, default)` (`agent.py:1667–1676`).
4. Registers a FastAPI `POST` handler at `endpoint_path` (`agent.py:1693–1744`).
5. Creates a `tracked_func` wrapper that routes through `workflow_handler.execute_with_tracking` when a context is present (`agent.py:1753–1776`).
6. Stores the entry in `_reasoner_registry[reasoner_id]` (`agent.py:1798–1804`).
7. Calls `workflow_handler.replace_function_references(original_func, tracked_func, func_name)` so intra-module direct calls also go through tracking (`agent.py:1812`).

#### 1.3 The `AgentFieldHandler` and Registration

**File:** `sdk/python/agentfield/agent_field_handler.py`

`AgentFieldHandler` owns two concerns:

- **Registration** (`agent_field_handler.py:41–~200`): `register_with_agentfield_server(port)` resolves the callback URL the control plane should use to reach the agent, then `POST`s the node manifest (node_id, version, reasoners list, base_url) to the control plane.
- **Heartbeat** (`agent_field_handler.py`): starts a background thread that `PUT`s a heartbeat every `heartbeat_interval` seconds (default 2 s) to keep the CP lease alive.

#### 1.4 How Reasoners Become HTTP Endpoints

**File:** `sdk/python/agentfield/agent_server.py`

`AgentServer.setup_agentfield_routes()` (`agent_server.py:37`) registers platform-level routes on the `Agent` FastAPI app:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Health probe (`agent_server.py:133`) |
| `GET` | `/reasoners` | List all registered reasoners (`agent_server.py:144`) |
| `GET` | `/skills` | List skills (`agent_server.py:148`) |
| `GET` | `/status` | Detailed status + resource usage (`agent_server.py:224`) |
| `GET` | `/info` | Node metadata (`agent_server.py:285`) |
| `POST` | `/shutdown` | Graceful stop (`agent_server.py:152`) |
| `GET` | `/agentfield/v1/logs` | NDJSON log stream (`agent_server.py:43`) |
| `POST` | `/webhooks/approval` | Approval-decision callback (`agent_server.py:301`) |

Each `@app.reasoner()` call also registers a `POST /reasoners/<id>` endpoint via `@self.post(endpoint_path)` (`agent.py:1693`).

#### 1.5 The FastAPI Endpoint Handler Flow (per request)

**File:** `sdk/python/agentfield/agent.py:1694–1744` and `agent.py:1831–1960`

1. The POST body is parsed to JSON (`agent.py:1697–1701`).
2. Input is runtime-validated against `handler_input_fields` via `_validate_handler_input` (`agent.py:1705–1713`).
3. If `X-Execution-ID` header is present and the agent is connected to a CP, the call is promoted to an **async fire-and-forget** task that calls back the CP with the result; the endpoint immediately returns `HTTP 202` with `{"status":"processing","execution_id":"..."}` (`agent.py:1724–1742`).
4. Otherwise it is a **synchronous** call: `_execute_reasoner_endpoint` is awaited in-place (`agent.py:1744`).

Inside `_execute_reasoner_endpoint` (`agent.py:1831–1960`):
1. `ExecutionContext.from_request(request, self.node_id)` is called to extract correlation headers (`agent.py:1843`).
2. `workflow_handler.notify_call_start(...)` fires (`agent.py:1850–1858`).
3. Pydantic args conversion via `convert_function_args` if applicable (`agent.py:1880–1898`).
4. If the function signature includes `execution_context`, it is injected automatically (`agent.py:1900–1901`).
5. `func(*args, **kwargs)` is awaited or called (`agent.py:1903–1906`).
6. `workflow_handler.notify_call_complete(...)` fires (`agent.py:1930–1940`).

#### 1.6 `AgentRouter` — Grouping Reasoners

**File:** `sdk/python/agentfield/router.py`

`AgentRouter(prefix="demo", tags=["example"])` collects reasoners before they are attached to an `Agent`:

```python
reasoners_router = AgentRouter(prefix="demo", tags=["example"])

@reasoners_router.reasoner()
async def echo(message: str) -> dict:
    ...

app.include_router(reasoners_router)   # agent.py:2666
```

`AgentRouter.reasoner()` (`router.py:28`) stores `{"func": func, "wrapper": wrapper, "path": ..., "tags": ...}` in `self.reasoners`. `_attach_agent` (`router.py:218`) links the router back to the agent so that `reasoners_router.ai(...)` and other agent methods work through `__getattr__` delegation (`router.py:131–163`). After `include_router`, the path for the echo reasoner above becomes `/reasoners/demo_echo` (prefix + name joined with `_`).

#### 1.7 The `@reasoner` Standalone Decorator

**File:** `sdk/python/agentfield/decorators.py`

`@reasoner` / `@reasoner(track_workflow=True)` can be applied to functions **not** registered on an `Agent`. It sets `wrapper._is_reasoner = True` and other metadata attributes (`decorators.py:71–83`). When called, it enters `_execute_with_tracking` (`decorators.py:96`), which:
- Gets the current `ExecutionContext` via `get_current_context()` (`decorators.py:109`)
- Creates a root or child context (`decorators.py:126–141`)
- Calls `workflow_handler._ensure_execution_registered(...)` (`decorators.py:151–154`)
- Fires start/completion/error notifications to the workflow handler (`decorators.py:262–307`)

#### 1.8 Minimal Python Agent Shape

Based on `sdk/python/agentfield/agent.py` and templates:

```python
# main.py
import os
from agentfield import Agent, AIConfig
from reasoners import reasoners_router

app = Agent(
    node_id=os.getenv("AGENT_NODE_ID", "copilot-agent"),
    agentfield_server=os.getenv("AGENTFIELD_SERVER", "http://localhost:8080"),
    version="1.0.0",
    dev_mode=True,
)
app.include_router(reasoners_router)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8001")), auto_port=False)

# reasoners.py
import subprocess
from agentfield import AgentRouter

reasoners_router = AgentRouter(prefix="copilot", tags=["cli"])

@reasoners_router.reasoner()
async def run_command(prompt: str) -> dict:
    result = subprocess.run(
        ["copilot", "--prompt", prompt],
        capture_output=True, text=True
    )
    return {"stdout": result.stdout, "returncode": result.returncode}
```

*All inputs arrive as keyword arguments matching the function signature; `execution_context: ExecutionContext` can be added as an extra parameter and it will be injected automatically.*

#### 1.9 `serve()` / `run()` — Server Start

**File:** `sdk/python/agentfield/agent_server.py:617–800`, `sdk/python/agentfield/agent.py:4350–4418`

`app.run(**serve_kwargs)` (`agent.py:4350`) inspects `sys.argv[1]`; if it's one of `["call", "list", "shell", "help"]` it enters CLI mode, otherwise it calls `app.serve(...)` (`agent.py:4413–4418`).

`serve()` port selection priority (`agent_server.py:657–714`):
1. Explicit `port` argument
2. `PORT` env var (set by `af run` / `af dev`)
3. `AGENTFIELD_AUTO_PORT=true` or `auto_port=True` → `get_free_port()`
4. Default: 8001 if free, otherwise next free port

`serve()` also reads `AGENT_CALLBACK_URL` to set `base_url` for CP-side reverse routing (`agent_server.py:721–742`).

---

### Part 2 — Go SDK

#### 2.1 `Config` Struct

**File:** `sdk/go/agent/agent.go:167–300`

```go
type Config struct {
    NodeID               string        // required
    Version              string        // required
    AgentFieldURL        string        // control-plane URL (empty = offline / CLI-only)
    ListenAddress        string        // default ":8001"
    PublicURL            string        // URL reported to CP; default "http://localhost" + ListenAddress
    Token                string        // Bearer token for CP auth
    InternalToken        string        // token validated on *inbound* execution requests
    DeploymentType       string        // "long_running" (default) or "serverless"
    LeaseRefreshInterval time.Duration // default 2m
    DisableLeaseLoop     bool
    RequireOriginAuth    bool          // validate inbound requests carry InternalToken
    LocalVerification    bool          // verify DID sigs locally, skipping CP per-call
    MemoryBackend        MemoryBackend // nil = in-process map
    AIConfig             *ai.Config
    CLIConfig            *CLIConfig
    Tags                 []string
    EnableDID            bool
    VCEnabled            bool
    Logger               *log.Logger
    HarnessConfig        *HarnessConfig
}
```

Defaults applied by `New()` (`agent.go:353–401`): `TeamID → "default"`, `ListenAddress → ":8001"`, `PublicURL → "http://localhost:8001"`, `DeploymentType → "long_running"`, `LeaseRefreshInterval → 2m`.

#### 2.2 `RegisterReasoner`

**File:** `sdk/go/agent/agent.go:522–551`

```go
func (a *Agent) RegisterReasoner(name string, handler HandlerFunc, opts ...ReasonerOption)
```

`HandlerFunc` (`agent.go:57`):
```go
type HandlerFunc func(ctx context.Context, input map[string]any) (any, error)
```

`RegisterReasoner` creates a `Reasoner` struct, applies `ReasonerOption` functions, and stores it in `a.reasoners[name]`. Available options:

| Option function | What it sets |
|---|---|
| `WithInputSchema(json.RawMessage)` | Override auto-generated JSON schema (`agent.go:63`) |
| `WithOutputSchema(json.RawMessage)` | Override output schema (`agent.go:72`) |
| `WithCLI()` | Mark as CLI-accessible (`agent.go:81`) |
| `WithDefaultCLI()` | Designate as the default CLI entry point (`agent.go:87`) |
| `WithCLIFormatter(func)` | Custom output formatter for CLI (`agent.go:96`) |
| `WithDescription(string)` | Human-readable description (`agent.go:103`) |
| `WithVCEnabled(bool)` | Per-reasoner VC override (`agent.go:132`) |
| `WithReasonerTags(...string)` | Tags for tag-based authorization (`agent.go:139`) |
| `WithRequireRealtimeValidation()` | Force CP verification (`agent.go:147`) |

#### 2.3 HTTP Routing (`handler()`)

**File:** `sdk/go/agent/agent.go:932–959`

```go
mux.HandleFunc("/health",              a.healthHandler)
mux.HandleFunc("/discover",            a.handleDiscover)
mux.HandleFunc("/agentfield/v1/logs",  a.handleAgentfieldLogs)
mux.HandleFunc("/execute",             a.handleExecute)
mux.HandleFunc("/execute/",            a.handleExecute)
mux.HandleFunc("/reasoners/",          a.handleReasoner)
```

Middleware is conditionally wrapped: `localVerificationMiddleware` when `LocalVerification=true` (`agent.go:944–947`), `originAuthMiddleware` when `RequireOriginAuth=true` (`agent.go:950–957`).

#### 2.4 `handleExecute` — Request-to-Handler Path

**File:** `sdk/go/agent/agent.go:1154–1241`

1. `targetName` is extracted from the URL path after `/execute/` (`agent.go:1160–1161`).
2. JSON body is decoded to `payload map[string]any` (`agent.go:1163–1170`).
3. `reasonerName` resolved from path, then from `payload["reasoner"|"target"|"skill"]` (`agent.go:1175–1183`).
4. `extractInputFromServerless(payload)` isolates the user input under the `"input"` key, or filters out control fields (`agent.go:1191`, `1243–1265`).
5. `buildExecutionContextFromServerless(r, payload, reasonerName)` reads HTTP headers into an `ExecutionContext` (`agent.go:1192`, `1267–1308`).
6. `ExecutionContext` is stored on `ctx` via `contextWithExecution` (`agent.go:1194`).
7. `reasoner.Handler(ctx, input)` is called directly (`agent.go:1201`).
8. Result written as JSON with `writeJSON` (`agent.go:1240`).

#### 2.5 `ExecutionContext` (Go)

**File:** `sdk/go/agent/agent.go:32–50`

```go
type ExecutionContext struct {
    RunID             string
    ExecutionID       string
    ParentExecutionID string
    SessionID         string
    ActorID           string
    WorkflowID        string
    ParentWorkflowID  string
    RootWorkflowID    string
    Depth             int
    AgentNodeID       string
    ReasonerName      string
    StartedAt         time.Time
    CallerDID         string
    TargetDID         string
    AgentNodeDID      string
}
```

Retrieved inside a handler via `agent.ExecutionContextFrom(ctx)` (`agent.go:2042`), which reads the value stored by `contextWithExecution`.

`ChildContext(agentNodeID, reasonerName)` (`agent.go:444–479`) creates a derived context for local nested calls, incrementing `Depth` and setting `ParentExecutionID`.

#### 2.6 `Run()` / `Serve()` / `Initialize()`

**File:** `sdk/go/agent/agent.go:616–660`

- `Run(ctx)` (`agent.go:616`): checks `os.Args[1:]`; if no args and no CLI-enabled reasoners, calls `Serve(ctx)`; if `args[0] == "serve"`, calls `Serve(ctx)`; otherwise dispatches to `runCLI(ctx, args)`.
- `Serve(ctx)` (`agent.go:630`): calls `Initialize(ctx)` then `startServer()`, then blocks on `ctx.Done()` or OS signal.
- `Initialize(ctx)` (`agent.go:554`): calls `registerNode(ctx)` (POST to CP with all reasoners listed), optionally `initializeDIDSystem`, then `markReady` and `startLeaseLoop()`.

Node registration payload (`agent.go:680–706`):
```go
types.NodeRegistrationRequest{
    ID:        cfg.NodeID,
    BaseURL:   cfg.PublicURL,
    Version:   cfg.Version,
    Reasoners: []types.ReasonerDefinition{...},  // one per registered reasoner
    CommunicationConfig: types.CommunicationConfig{
        Protocols:         ["http"],
        HeartbeatInterval: ...,
    },
}
```

#### 2.7 `Call()` — Cross-Agent Invocation

**File:** `sdk/go/agent/agent.go:1570–~1680`

`Call(ctx, "other-node.some_reasoner", input)` (`agent.go:1570`) POSTs to `<AgentFieldURL>/api/v1/execute/<target>`, forwarding `X-Run-ID`, `X-Parent-Execution-ID`, `X-Workflow-ID`, `X-Session-ID`, `X-Actor-ID`, and DID headers extracted from the current `ExecutionContext` (`agent.go:1606–1628`).

#### 2.8 Memory (Go)

**File:** `sdk/go/agent/memory.go`

`agent.Memory()` returns `*Memory` (`agent.go:2059`). Default scope methods use `ScopeSession`, deriving the scope-ID from `ctx` via `ExecutionContextFrom(ctx).SessionID` (falling back to `RunID`) (`memory.go:80–85`). Explicit scopes via:

| Method | Scope | Scope-ID source |
|---|---|---|
| `Memory().Set/Get(ctx, key)` | `ScopeSession` | `execCtx.SessionID` or `RunID` |
| `Memory().WorkflowScope()` | `ScopeWorkflow` | `execCtx.WorkflowID` or `RunID` (`memory.go:196–207`) |
| `Memory().SessionScope()` | `ScopeSession` | `execCtx.SessionID` (`memory.go:212`) |
| `Memory().UserScope()` | `ScopeUser` | `execCtx.ActorID` |
| `Memory().GlobalScope()` | `ScopeGlobal` | `"global"` |
| `Memory().Scoped(scope, id)` | explicit | explicit (`memory.go:101`) |

`MemoryBackend` is pluggable (`agent.go:238`). The built-in is `InMemoryBackend`. A `ControlPlaneMemoryBackend` that proxies to the CP's `/memory/set` and `/memory/get` endpoints is available separately (`control_plane_memory_backend.go`).

#### 2.9 Minimal Go Agent Shape

Based on `control-plane/internal/templates/go/main.go.tmpl` and `reasoners.go.tmpl`:

```go
// main.go
package main

import (
    "context"
    "log"
    "os/exec"
    "github.com/Agent-Field/agentfield/sdk/go/agent"
)

func main() {
    cfg := agent.Config{
        NodeID:        "copilot-agent",
        Version:       "1.0.0",
        AgentFieldURL: "http://localhost:8080",
        ListenAddress: ":8001",
    }
    app, err := agent.New(cfg)
    if err != nil { log.Fatal(err) }

    app.RegisterReasoner("run_command", func(ctx context.Context, input map[string]any) (any, error) {
        prompt, _ := input["prompt"].(string)
        out, err := exec.CommandContext(ctx, "copilot", "--prompt", prompt).Output()
        if err != nil {
            return nil, err
        }
        return map[string]any{"stdout": string(out)}, nil
    }, agent.WithDescription("Shell out to copilot CLI"))

    log.Fatal(app.Run(context.Background()))
}
```

*`input` is always `map[string]any`. The `ctx` carries the full `ExecutionContext`; retrieve it with `agent.ExecutionContextFrom(ctx)` if needed.*

---

### Part 3 — `af init` Templates

#### 3.1 Files Generated for Python

**Directory:** `control-plane/internal/templates/python/`

Template files and their output names after stripping `.tmpl` (`templates.go:52–61`):

| Template | Output file |
|---|---|
| `python/main.py.tmpl` | `main.py` |
| `python/reasoners.py.tmpl` | `reasoners.py` |
| `python/requirements.txt.tmpl` | `requirements.txt` |
| `python/.env.example.tmpl` | `.env.example` |
| `python/.gitignore.tmpl` | `.gitignore` |
| `python/README.md.tmpl` | `README.md` |

`requirements.txt.tmpl` contains a single line: `agentfield` (`requirements.txt.tmpl:1`).

`.env.example.tmpl` contains: `AGENTFIELD_CONTROL_PLANE_URL=http://localhost:8080` (`python/.env.example.tmpl:1`).

Generated `main.py` (`main.py.tmpl:1–32`):
- Constructs `Agent(node_id=..., agentfield_server=os.getenv("AGENTFIELD_SERVER", "http://localhost:8080"), ...)` where `node_id` comes from `AGENT_NODE_ID` env var or the baked-in `{{.NodeID}}` value.
- Calls `app.include_router(reasoners_router)` linking the generated `reasoners.py`.
- Entry: `app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8001")), auto_port=False)`.

Generated `reasoners.py` (`reasoners.py.tmpl:1–57`):
- Creates `reasoners_router = AgentRouter(prefix="demo", tags=["example"])`.
- Registers one `@reasoners_router.reasoner()` function named `echo(message: str) -> dict` that returns `{"original": message, "echoed": message, "length": len(message)}`.

#### 3.2 Files Generated for Go

**Directory:** `control-plane/internal/templates/go/`

| Template | Output file |
|---|---|
| `go/main.go.tmpl` | `main.go` |
| `go/reasoners.go.tmpl` | `reasoners.go` |
| `go/go.mod.tmpl` | `go.mod` |
| `go/.env.example.tmpl` | `.env.example` |
| `go/.gitignore.tmpl` | `.gitignore` |
| `go/README.md.tmpl` | `README.md` |

Generated `main.go` (`main.go.tmpl:1–41`):
- `agent.Config{NodeID: "{{.NodeID}}", Version: "1.0.0", AgentFieldURL: "http://localhost:8080", ListenAddress: ":0"}` — port `0` means the OS assigns any free port.
- `agent.New(cfg)` → `registerReasoners(app)` → `app.Run(context.Background())`.

Generated `reasoners.go` (`reasoners.go.tmpl:1–49`):
- `func registerReasoners(app *agent.Agent)` calls `app.RegisterReasoner("echo", ...)` with the same echo logic.

#### 3.3 Optional Docker Scaffold (`--docker` flag)

If `af init --docker` is used, `GetDockerTemplateFiles(language)` (`templates.go:92–103`) also produces:
- `docker-compose.yml`
- `.env.example` (overwritten)
- `.dockerignore`
- `Dockerfile` (Python only)

#### 3.4 `TemplateData` Fields

**File:** `control-plane/internal/templates/templates.go:15–29`

`{{.ProjectName}}`, `{{.NodeID}}`, `{{.GoModule}}`, `{{.AuthorName}}`, `{{.AuthorEmail}}`, `{{.CurrentYear}}`, `{{.CreatedAt}}`, `{{.Language}}`, `{{.ControlPlaneImage}}`, `{{.ControlPlanePort}}`, `{{.AgentPort}}`, `{{.DefaultModel}}`.

`NodeID` is derived from `ProjectName` by lowercasing, replacing `_` → `-`, and collapsing multiple hyphens (`init.go:526–532`).

#### 3.5 `af init` Command

**File:** `control-plane/internal/cli/init.go:204–511`

Interactive TUI (Bubble Tea) with 4 steps: project-name → language selection → author name → author email. Non-interactive via `--non-interactive` / `--defaults` flags. `--language` / `-l` accepts `python`, `go`, `typescript` (with aliases `py`, `ts`, `golang`, etc.) (`init.go:558–569`).

---

### Part 4 — `af run` Discovery and Execution

#### 4.1 `af run <agent-name>`

**File:** `control-plane/internal/cli/commands/run.go:36–127`, `control-plane/internal/core/services/agent_service.go:51–136`

`RunCommand.execute(name, port, detach, verbose)` (`run.go:69`) calls `cmd.Services.AgentService.RunAgent(name, options)`.

`DefaultAgentService.RunAgent` (`agent_service.go:51`):
1. Loads `~/.agentfield/installed.yaml` registry (`agent_service.go:55–64`).
2. Finds the agent entry, reconciles stale PID state.
3. Calls `portManager.FindFreePort(8001)` for port `8001+` if none given (`agent_service.go:87–93`).
4. Calls `buildProcessConfig(agentNode, port)` (`agent_service.go:448–525`) which:
   - Sets `env PORT=<port>` (`agent_service.go:451`).
   - Sets `env AGENTFIELD_SERVER_URL=<resolved-url>` (`agent_service.go:452`).
   - Loads `.env` from the package directory (`agent_service.go:455–460`).
   - Resolves Python path: `<pkg>/venv/bin/python` → `<pkg>/venv/Scripts/python.exe` → system `python` (`agent_service.go:463–516`).
   - Returns `ProcessConfig{Command: pythonPath, Args: ["main.py"], WorkDir: agentNode.Path}`.
5. Starts the process and polls `http://localhost:<port>/health` every 500 ms until `HTTP 200` within 10 s (`agent_service.go:104–111`, `528–546`).
6. Updates `installed.yaml` with PID + port + started-at (`agent_service.go:116`).

#### 4.2 `af dev [path]`

**File:** `control-plane/internal/cli/commands/dev.go`, `control-plane/internal/core/services/dev_service.go`

`DevService.RunInDevMode(path, options)` (`dev_service.go:40`) checks for `agentfield.yaml` in the target directory (`dev_service.go:49`), then calls `runDev(absPath, options)`.

`startDevProcess` (`dev_service.go:183`) sets the same env vars (`PORT`, `AGENTFIELD_SERVER_URL`, `AGENTFIELD_DEV_MODE=true`), runs `python main.py` with stdout/stderr piped to the terminal, and does **not** detach.

`discoverAgentPort` (`dev_service.go:242`) polls `http://localhost:<n>/health` for ports 8001–8999 with 2 s per-request timeout up to the overall timeout (120 s default) (`dev_service.go:260–267`).

The old `packages/runner.go` path (`runner.go:112`) does the same: sets `PORT`, `AGENTFIELD_SERVER_URL`, runs `python main.py`.

#### 4.3 Key Environment Variables Consumed by the Agent

| Variable | Set by | Consumed at |
|---|---|---|
| `PORT` | `af run` / `af dev` | `agent_server.py:659` (Python); agent uses it as the listen port |
| `AGENTFIELD_SERVER` | user / `.env` | `agent.py:556` (Python `agentfield_server` default) |
| `AGENTFIELD_SERVER_URL` | `af run` / `af dev` (`agent_service.go:452`) | Same fallback as above |
| `AGENT_NODE_ID` | user | `main.py.tmpl:16` — overrides the baked-in node ID |
| `AGENT_CALLBACK_URL` | user / container env | `agent_server.py:721` — the URL the CP uses to call back |
| `AGENTFIELD_AUTO_PORT` | user | `agent_server.py:678` — triggers auto-port mode |

For the Go SDK, the go template uses `AGENTFIELD_URL` (`go_agent_nodes/main.go:21`) and `AGENT_NODE_ID` (`main.go:15`), `AGENT_LISTEN_ADDR` (`main.go:22`), `AGENT_PUBLIC_URL` (`main.go:26`), `AGENTFIELD_TOKEN` (`main.go:35`).

#### 4.4 `af install` — Package Registry

Before `af run` can find an agent, it must be installed via `af install <source>`, which:
- Validates presence of `agentfield-package.yaml` and `main.py` in the source (`installer.go:392–406`).
- Reads `agentfield-package.yaml` fields: `name`, `version`, `description`, `main` (default `"main.py"`), `agent_node.node_id`, `dependencies.python[]` (`installer.go:32–69`).
- Copies the package directory to `~/.agentfield/packages/<name>/` and installs Python dependencies into `<pkg>/venv/`.
- Writes an entry into `~/.agentfield/installed.yaml`.

---

### Part 5 — Memory Scopes and `ExecutionContext` at Runtime

#### 5.1 Python `ExecutionContext`

**File:** `sdk/python/agentfield/execution_context.py`

`@dataclass ExecutionContext` fields:

| Field | Type | Source |
|---|---|---|
| `run_id` | `str` | `X-Run-ID` header or generated |
| `execution_id` | `str` | `X-Execution-ID` header or generated |
| `agent_instance` | `Any` | set to current `Agent` from registry |
| `reasoner_name` | `str` | set by endpoint handler |
| `agent_node_id` | `Optional[str]` | `agent.node_id` |
| `parent_execution_id` | `Optional[str]` | `X-Parent-Execution-ID` header |
| `depth` | `int` | incremented for child contexts |
| `session_id` | `Optional[str]` | `X-Session-ID` header |
| `actor_id` | `Optional[str]` | `X-Actor-ID` header |
| `caller_did` | `Optional[str]` | `X-Caller-DID` header |
| `target_did` | `Optional[str]` | `X-Target-DID` header |
| `agent_node_did` | `Optional[str]` | `X-Agent-Node-DID` header |
| `workflow_id` | `Optional[str]` | `X-Workflow-ID` header; defaults to `run_id` |
| `parent_workflow_id` | `Optional[str]` | `X-Parent-Workflow-ID` |
| `root_workflow_id` | `Optional[str]` | `X-Root-Workflow-ID` |
| `registered` | `bool` | `True` when built from request headers |

**Construction:**
- From HTTP request: `ExecutionContext.from_request(request, agent_node_id)` (`execution_context.py:174`).
- Fresh root: `ExecutionContext.new_root(agent_node_id, reasoner_name)` (`execution_context.py:223`).
- Child: `ctx.child_context()` / `ctx.create_child_context()` (`execution_context.py:136–168`).

**Context-var propagation:** A module-level `contextvars.ContextVar` (`execution_context.py:259`) stores the current context. `set_execution_context(ctx)` sets it and returns a token; `reset_execution_context(token)` restores previous value. The decorator system sets/resets this around every reasoner call (`decorators.py:228`, `311`).

**Injection into reasoner:** If the decorated function declares `execution_context` as a parameter, it is passed in automatically (`decorators.py:231–232`, `agent.py:1900–1901`).

**Header forwarding:** `ctx.to_headers()` (`execution_context.py:56`) returns the dict of correlation headers to send on downstream calls:
```
X-Run-ID, X-Workflow-ID, X-Parent-Execution-ID, X-Execution-ID,
X-Workflow-Run-ID, X-Agent-Node-ID, X-Session-ID, X-Actor-ID,
X-Parent-Workflow-ID, X-Root-Workflow-ID, X-Caller-DID, X-Target-DID, X-Agent-Node-DID
```

#### 5.2 Python Memory Scopes

**File:** `sdk/python/agentfield/memory.py:1–99` (docstring) and `memory.py:100+` (implementation)

Four scopes, resolved from the `ExecutionContext` at call time:

| Scope name | Retention | Scope-ID source | Access pattern |
|---|---|---|---|
| `global` | Until explicitly deleted | `"global"` literal | `agent.memory.global_scope.set(key, value)` |
| `session` | Until session ends | `execution_context.session_id` | `agent.memory.session(session_id).set(key, value)` |
| `actor` | Across sessions per actor | `execution_context.actor_id` | `agent.memory.actor(actor_id).set(key, value)` |
| `workflow` | Until run completes | `execution_context.workflow_id` | `agent.memory.workflow(workflow_id).set(key, value)` |

Hierarchical lookup (`memory.py:45–56`): `agent.memory.get(key)` without an explicit scope resolves `workflow → session → actor → global`, returning the first match.

The `MemoryClient._build_headers()` (`memory.py:149–168`) merges `execution_context.to_headers()` with scope-override headers before POSTing to `<agentfield_server>/memory/set` or `.../memory/get`.

#### 5.3 Go Memory Scopes

**File:** `sdk/go/agent/memory.go`

```go
const (
    ScopeWorkflow MemoryScope = "workflow"
    ScopeSession  MemoryScope = "session"
    ScopeUser     MemoryScope = "user"
    ScopeGlobal   MemoryScope = "global"
)
```

Default `Memory.Set/Get(ctx, key)` uses `ScopeSession` with scope-ID = `ExecutionContextFrom(ctx).SessionID` (fallback: `RunID`) (`memory.go:80–86`).

Explicit scoped accessors:
- `Memory().WorkflowScope()` → `ScopeWorkflow`, ID from `execCtx.WorkflowID` or `RunID` (`memory.go:196–207`).
- `Memory().SessionScope()` → `ScopeSession` (`memory.go:212`).
- `Memory().UserScope()` → `ScopeUser`, ID from `execCtx.ActorID`.
- `Memory().GlobalScope()` → `ScopeGlobal`, fixed ID `"global"`.

The `ControlPlaneMemoryBackend` (`control_plane_memory_backend.go`) forwards each call as an HTTP request to the CP's `/memory/set` and `/memory/get`, attaching `X-Workflow-ID`, `X-Session-ID`, `X-Actor-ID`, and `X-Agent-Node-ID` headers to scope the data (`control_plane_memory_backend.go:357–385`).

---

### Key Patterns Summary

- **Agent = FastAPI app (Python) / `http.ServeMux` wrapper (Go)**: the same binary serves both the reasoner POST endpoints and the platform routes (`/health`, `/discover`, `/reasoners`, etc.).
- **`@reasoner` or `RegisterReasoner` = route registration + CP metadata**: each call both adds an HTTP route and registers schema metadata that the CP receives at node-registration time.
- **`ExecutionContext` flows via HTTP headers**: the CP injects `X-Run-ID`, `X-Execution-ID`, etc. on inbound calls; the agent rebuilds the context from those headers and propagates them on outbound `Call()` / cross-agent requests.
- **`af run` = `python main.py` with `PORT` and `AGENTFIELD_SERVER_URL` injected**: discovery of the agent's actual port is done by polling `http://localhost:<n>/health`.
- **Memory scopes are header-driven**: the scope-ID for session, actor, and workflow memory is read from the same execution-context headers, so the CP can route storage to the correct shard without the agent needing to carry explicit scope IDs.
- **CLI-wrapping pattern**: a reasoner that shells out to an external CLI tool (`subprocess.run` / `exec.CommandContext`) fits naturally — the `input` dict carries the CLI arguments, `subprocess.run` is called synchronously inside the async handler, and the captured stdout/returncode are returned as the reasoner's output dict.
