# AgentMesh (Python)

`AgentMesh` hosts several Python agents in one process and mounts them on one
FastAPI/uvicorn port. Calls between members are dispatched through the target
agent's ASGI application, so its normal validation, execution context, usage,
workflow event, trigger, and response handling still run.

```python
import asyncio
from agentfield import Agent, AgentMesh

writer = Agent(node_id="writer")
editor = Agent(node_id="editor")

@editor.reasoner()
async def revise(text: str) -> dict:
    return {"text": text.strip()}

@writer.reasoner()
async def draft(topic: str) -> dict:
    return await writer.call("editor.revise", text=f"Draft about {topic}")

mesh = AgentMesh([writer, editor])
mesh.run(port=8000)
```

The mounted routes include `/writer/reasoners/draft` and
`/editor/reasoners/revise`; each member's health endpoint is under its mount,
and the root `/health` lists all members. Custom reasoner `path=` values are
recorded in the registry and used by mesh dispatch.

For explicit in-process invocation without a mesh or control plane, use
`await agent.call_local("reasoner-name", value="x")`. This is the Python
counterpart of the Go SDK's `CallLocal`.

## Limitations (v1)

- AgentMesh is offline only. It sets every member's `auto_register` to `False`.
  `AgentMesh(register=True)` raises `NotImplementedError` because the control
  plane strips the mount prefix from a callback URL path and cannot route back
  to a mounted member. Unknown nodes raise `MeshTargetNotFound`; there is no
  control-plane fallthrough. Members never connect, so such a fallthrough would
  only produce the misleading server-unavailable error.
- Mesh calls do not carry DID signatures. An agent configured with
  `local_verification=True` will therefore reject its own mesh traffic with 401.
- `Agent.get_current()` and the `set_current_agent` context variable are
  last-writer-wins when several agents share a process. Target resolution uses
  the mesh registry, never either ambient current-agent mechanism.
- The connection manager and memory-event client do not connect: the mesh does
  not run `AgentServer.serve()`'s resilient startup lifecycle.
- Header-transported child executions have `depth == 0` because
  `ExecutionContext.to_headers()` does not emit depth and `from_request()` does
  not read it. This matches the control-plane HTTP path.
- Unknown nodes or members raise `MeshTargetNotFound`; validation and application
  HTTP failures preserve their status in `ExecuteError`; an unhandled reasoner
  exception becomes `ExecutionFailedError`, with the original exception available
  as `__cause__`.
- The mesh adds no runtime dependency. Dispatch speaks raw ASGI directly.
