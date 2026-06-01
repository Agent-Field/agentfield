"""
Serverless Hello World (Python)

Deploy-friendly example that exposes a single handler for serverless
runtimes (AWS Lambda, Cloud Functions, jobs, etc.) while still supporting
local execution for smoke tests.
"""

import os
from agentfield import Agent
from agentfield.async_config import AsyncConfig


# Minimal agent with no heartbeat/lease loop. The control plane discovers
# capabilities via /discover and invokes via /execute.
app = Agent(
    node_id=os.getenv("AGENT_NODE_ID", "python-serverless-hello"),
    agentfield_server=os.getenv("AGENTFIELD_URL", "http://localhost:8080"),
    auto_register=False,
    dev_mode=True,
    async_config=AsyncConfig(enable_async_execution=False, fallback_to_sync=True),
)


@app.reasoner()
async def hello(name: str = "AgentField") -> dict:
    """Return a greeting and echo execution metadata for debugging."""
    ctx = app.ctx
    return {
        "greeting": f"Hello, {name}!",
        "run_id": getattr(ctx, "workflow_id", None),
        "execution_id": getattr(ctx, "execution_id", None),
    }


@app.reasoner()
async def ping_child(target: str, message: str) -> dict:
    """
    Optional cross-agent call to prove parent/child wiring works in
    serverless mode. Set TARGET_NODE to something like "child-agent.echo".
    """
    downstream = await app.call(target, name=message)
    return {"from": app.node_id, "child": downstream}


def handler(event, context=None):
    """
    Universal handler for serverless platforms.

    Example (AWS Lambda):
        def lambda_handler(event, context):
            return handler(event, context)
    """
    return app.handle_serverless(event)


if __name__ == "__main__":
    # Lightweight local runner for manual testing (no heartbeats).
    import asyncio
    import json
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    api = FastAPI()

    @api.get("/discover")
    async def discover():
        return await asyncio.to_thread(app.handle_serverless, {"path": "/discover"})

    @api.post("/execute")
    async def execute(request: Request):
        payload = await request.json()
        result = await asyncio.to_thread(app.handle_serverless, {"path": "/execute", **payload})
        status = result.get("statusCode", 200)
        body = result.get("body", result)
        return JSONResponse(content=json.loads(json.dumps(body)), status_code=status)

    uvicorn.run(api, host="0.0.0.0", port=int(os.getenv("PORT", "8081")))
