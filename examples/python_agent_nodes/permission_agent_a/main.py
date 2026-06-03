"""
Permission Agent A (Caller)

An agent with tag "analytics" that demonstrates the policy engine:
  - call_query_data  -> calls permission-agent-b.query_data (ALLOWED by policy)
  - call_query_large -> calls permission-agent-b.query_data with limit=5000 (DENIED: constraint violation)
  - call_delete      -> calls permission-agent-b.delete_records (DENIED: deny_functions)

The "analytics" tag auto-approves (tag_approval_rules), so this agent starts
immediately in "active" state.

Test flow:
  1. Start control plane with authorization enabled
  2. Start permission-agent-b -> enters pending_approval
  3. Admin approves permission-agent-b's tags
  4. Start permission-agent-a (this agent) -> auto-approved
  5. POST /api/v1/execute/permission-agent-a.call_query_data -> 200 OK
  6. POST /api/v1/execute/permission-agent-a.call_query_large -> 403 constraint
  7. POST /api/v1/execute/permission-agent-a.call_delete -> 403 denied function
"""

from agentfield import Agent
import os

app = Agent(
    node_id="permission-agent-a",
    agentfield_server=os.getenv("AGENTFIELD_URL", "http://localhost:8080"),
    tags=["analytics"],
    enable_did=True,
    vc_enabled=True,
)


@app.skill()
def ping() -> dict:
    """Simple health check - no cross-agent call, should always work."""
    return {"status": "ok", "agent": "permission-agent-a"}


@app.reasoner()
async def call_query_data(query: str = "SELECT * FROM data") -> dict:
    """
    Calls permission-agent-b.query_data with a small limit.
    Should succeed: analytics -> data-service, query_* is in allow_functions,
    limit=100 satisfies the <= 1000 constraint.
    """
    result = await app.call(
        "permission-agent-b.query_data",
        query=query,
        limit=100,
    )
    return {
        "source": "permission-agent-a",
        "test": "allowed_query",
        "delegation_result": result,
    }


@app.reasoner()
async def call_query_large(query: str = "SELECT * FROM big_table") -> dict:
    """
    Calls permission-agent-b.query_data with limit=5000.
    Should fail: limit=5000 violates the <= 1000 constraint.
    """
    result = await app.call(
        "permission-agent-b.query_data",
        query=query,
        limit=5000,
    )
    return {
        "source": "permission-agent-a",
        "test": "constraint_violation",
        "delegation_result": result,
    }


@app.reasoner()
async def call_delete(table: str = "sensitive_records") -> dict:
    """
    Calls permission-agent-b.delete_records.
    Should fail: delete_* is in deny_functions for the analytics->data-service policy.
    """
    result = await app.call(
        "permission-agent-b.delete_records",
        table=table,
    )
    return {
        "source": "permission-agent-a",
        "test": "deny_function",
        "delegation_result": result,
    }


if __name__ == "__main__":
    print("Permission Agent A (Caller)")
    print("Node: permission-agent-a")
    print("Tags: analytics")
    print("Test reasoners: call_query_data (allow), call_query_large (constraint), call_delete (deny)")
    app.run(auto_port=True)
