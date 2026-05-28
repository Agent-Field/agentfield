"""
Permission Agent B (Protected Target)

A protected agent that demonstrates tag-based authorization with access policies.
Tags: ["sensitive", "data-service", "payments"]

The "sensitive" tag triggers manual approval (tag_approval_rules in config),
so this agent starts in "pending_approval" state until an admin approves its tags.

Once approved, access policies control which callers can invoke which reasoners:
  - analytics callers can call query_data and get_schema (allowed by policy)
  - analytics callers are denied delete_records (deny_functions in policy)
  - constraint violations (e.g. limit > 1000) are rejected

Reasoners:
  - query_data(query, limit)  — simulates a data query (allowed for analytics)
  - delete_records(table)     — simulates record deletion (denied for analytics)
  - process_payment(amount, currency) — simulates payment processing
"""

from agentfield import Agent
import os

app = Agent(
    node_id="permission-agent-b",
    agentfield_server=os.getenv("AGENTFIELD_URL", "http://localhost:8080"),
    tags=["sensitive", "data-service", "payments"],
    enable_did=True,
    vc_enabled=True,
)


@app.skill(tags=["data-service", "sensitive"])
def query_data(query: str = "SELECT *", limit: int = 100) -> dict:
    """Execute a data query. Protected by access policy — analytics callers allowed."""
    return {
        "status": "success",
        "agent": "permission-agent-b",
        "query": query,
        "limit": limit,
        "results": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
        "message": f"Query executed: {query} (limit={limit})",
    }


@app.skill(tags=["data-service"])
def delete_records(table: str = "records") -> dict:
    """Delete records from a table. Denied for analytics callers by policy."""
    return {
        "status": "deleted",
        "agent": "permission-agent-b",
        "table": table,
        "message": f"Records deleted from {table}",
    }


@app.skill(tags=["payments", "financial"])
def process_payment(amount: float, currency: str = "USD") -> dict:
    """Process a payment. Protected operation."""
    return {
        "status": "processed",
        "amount": amount,
        "currency": currency,
        "agent": "permission-agent-b",
        "message": f"Payment of {amount} {currency} processed successfully",
    }


if __name__ == "__main__":
    print("Permission Agent B (Protected Target)")
    print("Node: permission-agent-b")
    print("Tags: sensitive, data-service, payments")
    print("Reasoners: query_data, delete_records, process_payment")
    app.run(auto_port=True)
