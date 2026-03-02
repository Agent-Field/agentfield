"""
Waiting State Agent - Human-in-the-Loop Approval Example

Demonstrates:
- Requesting human approval mid-execution (waiting state)
- Polling for approval status
- Handling approved / rejected / expired decisions
- Using app.pause() for a blocking approval flow inside a reasoner
"""

import os
import uuid

from agentfield import Agent, AIConfig, ApprovalResult

app = Agent(
    node_id="waiting-state-demo",
    agentfield_server=os.getenv("AGENTFIELD_URL", "http://localhost:8080"),
    ai_config=AIConfig(
        model=os.getenv("SMALL_MODEL", "openai/gpt-4o-mini"), temperature=0.7
    ),
)


# ============= SKILL (DETERMINISTIC) =============


@app.skill()
def build_proposal(title: str, description: str) -> dict:
    """Builds a structured proposal document for human review."""
    return {
        "title": title,
        "description": description,
        "proposal_id": f"prop-{uuid.uuid4().hex[:8]}",
        "status": "draft",
    }


# ============= REASONERS (AI-POWERED) =============


@app.reasoner()
async def plan_with_approval(task: str) -> dict:
    """
    Creates a plan and pauses execution for human approval before proceeding.

    Flow:
    1. AI generates a plan for the given task
    2. Execution transitions to "waiting" state
    3. Human reviews and approves/rejects
    4. Execution resumes based on the decision

    This demonstrates the full waiting state lifecycle using app.pause(),
    which blocks until the approval resolves or times out.
    """
    # Step 1: Generate a plan using AI
    plan = await app.ai(
        system="You are a project planner. Create a concise 3-step plan.",
        user=f"Create a plan for: {task}",
    )

    # Step 2: Build the proposal
    proposal = build_proposal(
        title=f"Plan: {task}",
        description=plan.text if hasattr(plan, "text") else str(plan),
    )

    # Step 3: Request human approval — execution pauses here
    # In production, approval_request_id would come from creating a request
    # on an external approval service (e.g. hax-sdk Response Hub).
    approval_request_id = f"req-{uuid.uuid4().hex[:12]}"

    result: ApprovalResult = await app.pause(
        approval_request_id=approval_request_id,
        approval_request_url=f"https://hub.example.com/review/{approval_request_id}",
        expires_in_hours=24,
        timeout=3600,  # Wait up to 1 hour
    )

    # Step 4: Handle the decision
    if result.approved:
        return {
            "status": "approved",
            "proposal": proposal,
            "feedback": result.feedback,
            "message": "Plan approved! Proceeding with execution.",
        }
    elif result.changes_requested:
        return {
            "status": "changes_requested",
            "proposal": proposal,
            "feedback": result.feedback,
            "message": "Changes requested. Revising the plan.",
        }
    else:
        # rejected or expired
        return {
            "status": result.decision,
            "proposal": proposal,
            "feedback": result.feedback,
            "message": f"Plan was {result.decision}. Halting execution.",
        }


@app.reasoner()
async def quick_review(content: str) -> dict:
    """
    Demonstrates the low-level approval API for more control.

    Instead of app.pause() (which blocks), this uses the client methods
    directly to request approval and poll for the result.
    """
    # Request approval via the low-level client API
    approval_request_id = f"req-{uuid.uuid4().hex[:12]}"
    exec_ctx = app._get_current_execution_context()

    await app.client.request_approval(
        execution_id=exec_ctx.execution_id,
        approval_request_id=approval_request_id,
        approval_request_url=f"https://hub.example.com/review/{approval_request_id}",
        expires_in_hours=1,
    )

    # Poll until resolved (with exponential backoff)
    status = await app.client.wait_for_approval(
        execution_id=exec_ctx.execution_id,
        poll_interval=5.0,
        max_interval=30.0,
        timeout=3600,
    )

    return {
        "content_reviewed": content[:100],
        "approval_status": status.status,
        "response": status.response,
    }


# ============= START SERVER OR CLI =============

if __name__ == "__main__":
    print("Waiting State Demo Agent")
    print("Node: waiting-state-demo")
    print("Control Plane: http://localhost:8080")
    print()
    print("Reasoners:")
    print("  - plan_with_approval: Full pause/resume flow with app.pause()")
    print("  - quick_review: Low-level approval API with polling")

    app.run(auto_port=True)
