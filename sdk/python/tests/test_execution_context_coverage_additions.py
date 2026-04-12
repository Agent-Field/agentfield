from types import SimpleNamespace

from agentfield.execution_context import ExecutionContext


def test_to_log_attributes_includes_actor_id():
    context = ExecutionContext(
        run_id="run-1",
        execution_id="exec-1",
        agent_instance=SimpleNamespace(node_id="node-1"),
        reasoner_name="summarize",
        actor_id="actor-1",
    )

    attributes = context.to_log_attributes()

    assert attributes["actor_id"] == "actor-1"
