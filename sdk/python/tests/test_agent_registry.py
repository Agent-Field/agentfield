from agentfield.agent_registry import (
    set_current_agent,
    get_current_agent_instance,
    clear_current_agent,
)


class DummyAgent:
    pass


def test_agent_registry_roundtrip():
    clear_current_agent()
    assert get_current_agent_instance() is None

    agent = DummyAgent()
    set_current_agent(agent)
    assert get_current_agent_instance() is agent

    clear_current_agent()
    assert get_current_agent_instance() is None
