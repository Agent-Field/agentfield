import asyncio

import httpx
import pytest
from fastapi import APIRouter

from agentfield.router import AgentRouter
from agentfield.decorators import reasoner as tracked_reasoner

from tests.helpers import create_test_agent


@pytest.mark.asyncio
async def test_agent_reasoner_routing_and_workflow(monkeypatch):
    agent, agentfield_client = create_test_agent(
        monkeypatch, callback_url="https://callback.example.com"
    )
    # Disable async execution for this test to get synchronous 200 responses
    agent.async_config.enable_async_execution = False
    # Disable agentfield_server to prevent async callback execution
    agent.agentfield_server = None

    @agent.reasoner()
    async def double(value: int) -> dict:
        memory = agent.memory
        memory_present = memory is not None
        fetched = await memory.get("last", default="missing") if memory else "missing"
        return {
            "value": value * 2,
            "memory_present": memory_present,
            "memory_value": fetched,
        }

    @agent.skill()
    def annotate(text: str) -> str:
        return f"annotated:{text}"

    router = APIRouter()

    @router.get("/status")
    async def status():
        return {"node": agent.node_id}

    agent.include_router(router, prefix="/ops")

    await agent.agentfield_handler.register_with_agentfield_server(port=9100)
    assert agentfield_client.register_calls
    registration = agentfield_client.register_calls[-1]
    assert registration["base_url"] == "https://callback.example.com:9100"
    assert registration["reasoners"][0]["id"] == "double"
    assert registration["skills"][0]["id"] == "annotate"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=agent), base_url="http://test"
    ) as client:
        reasoner_resp = await client.post(
            "/reasoners/double",
            json={"value": 3},
            headers={"x-workflow-id": "wf-123", "x-execution-id": "exec-root"},
        )
        router_resp = await client.get("/ops/status")

    assert reasoner_resp.status_code == 200
    data = reasoner_resp.json()
    assert data["value"] == 6
    assert data["memory_present"] is True
    assert data["memory_value"] == "missing"

    assert router_resp.status_code == 200
    assert router_resp.json() == {"node": agent.node_id}

    await asyncio.sleep(0)
    events = getattr(agent, "_captured_workflow_events", [])
    assert ("start", "exec-root", "double", None) in events
    assert any(evt[0] == "complete" and evt[2] == "double" for evt in events)


@pytest.mark.asyncio
async def test_agent_reasoner_custom_name(monkeypatch):
    agent, _ = create_test_agent(monkeypatch)
    # Disable async execution for this test to get synchronous 200 responses
    agent.async_config.enable_async_execution = False
    # Disable agentfield_server to prevent async callback execution
    agent.agentfield_server = None

    @agent.reasoner(name="reports_generate")
    async def generate_report(report_id: str) -> dict:
        return {"report_id": report_id}

    assert any(r["id"] == "reports_generate" for r in agent.reasoners)
    assert "reports_generate" in agent._reasoner_registry
    assert hasattr(agent, "reports_generate")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=agent), base_url="http://test"
    ) as client:
        response = await client.post(
            "/reasoners/generate_report",
            json={"report_id": "r-123"},
            headers={
                "x-workflow-id": "wf-custom",
                "x-execution-id": "exec-custom",
            },
        )

    assert response.status_code == 200
    assert response.json() == {"report_id": "r-123"}


@pytest.mark.asyncio
async def test_agent_reasoner_without_parentheses(monkeypatch):
    agent, _ = create_test_agent(monkeypatch)
    agent.async_config.enable_async_execution = False
    agent.agentfield_server = None

    @agent.reasoner
    async def greet(name: str) -> dict:
        return {"message": f"hello {name}"}

    assert any(r["id"] == "greet" for r in agent.reasoners)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=agent), base_url="http://test"
    ) as client:
        response = await client.post(
            "/reasoners/greet",
            json={"name": "AgentField"},
            headers={
                "x-workflow-id": "wf-parentheses",
                "x-execution-id": "exec-parentheses",
            },
        )

    assert response.status_code == 200
    assert response.json() == {"message": "hello AgentField"}


@pytest.mark.asyncio
async def test_agent_router_prefix_registration(monkeypatch):
    agent, _ = create_test_agent(monkeypatch)
    # Disable async execution for this test to get synchronous 200 responses
    agent.async_config.enable_async_execution = False
    # Disable agentfield_server to prevent async callback execution
    agent.agentfield_server = None

    quickstart = AgentRouter(prefix="demo")

    @quickstart.reasoner()
    async def hello(name: str) -> dict:
        return {"message": f"hello {name}"}

    @quickstart.skill()
    def repeat(text: str) -> dict:
        return {"echo": text}

    agent.include_router(quickstart)

    assert any(r["id"] == "demo_hello" for r in agent.reasoners)
    assert any(s["id"] == "demo_repeat" for s in agent.skills)
    assert "demo_hello" in agent._reasoner_registry
    assert hasattr(agent, "demo_hello")
    assert hasattr(agent, "demo_repeat")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=agent), base_url="http://test"
    ) as client:
        reasoner_resp = await client.post(
            "/reasoners/demo_hello",
            json={"name": "Agent"},
            headers={"x-workflow-id": "wf-router", "x-execution-id": "exec-router"},
        )

        skill_resp = await client.post(
            "/skills/demo_repeat",
            json={"text": "ping"},
            headers={"x-workflow-id": "wf-router", "x-execution-id": "exec-router"},
        )

    assert reasoner_resp.status_code == 200
    assert reasoner_resp.json() == {"message": "hello Agent"}

    assert skill_resp.status_code == 200
    assert skill_resp.json() == {"echo": "ping"}


@pytest.mark.asyncio
async def test_agent_router_prefix_sanitization(monkeypatch):
    agent, _ = create_test_agent(monkeypatch)

    router = AgentRouter(prefix="/Users/Profile-v1/")

    @router.reasoner()
    async def fetch_order(order_id: int) -> dict:
        return {"order_id": order_id}

    agent.include_router(router)

    assert any(r["id"] == "users_profile_v1_fetch_order" for r in agent.reasoners)
    assert hasattr(agent, "users_profile_v1_fetch_order")


@pytest.mark.parametrize(
    "prefix,func_name,expected_id",
    [
        # Documented examples from documentation
        ("billing", "calculate_cost", "billing_calculate_cost"),
        ("Billing", "calculate_cost", "billing_calculate_cost"),
        ("Support/Inbox", "route_ticket", "support_inbox_route_ticket"),
        ("API/v2/Users", "create_user", "api_v2_users_create_user"),
        ("ML-Models/GPT-4", "generate_text", "ml_models_gpt_4_generate_text"),
        ("Users/Profile-v1", "get_profile", "users_profile_v1_get_profile"),
        ("", "my_function", "my_function"),  # Empty prefix
        # Edge cases
        (
            "/billing/",
            "calculate_cost",
            "billing_calculate_cost",
        ),  # Leading/trailing slashes
        (
            "billing//api",
            "calculate_cost",
            "billing_api_calculate_cost",
        ),  # Multiple slashes
        (
            "API///v2",
            "create_user",
            "api_v2_create_user",
        ),  # Multiple consecutive slashes
        ("test---prefix", "my_func", "test_prefix_my_func"),  # Multiple hyphens
        ("test___prefix", "my_func", "test_prefix_my_func"),  # Multiple underscores
        ("Test@#$%Prefix", "my_func", "test_prefix_my_func"),  # Special characters
        ("123test", "my_func", "123test_my_func"),  # Starts with number
        ("test/123/v2", "my_func", "test_123_v2_my_func"),  # Numbers in path
    ],
)
@pytest.mark.asyncio
async def test_router_prefix_translation_examples(
    monkeypatch, prefix, func_name, expected_id
):
    """Test all documented prefix translation examples and edge cases."""
    agent, _ = create_test_agent(monkeypatch)
    agent.async_config.enable_async_execution = False
    agent.agentfield_server = None

    router = AgentRouter(prefix=prefix)

    # Create a function with the desired name using exec
    func_code = f"""
async def {func_name}() -> dict:
    return {{"result": "ok"}}
"""
    namespace = {}
    exec(func_code, namespace)
    test_func = namespace[func_name]

    # Register the function manually
    router.reasoners.append(
        {
            "func": test_func,
            "path": None,
            "tags": [],
            "kwargs": {},
            "registered": False,
        }
    )

    agent.include_router(router)

    assert any(r["id"] == expected_id for r in agent.reasoners), (
        f"Expected ID '{expected_id}' not found. "
        f"Found IDs: {[r['id'] for r in agent.reasoners]}"
    )
    assert hasattr(agent, expected_id)


@pytest.mark.asyncio
async def test_router_empty_prefix(monkeypatch):
    """Test router with empty prefix - should not add prefix."""
    agent, _ = create_test_agent(monkeypatch)
    agent.async_config.enable_async_execution = False
    agent.agentfield_server = None

    router = AgentRouter(prefix="")

    @router.reasoner()
    async def my_function() -> dict:
        return {"result": "ok"}

    agent.include_router(router)

    assert any(r["id"] == "my_function" for r in agent.reasoners)
    assert hasattr(agent, "my_function")


@pytest.mark.asyncio
async def test_router_include_router_with_additional_prefix(monkeypatch):
    """Test include_router with additional prefix parameter.

    Note: The include_router prefix parameter affects HTTP paths, not function IDs.
    Function IDs only use the router's prefix, not the include_router prefix.
    """
    agent, _ = create_test_agent(monkeypatch)
    agent.async_config.enable_async_execution = False
    agent.agentfield_server = None

    router = AgentRouter(prefix="users")

    @router.reasoner()
    async def get_profile() -> dict:
        return {"profile": "data"}

    # Include router with additional prefix
    # The function ID should only use the router's prefix, not include_router prefix
    agent.include_router(router, prefix="api/v2")

    # Function ID only uses router prefix, not include_router prefix
    assert any(r["id"] == "users_get_profile" for r in agent.reasoners)
    assert hasattr(agent, "users_get_profile")

    # The HTTP path should include the include_router prefix
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=agent), base_url="http://test"
    ) as client:
        response = await client.post(
            "/reasoners/users_get_profile",
            json={},
            headers={"x-workflow-id": "wf-router", "x-execution-id": "exec-router"},
        )

    assert response.status_code == 200
    assert response.json() == {"profile": "data"}


@pytest.mark.asyncio
async def test_router_nested_paths(monkeypatch):
    """Test router with deeply nested paths."""
    agent, _ = create_test_agent(monkeypatch)
    agent.async_config.enable_async_execution = False
    agent.agentfield_server = None

    router = AgentRouter(prefix="api/v1/users/profiles")

    @router.reasoner()
    async def get_data() -> dict:
        return {"data": "test"}

    agent.include_router(router)

    assert any(r["id"] == "api_v1_users_profiles_get_data" for r in agent.reasoners)
    assert hasattr(agent, "api_v1_users_profiles_get_data")


@pytest.mark.asyncio
async def test_router_special_characters_comprehensive(monkeypatch):
    """Test router prefix with various special characters."""
    agent, _ = create_test_agent(monkeypatch)
    agent.async_config.enable_async_execution = False
    agent.agentfield_server = None

    # Test various special character combinations
    test_cases = [
        ("test@domain.com", "test_domain_com"),
        ("test#hash", "test_hash"),
        ("test$dollar", "test_dollar"),
        ("test%percent", "test_percent"),
        ("test&and", "test_and"),
        ("test*star", "test_star"),
        ("test+plus", "test_plus"),
        ("test=equals", "test_equals"),
        ("test?question", "test_question"),
        ("test[open", "test_open"),
        ("test]close", "test_close"),
        ("test|pipe", "test_pipe"),
        ("test\\backslash", "test_backslash"),
        ("test^caret", "test_caret"),
        ("test~tilde", "test_tilde"),
        ("test`backtick", "test_backtick"),
        ("test{brace", "test_brace"),
        ("test}close", "test_close"),
    ]

    for prefix, expected_segment in test_cases:
        router = AgentRouter(prefix=prefix)

        @router.reasoner()
        async def test_func() -> dict:
            return {"result": "ok"}

        agent.include_router(router)

        expected_id = f"{expected_segment}_test_func"
        assert any(
            r["id"] == expected_id for r in agent.reasoners
        ), f"Failed for prefix '{prefix}': expected '{expected_id}'"


@pytest.mark.asyncio
async def test_router_skill_with_prefix(monkeypatch):
    """Test router skill registration with prefix."""
    agent, _ = create_test_agent(monkeypatch)
    agent.async_config.enable_async_execution = False
    agent.agentfield_server = None

    router = AgentRouter(prefix="billing")

    @router.skill()
    def calculate_cost(amount: float) -> dict:
        return {"cost": amount * 1.1}

    agent.include_router(router)

    assert any(s["id"] == "billing_calculate_cost" for s in agent.skills)
    assert hasattr(agent, "billing_calculate_cost")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=agent), base_url="http://test"
    ) as client:
        response = await client.post(
            "/skills/billing_calculate_cost",
            json={"amount": 100.0},
            headers={"x-workflow-id": "wf-skill", "x-execution-id": "exec-skill"},
        )

    assert response.status_code == 200
    # Handle floating point precision
    result = response.json()
    assert abs(result["cost"] - 110.0) < 0.0001


@pytest.mark.asyncio
async def test_agent_skill_without_parentheses(monkeypatch):
    agent, _ = create_test_agent(monkeypatch)
    agent.async_config.enable_async_execution = False
    agent.agentfield_server = None

    @agent.skill
    def shout(text: str) -> dict:
        return {"value": text.upper()}

    assert any(s["id"] == "shout" for s in agent.skills)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=agent), base_url="http://test"
    ) as client:
        response = await client.post(
            "/skills/shout",
            json={"text": "agentfield"},
            headers={
                "x-workflow-id": "wf-skill",
                "x-execution-id": "exec-skill",
            },
        )

    assert response.status_code == 200
    assert response.json() == {"value": "AGENTFIELD"}


@pytest.mark.asyncio
async def test_reasoner_tags_propagate_to_metadata(monkeypatch):
    agent, _ = create_test_agent(monkeypatch)
    agent.async_config.enable_async_execution = False
    agent.agentfield_server = None

    @agent.reasoner(tags=["ai", "personalization"])
    async def personalize(name: str) -> dict:
        return {"greeting": name}

    tagged_reasoner = next(r for r in agent.reasoners if r["id"] == "personalize")
    assert tagged_reasoner["tags"] == ["ai", "personalization"]

    @agent.reasoner()
    @tracked_reasoner(tags=["decorated"])
    async def decorated_reasoner(topic: str) -> dict:
        return {"topic": topic}

    decorated_meta = next(r for r in agent.reasoners if r["id"] == "decorated_reasoner")
    assert decorated_meta["tags"] == ["decorated"]

    router = AgentRouter(prefix="suite", tags=["router"])

    @router.reasoner(tags=["local"])
    async def hello(name: str) -> dict:
        return {"message": name}

    agent.include_router(router, tags=["include"])
    router_meta = next(r for r in agent.reasoners if r["id"] == "suite_hello")
    assert router_meta["tags"] == ["include", "router", "local"]


@pytest.mark.asyncio
async def test_callback_url_precedence_and_env(monkeypatch):
    monkeypatch.setenv("AGENT_CALLBACK_URL", "https://env.example.com")

    explicit_agent, explicit_client = create_test_agent(
        monkeypatch, callback_url="https://explicit.example.com"
    )
    await explicit_agent.agentfield_handler.register_with_agentfield_server(port=9200)
    assert explicit_agent.base_url == "https://explicit.example.com:9200"
    assert (
        explicit_client.register_calls[-1]["base_url"]
        == "https://explicit.example.com:9200"
    )

    env_agent, env_client = create_test_agent(monkeypatch)
    await env_agent.agentfield_handler.register_with_agentfield_server(port=9300)
    assert env_agent.base_url == "https://env.example.com:9300"
    assert env_client.register_calls[-1]["base_url"] == "https://env.example.com:9300"
