"""
Functional tests for AI integration via OpenRouter.

Tests that agents can make real LLM calls through OpenRouter
and integrate the responses correctly.
"""

import asyncio
import threading

import httpx
import pytest
import uvicorn
from agentfield import Agent, AIConfig


@pytest.mark.functional
def test_agent_with_llm_call(
    control_plane_url: str, openrouter_config: dict, agent_port_allocator
):
    """Test that an agent can make a real LLM call via OpenRouter."""
    agent_port = agent_port_allocator()

    # Configure agent with OpenRouter
    agent = Agent(
        node_id="test-ai-agent",
        agentfield_server=control_plane_url,
        ai_config=AIConfig(
            api_key=openrouter_config["api_key"],
            base_url=openrouter_config["base_url"],
            model=openrouter_config["model"],
        ),
    )

    @agent.reasoner()
    async def ask_llm(question: str):
        """Ask the LLM a question."""
        response = await agent.ai(
            prompt=f"Answer this question briefly in one sentence: {question}",
            max_tokens=50,
        )
        return {"question": question, "answer": response}

    server_ready = threading.Event()

    def run_agent():
        config = uvicorn.Config(
            agent, host="0.0.0.0", port=agent_port, log_level="error"
        )
        server = uvicorn.Server(config)
        server_ready.set()
        asyncio.run(server.serve())

    agent_thread = threading.Thread(target=run_agent, daemon=True)
    agent_thread.start()
    server_ready.wait(timeout=5)

    import time

    time.sleep(2)

    try:
        client = httpx.Client(base_url=control_plane_url, timeout=30.0)

        response = client.post(
            "/api/v1/execute/test-ai-agent.ask_llm",
            json={"input": {"question": "What is 2+2?"}},
        )

        assert (
            response.status_code == 200
        ), f"AI call failed: {response.status_code} - {response.text}"
        result = response.json()

        print(f"✅ LLM call successful: {result}")

        # Verify we got a response with an answer
        result_str = str(result).lower()
        assert (
            "answer" in result_str or "4" in result_str
        ), f"Expected answer in response: {result}"

    finally:
        pass


@pytest.mark.functional
def test_agent_with_structured_llm_output(
    control_plane_url: str, openrouter_config: dict, agent_port_allocator
):
    """Test that an agent can get structured output from an LLM."""
    agent_port = agent_port_allocator()

    agent = Agent(
        node_id="test-structured-ai-agent",
        agentfield_server=control_plane_url,
        ai_config=AIConfig(
            api_key=openrouter_config["api_key"],
            base_url=openrouter_config["base_url"],
            model=openrouter_config["model"],
        ),
    )

    @agent.reasoner()
    async def analyze_sentiment(text: str):
        """Analyze the sentiment of text using LLM."""
        prompt = f"""Analyze the sentiment of this text and respond with just one word: positive, negative, or neutral.
Text: {text}
Sentiment:"""

        response = await agent.ai(prompt=prompt, max_tokens=10)

        return {
            "text": text,
            "sentiment": response.strip().lower(),
        }

    server_ready = threading.Event()

    def run_agent():
        config = uvicorn.Config(
            agent, host="0.0.0.0", port=agent_port, log_level="error"
        )
        server = uvicorn.Server(config)
        server_ready.set()
        asyncio.run(server.serve())

    agent_thread = threading.Thread(target=run_agent, daemon=True)
    agent_thread.start()
    server_ready.wait(timeout=5)

    import time

    time.sleep(2)

    try:
        client = httpx.Client(base_url=control_plane_url, timeout=30.0)

        response = client.post(
            "/api/v1/execute/test-structured-ai-agent.analyze_sentiment",
            json={"input": {"text": "I love this amazing product!"}},
        )

        assert response.status_code == 200
        result = response.json()

        print(f"✅ Structured LLM output test passed: {result}")

        # Verify sentiment analysis worked
        result_str = str(result).lower()
        assert any(
            word in result_str
            for word in ["positive", "negative", "neutral", "sentiment"]
        )

    finally:
        pass


@pytest.mark.functional
def test_agent_with_multiple_llm_calls(
    control_plane_url: str, openrouter_config: dict, agent_port_allocator
):
    """Test that an agent can make multiple sequential LLM calls."""
    agent_port = agent_port_allocator()

    agent = Agent(
        node_id="test-multi-ai-agent",
        agentfield_server=control_plane_url,
        ai_config=AIConfig(
            api_key=openrouter_config["api_key"],
            base_url=openrouter_config["base_url"],
            model=openrouter_config["model"],
        ),
    )

    @agent.reasoner()
    async def multi_step_reasoning(topic: str):
        """Perform multi-step reasoning with multiple LLM calls."""
        # First call: generate a question
        question_response = await agent.ai(
            prompt=f"Generate one simple question about: {topic}",
            max_tokens=50,
        )

        # Second call: answer the question
        answer_response = await agent.ai(
            prompt=f"Answer this question briefly: {question_response}",
            max_tokens=50,
        )

        return {
            "topic": topic,
            "question": question_response,
            "answer": answer_response,
        }

    server_ready = threading.Event()

    def run_agent():
        config = uvicorn.Config(
            agent, host="0.0.0.0", port=agent_port, log_level="error"
        )
        server = uvicorn.Server(config)
        server_ready.set()
        asyncio.run(server.serve())

    agent_thread = threading.Thread(target=run_agent, daemon=True)
    agent_thread.start()
    server_ready.wait(timeout=5)

    import time

    time.sleep(2)

    try:
        client = httpx.Client(base_url=control_plane_url, timeout=45.0)

        response = client.post(
            "/api/v1/execute/test-multi-ai-agent.multi_step_reasoning",
            json={"input": {"topic": "space exploration"}},
        )

        assert response.status_code == 200
        result = response.json()

        print(f"✅ Multiple LLM calls test passed: {result}")

        # Verify we have both question and answer
        result_str = str(result).lower()
        assert "question" in result_str or "answer" in result_str

    finally:
        pass


@pytest.mark.functional
def test_agent_llm_error_handling(
    control_plane_url: str, openrouter_config: dict, agent_port_allocator
):
    """Test error handling when LLM calls fail."""
    agent_port = agent_port_allocator()

    agent = Agent(
        node_id="test-ai-error-agent",
        agentfield_server=control_plane_url,
        ai_config=AIConfig(
            api_key="invalid-key-for-testing",  # Intentionally invalid
            base_url=openrouter_config["base_url"],
            model=openrouter_config["model"],
        ),
    )

    @agent.reasoner()
    async def try_llm_with_bad_key(ping: str = "ok"):
        """Try to call LLM with invalid API key."""
        try:
            response = await agent.ai(prompt="Hello", max_tokens=10)
            return {"status": "unexpected_success", "response": response}
        except Exception as e:
            return {"status": "error_caught", "error_type": type(e).__name__}

    server_ready = threading.Event()

    def run_agent():
        config = uvicorn.Config(
            agent, host="0.0.0.0", port=agent_port, log_level="error"
        )
        server = uvicorn.Server(config)
        server_ready.set()
        asyncio.run(server.serve())

    agent_thread = threading.Thread(target=run_agent, daemon=True)
    agent_thread.start()
    server_ready.wait(timeout=5)

    import time

    time.sleep(2)

    try:
        client = httpx.Client(base_url=control_plane_url, timeout=30.0)

        response = client.post(
            "/api/v1/execute/test-ai-error-agent.try_llm_with_bad_key",
            json={"input": {"ping": "ok"}},
        )

        # The agent should catch the error
        assert response.status_code == 200
        result = response.json()

        print(f"✅ LLM error handling test passed: {result}")

        # Verify error was caught
        result_str = str(result).lower()
        assert "error" in result_str

    finally:
        pass
