"""
Pytest configuration and fixtures for AgentField functional tests.

These fixtures provide integration with the Docker-based test environment,
allowing tests to interact with the control plane and create test agents.
"""

import asyncio
import os
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator, Callable, Dict, Optional

import httpx
import pytest
from agentfield import Agent, AIConfig

from utils import FunctionalTestLogger, InstrumentedAsyncClient

pytest_plugins = ("pytest_asyncio",)

AGENT_BIND_HOST = os.environ.get("TEST_AGENT_BIND_HOST", "127.0.0.1")
AGENT_CALLBACK_HOST = os.environ.get("TEST_AGENT_CALLBACK_HOST", "127.0.0.1")
HTTP_LOGGING_ENABLED = os.environ.get("FUNCTIONAL_HTTP_LOGGING", "1") != "0"

CONFTEST_DIR = Path(__file__).resolve().parent
_SESSION_LOGGER: Optional[FunctionalTestLogger] = None


def _init_session_logger() -> FunctionalTestLogger:
    """Create (or reuse) the global FunctionalTestLogger instance."""
    log_candidates = []
    preferred = os.environ.get("FUNCTIONAL_LOG_FILE")
    if preferred:
        log_candidates.append(Path(preferred).expanduser())
    log_candidates.append(CONFTEST_DIR / "logs" / "functional-tests.log")
    log_candidates.append(Path("/reports/functional-tests.log"))
    log_candidates.append(Path("/tmp/functional-tests.log"))

    max_chars = int(os.environ.get("FUNCTIONAL_LOG_MAX_BODY", "600"))
    retention_seconds = int(os.environ.get("FUNCTIONAL_LOG_RETENTION_SECONDS", "86400"))

    last_error: Optional[OSError] = None
    for candidate in log_candidates:
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            last_error = exc
            continue

        try:
            _prune_old_logs(candidate.parent, retention_seconds)
        except OSError:
            # Best-effort cleanup; don't block logger creation if pruning fails
            pass

        try:
            return FunctionalTestLogger(log_file=candidate, max_body_chars=max_chars)
        except OSError as exc:
            last_error = exc
            continue

    logger = FunctionalTestLogger(log_file=None, max_body_chars=max_chars)
    if last_error:
        logger.log(
            "⚠️ Functional logs will only stream to stdout; "
            f"unable to persist log file ({last_error})."
        )
    return logger


def _get_session_logger() -> FunctionalTestLogger:
    global _SESSION_LOGGER
    if _SESSION_LOGGER is None:
        _SESSION_LOGGER = _init_session_logger()
    return _SESSION_LOGGER


# ============================================================================
# Environment and Configuration Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def control_plane_url() -> str:
    """Get the AgentField control plane URL from environment."""
    url = os.environ.get("AGENTFIELD_SERVER", "http://localhost:8080")
    return url.rstrip("/")


@pytest.fixture(scope="session")
def openrouter_api_key() -> str:
    """Get the OpenRouter API key from environment."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        pytest.skip("OPENROUTER_API_KEY environment variable not set")
    return key


@pytest.fixture(scope="session")
def openrouter_model() -> str:
    """
    Get the OpenRouter model to use for tests from environment.
    
    IMPORTANT: All tests MUST use this fixture and NOT hardcode model names.
    This allows us to use cost-effective models for testing.
    """
    model = os.environ.get("OPENROUTER_MODEL", "openrouter/google/gemini-2.5-flash-lite")
    return model


@pytest.fixture(scope="session")
def storage_mode() -> str:
    """Get the current storage mode being tested."""
    return os.environ.get("STORAGE_MODE", "local")


@pytest.fixture(scope="session")
def test_timeout() -> int:
    """Get the test timeout in seconds."""
    return int(os.environ.get("TEST_TIMEOUT", "300"))


# ============================================================================
# Control Plane Health Check
# ============================================================================

@pytest.fixture(scope="session")
def functional_logger() -> FunctionalTestLogger:
    """Shared structured logger for the entire functional suite."""
    return _get_session_logger()


@pytest.fixture(scope="session", autouse=True)
def verify_control_plane(control_plane_url: str, functional_logger: FunctionalTestLogger):
    """Verify that the control plane is accessible before running tests."""
    health_url = f"{control_plane_url}/api/v1/health"
    max_attempts = 30

    functional_logger.section(f"Verifying control plane at {control_plane_url}")

    for attempt in range(max_attempts):
        try:
            response = httpx.get(health_url, timeout=2.0)
            if response.status_code == 200:
                functional_logger.log(f"✓ Control plane is healthy (attempt {attempt + 1})")
                return
        except (httpx.RequestError, httpx.TimeoutException):
            pass

        if attempt < max_attempts - 1:
            time.sleep(1)

    functional_logger.log("Control plane did not respond to health checks in time")
    pytest.fail(f"Control plane at {control_plane_url} is not responding to health checks")


def _prune_old_logs(directory: Path, retention_seconds: int) -> None:
    """Remove log files older than the configured retention period."""

    if retention_seconds <= 0:
        return

    now = time.time()
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    for log_path in directory.glob("*.log"):
        try:
            if now - log_path.stat().st_mtime > retention_seconds:
                try:
                    log_path.unlink()
                except FileNotFoundError:
                    continue
        except OSError:
            continue


# ============================================================================
# HTTP Client Fixtures
# ============================================================================

@pytest.fixture
async def async_http_client(
    control_plane_url: str,
    functional_logger: FunctionalTestLogger,
) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Provide an async HTTP client configured for the control plane."""
    if HTTP_LOGGING_ENABLED:
        async with InstrumentedAsyncClient(
            logger=functional_logger,
            base_url=control_plane_url,
            timeout=30.0,
            follow_redirects=True,
        ) as client:
            yield client
    else:  # pragma: no cover - fallback path for disabling verbose logging
        async with httpx.AsyncClient(
            base_url=control_plane_url,
            timeout=30.0,
            follow_redirects=True,
        ) as client:
            yield client


# ============================================================================
# AI Configuration Fixtures
# ============================================================================

@pytest.fixture
def openrouter_config(openrouter_api_key: str, openrouter_model: str) -> AIConfig:
    """
    Provide an AIConfig configured for OpenRouter.
    
    IMPORTANT: Uses the OPENROUTER_MODEL environment variable.
    Default model is cost-effective for testing (gemini-2.5-flash-lite).
    DO NOT hardcode model names in tests - always use this fixture.
    """
    return AIConfig(
        model=openrouter_model,
        api_key=openrouter_api_key,
        temperature=0.7,
        max_tokens=500,
        timeout=120.0,
        retry_attempts=3,
    )


# ============================================================================
# Agent Factory Fixtures
# ============================================================================

@pytest.fixture
def make_test_agent(control_plane_url: str) -> Callable[..., Agent]:
    """
    Factory fixture to create test agents.
    
    Returns a callable that creates and configures agents for testing.
    Agents are automatically configured to connect to the control plane.
    
    Usage:
        def test_example(make_test_agent, openrouter_config):
            agent = make_test_agent(
                node_id="test-agent",
                ai_config=openrouter_config
            )
            
            @agent.reasoner()
            async def my_reasoner():
                return {"status": "ok"}
    """
    created_agents = []
    
    def _factory(
        node_id: Optional[str] = None,
        ai_config: Optional[AIConfig] = None,
        **kwargs
    ) -> Agent:
        # Generate unique node ID if not provided
        if node_id is None:
            node_id = f"test-agent-{uuid.uuid4().hex[:8]}"
        
        # Set sensible defaults for testing
        kwargs.setdefault("agentfield_server", control_plane_url)
        kwargs.setdefault("dev_mode", True)
        kwargs.setdefault("callback_url", "http://test-agent")
        
        if ai_config is not None:
            kwargs["ai_config"] = ai_config
        
        agent = Agent(node_id=node_id, **kwargs)
        created_agents.append(agent)
        return agent
    
    yield _factory
    
    # Cleanup: No explicit cleanup needed as agents are ephemeral in tests


@pytest.fixture
async def registered_agent(
    make_test_agent: Callable,
    openrouter_config: AIConfig,
    async_http_client: httpx.AsyncClient
) -> AsyncGenerator[Agent, None]:
    """
    Provide a test agent that is already registered with the control plane.
    
    This is a convenience fixture for tests that need a ready-to-use agent.
    """
    import threading
    import uvicorn
    
    # Create agent
    agent = make_test_agent(ai_config=openrouter_config)
    
    # Add a simple test reasoner
    @agent.reasoner()
    async def echo(message: str) -> Dict[str, str]:
        """Echo back the input message."""
        return {"message": message}
    
    # Find a free port
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((AGENT_BIND_HOST, 0))
        port = s.getsockname()[1]
    
    # Start agent in background thread
    agent.base_url = f"http://{AGENT_CALLBACK_HOST}:{port}"
    
    config = uvicorn.Config(
        app=agent,
        host=AGENT_BIND_HOST,
        port=port,
        log_level="error",
        access_log=False,
        ws="websockets-sansio",
    )
    server = uvicorn.Server(config)
    loop = asyncio.new_event_loop()
    
    def run_server():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    
    # Wait for agent to be ready
    await asyncio.sleep(1)
    
    # Register with control plane
    try:
        await agent.agentfield_handler.register_with_agentfield_server(port)
        agent.agentfield_server = None
        
        # Wait for registration to complete
        await asyncio.sleep(1)
        
        yield agent
    finally:
        # Cleanup
        server.should_exit = True
        if loop.is_running():
            loop.call_soon_threadsafe(lambda: None)
        thread.join(timeout=5)


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_test_input() -> Dict[str, str]:
    """Provide sample test input data."""
    return {
        "prompt": "What is 2 + 2? Reply with just the number.",
        "context": "This is a functional test.",
    }


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "functional: Functional integration tests with real services"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that may take longer to execute"
    )
    config.addinivalue_line(
        "markers", "openrouter: Tests that require OpenRouter API access"
    )
    # Ensure the session logger is ready before tests begin so early logs aren't lost.
    _get_session_logger()


def pytest_runtest_setup(item):
    if _SESSION_LOGGER:
        _SESSION_LOGGER.start_test(item.nodeid)


def pytest_runtest_logreport(report):
    if not _SESSION_LOGGER:
        return

    if report.when == "setup" and report.skipped:
        _SESSION_LOGGER.finish_test(report.nodeid, "SKIPPED")
    elif report.when == "setup" and report.failed:
        _SESSION_LOGGER.finish_test(report.nodeid, "FAILED (setup)")
    elif report.when == "call":
        if report.failed:
            outcome = "FAILED"
        elif report.skipped:
            outcome = "SKIPPED"
        else:
            outcome = "PASSED"
        _SESSION_LOGGER.finish_test(report.nodeid, outcome)
    elif report.when == "teardown" and report.failed:
        _SESSION_LOGGER.finish_test(report.nodeid, "FAILED (teardown)")


def pytest_sessionfinish(session, exitstatus):
    if _SESSION_LOGGER:
        _SESSION_LOGGER.summarize()
