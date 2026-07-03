"""
Pytest configuration and fixtures for functional tests.

Functional tests validate end-to-end behavior against a real control plane
and PostgreSQL database running in Docker.
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Generator

import httpx
import pytest


# Path to test-infra directory (relative to this file)
TEST_INFRA_DIR = Path(__file__).parent.parent.parent.parent.parent / "test-infra"


@pytest.fixture(scope="session")
def test_environment() -> Generator[str, None, None]:
    """
    Start the test infrastructure (Docker Compose environment).

    This fixture:
    1. Starts Docker Compose with control plane and PostgreSQL
    2. Waits for services to be healthy
    3. Returns the control plane URL
    4. Cleans up on test session end

    Yields:
        str: Control plane URL (http://localhost:8080)
    """
    print("\n" + "=" * 70)
    print("🚀 Starting test environment...")
    print("=" * 70)

    # Path to start script
    start_script = TEST_INFRA_DIR / "scripts" / "start-env.sh"

    if not start_script.exists():
        pytest.fail(
            f"Test infrastructure not found at {TEST_INFRA_DIR}. "
            "Please ensure test-infra/ directory exists."
        )

    try:
        # Start infrastructure
        result = subprocess.run(
            [str(start_script)],
            cwd=TEST_INFRA_DIR,
            check=True,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        print(result.stdout)

        control_plane_url = os.getenv("AGENTFIELD_SERVER", "http://localhost:8080")
        print(f"\n✅ Test environment ready at {control_plane_url}")
        print("=" * 70 + "\n")

        yield control_plane_url

    except subprocess.CalledProcessError as e:
        print("\n❌ Failed to start test environment:")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        pytest.fail(f"Failed to start test environment: {e}")

    finally:
        # Cleanup
        print("\n" + "=" * 70)
        print("🛑 Stopping test environment...")
        print("=" * 70)

        stop_script = TEST_INFRA_DIR / "scripts" / "stop-env.sh"
        try:
            result = subprocess.run(
                [str(stop_script)],
                cwd=TEST_INFRA_DIR,
                check=True,
                capture_output=True,
                text=True,
            )
            print(result.stdout)
            print("=" * 70 + "\n")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Warning: Cleanup failed: {e}")


@pytest.fixture
def control_plane_url(test_environment: str) -> str:
    """
    Get the control plane URL.

    Args:
        test_environment: The test environment fixture

    Returns:
        str: Control plane URL
    """
    return test_environment


@pytest.fixture
def control_plane_client(control_plane_url: str) -> httpx.Client:
    """
    Create an HTTP client for the control plane.

    Args:
        control_plane_url: Control plane base URL

    Returns:
        httpx.Client: Configured HTTP client
    """
    return httpx.Client(base_url=control_plane_url, timeout=10.0)


@pytest.fixture
def openrouter_config() -> dict[str, str]:
    """
    Get OpenRouter configuration from environment.

    Returns:
        dict: OpenRouter configuration with api_key, base_url, and model

    Raises:
        pytest.skip: If OPENROUTER_API_KEY is not set
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip(
            "OPENROUTER_API_KEY not set. "
            "Set it in test-infra/.env.test or as an environment variable."
        )

    return {
        "api_key": api_key,
        "base_url": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        "model": os.getenv("OPENROUTER_MODEL", "google/gemini-flash-1.5"),
    }


@pytest.fixture
def agent_port_allocator():
    """
    Allocate unique ports for test agents.

    Returns a function that allocates sequential ports starting from 9000.
    This prevents port conflicts when running multiple agents in tests.
    """
    port_counter = {"current": 9000}

    def allocate_port() -> int:
        """Get the next available port."""
        port = port_counter["current"]
        port_counter["current"] += 1
        return port

    return allocate_port


@pytest.fixture(autouse=True)
def wait_between_tests():
    """
    Add a small delay between tests to avoid race conditions.

    This helps ensure agents have fully shut down before the next test starts.
    """
    yield
    time.sleep(0.1)  # 100ms delay between tests
