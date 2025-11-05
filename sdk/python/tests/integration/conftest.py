from __future__ import annotations

import asyncio
import os
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Generator, Optional

import pytest
import requests
import uvicorn

if TYPE_CHECKING:
    from agentfield.agent import Agent


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _write_agentfield_config(config_path: Path, db_path: Path, kv_path: Path) -> None:
    db_uri = db_path.as_posix()
    kv_uri = kv_path.as_posix()
    config_content = f"""
agentfield:
  port: 0
  mode: "local"
  request_timeout: 60s
  circuit_breaker_threshold: 5
ui:
  enabled: false
  mode: "embedded"
storage:
  mode: "local"
  local:
    database_path: "{db_uri}"
    kv_store_path: "{kv_uri}"
    cache_size: 128
    retention_days: 7
    auto_vacuum: true
features:
  did:
    enabled: false
agents:
  discovery:
    scan_interval: "5m"
    health_check_interval: "5m"
""".strip()
    config_path.write_text(config_content)


@dataclass
class AgentFieldServerInfo:
    base_url: str
    port: int
    agentfield_home: Path


@pytest.fixture(scope="session")
def agentfield_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    agentfield_go_root = repo_root / "apps" / "platform" / "agentfield"
    if not agentfield_go_root.exists():
        pytest.skip("AgentField server sources not available in this checkout")
    build_dir = tmp_path_factory.mktemp("agentfield-server-bin")
    binary_name = "agentfield-test-server.exe" if os.name == "nt" else "agentfield-test-server"
    binary_path = build_dir / binary_name

    releases_dir = agentfield_go_root / "dist" / "releases"
    os_part = sys.platform
    if os_part.startswith("darwin"):
        os_part = "darwin"
    elif os_part.startswith("linux"):
        os_part = "linux"
    else:
        os_part = None

    arch = platform.machine().lower()
    arch_map = {
        "x86_64": "amd64",
        "amd64": "amd64",
        "arm64": "arm64",
        "aarch64": "arm64",
    }
    arch_part = arch_map.get(arch, arch)

    prebuilt_path: Optional[Path] = None
    if os_part:
        candidate = releases_dir / f"agentfield-{os_part}-{arch_part}"
        if candidate.exists():
            prebuilt_path = candidate
        elif os_part == "darwin":
            universal = releases_dir / "agentfield-darwin-arm64"
            if universal.exists():
                prebuilt_path = universal

    if prebuilt_path is not None:
        shutil.copy(prebuilt_path, binary_path)
        binary_path.chmod(0o755)
        return binary_path

    build_cmd = ["go", "build", "-o", str(binary_path), "./cmd/agentfield"]
    env = os.environ.copy()
    env["GOCACHE"] = str(tmp_path_factory.mktemp("go-cache"))
    env["GOMODCACHE"] = str(tmp_path_factory.mktemp("go-modcache"))
    subprocess.run(build_cmd, check=True, cwd=agentfield_go_root, env=env)
    return binary_path


@pytest.fixture
def agentfield_server(
    tmp_path_factory: pytest.TempPathFactory, agentfield_binary: Path
) -> Generator[AgentFieldServerInfo, None, None]:
    repo_root = Path(__file__).resolve().parents[4]
    agentfield_go_root = repo_root / "apps" / "platform" / "agentfield"

    agentfield_home = Path(tmp_path_factory.mktemp("agentfield-home"))
    data_dir = agentfield_home / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    db_path = data_dir / "agentfield.db"
    kv_path = data_dir / "agentfield.bolt"
    config_path = agentfield_home / "agentfield.yaml"

    _write_agentfield_config(config_path, db_path, kv_path)

    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"

    env = os.environ.copy()
    env.update(
        {
            "AGENTFIELD_HOME": str(agentfield_home),
            "AGENTFIELD_STORAGE_MODE": "local",
        }
    )

    cmd = [
        str(agentfield_binary),
        "server",
        "--backend-only",
        "--port",
        str(port),
        "--config",
        str(config_path),
        "--no-vc-execution",
    ]

    log_path = agentfield_home / "agentfield.log"
    log_file = log_path.open("w")

    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=agentfield_go_root,
    )

    try:
        health_url = f"{base_url}/api/v1/health"
        deadline = time.time() + 60
        while time.time() < deadline:
            if process.poll() is not None:
                raise RuntimeError("AgentField server exited before becoming healthy")
            try:
                response = requests.get(health_url, timeout=1.0)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError("AgentField server did not become healthy in time")

        yield AgentFieldServerInfo(base_url=base_url, port=port, agentfield_home=agentfield_home)

    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        log_file.close()


@dataclass
class AgentRuntime:
    agent: Agent
    base_url: str
    port: int
    server: uvicorn.Server
    loop: asyncio.AbstractEventLoop
    thread: threading.Thread


@pytest.fixture
def run_agent() -> Generator[
    Callable[[Agent, Optional[int]], AgentRuntime], None, None
]:
    runtimes: list[AgentRuntime] = []

    def _start(agent: Agent, port: Optional[int] = None) -> AgentRuntime:
        assigned_port = port or _find_free_port()
        base_url = f"http://127.0.0.1:{assigned_port}"
        agent.base_url = base_url

        config = uvicorn.Config(
            app=agent,
            host="127.0.0.1",
            port=assigned_port,
            log_level="warning",
            access_log=False,
            loop="asyncio",
        )
        server = uvicorn.Server(config)
        loop = asyncio.new_event_loop()

        def _run() -> None:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(server.serve())
            loop.close()

        thread = threading.Thread(
            target=_run, name=f"uvicorn-{assigned_port}", daemon=True
        )
        thread.start()

        health_url = f"{base_url}/health"
        deadline = time.time() + 30
        while time.time() < deadline:
            if not thread.is_alive():
                raise RuntimeError("Agent server exited during startup")
            try:
                resp = requests.get(health_url, timeout=1.0)
                if resp.status_code < 500:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.2)
        else:
            raise RuntimeError("Agent server health endpoint unavailable")

        runtime = AgentRuntime(
            agent=agent,
            base_url=base_url,
            port=assigned_port,
            server=server,
            loop=loop,
            thread=thread,
        )
        runtimes.append(runtime)
        return runtime

    try:
        yield _start
    finally:
        for runtime in reversed(runtimes):
            runtime.server.should_exit = True
            if runtime.loop.is_running():
                runtime.loop.call_soon_threadsafe(lambda: None)
            runtime.thread.join(timeout=10)
            if runtime.thread.is_alive():
                runtime.server.force_exit = True
                if runtime.loop.is_running():
                    runtime.loop.call_soon_threadsafe(lambda: None)
                runtime.thread.join(timeout=5)
