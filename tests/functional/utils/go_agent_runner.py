import asyncio
import os
import shutil
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Iterable, Mapping, Optional


@dataclass
class GoAgentProcess:
    name: str
    process: asyncio.subprocess.Process


def get_go_agent_binary(agent_name: str) -> str:
    binary_name = f"go-agent-{agent_name}"
    path = shutil.which(binary_name)
    if not path:
        raise FileNotFoundError(f"{binary_name} not found on PATH")
    return path


@asynccontextmanager
async def run_go_agent(
    agent_name: str,
    args: Optional[Iterable[str]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> AsyncIterator[GoAgentProcess]:
    """
    Launch a Go agent binary as an async subprocess and ensure it is terminated
    when the context exits.
    """
    binary = get_go_agent_binary(agent_name)
    cmd = [binary]
    if args:
        cmd.extend(args)

    merged_env = {**os.environ}
    if env:
        merged_env.update(env)

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=merged_env,
    )

    proc_info = GoAgentProcess(name=agent_name, process=process)
    try:
        yield proc_info
    finally:
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                process.kill()
