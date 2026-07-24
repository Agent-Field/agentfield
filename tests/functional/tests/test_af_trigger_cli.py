import asyncio
import json
import os
import shutil

import pytest

from utils import run_agent_server, unique_node_id


async def _run_af(*args: str, input_text: str | None = None):
    env = {**os.environ}
    proc = await asyncio.create_subprocess_exec(
        "af",
        *args,
        stdin=asyncio.subprocess.PIPE if input_text is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    stdout, stderr = await proc.communicate(
        input_text.encode("utf-8") if input_text is not None else None
    )
    return proc.returncode, stdout.decode("utf-8"), stderr.decode("utf-8")


@pytest.mark.functional
@pytest.mark.asyncio
async def test_af_call_ls_tail_cli_trigger_surface(make_test_agent):
    if shutil.which("af") is None:
        pytest.skip("af binary not available in test runner")

    agent = make_test_agent(node_id=unique_node_id("af-trigger-cli"))

    @agent.reasoner()
    async def echo(message: str) -> dict:
        return {"echo": message, "length": len(message)}

    async with run_agent_server(agent):
        target = f"{agent.node_id}.echo"

        rc, stdout, stderr = await _run_af(
            "call",
            target,
            "--in",
            '{"message":"from inline"}',
            "-o",
            "json",
            "--field",
            ".echo",
        )
        assert rc == 0, stderr
        assert json.loads(stdout) == "from inline"

        rc, stdout, stderr = await _run_af(
            "call",
            target,
            "-o",
            "json",
            input_text='{"message":"from stdin"}',
        )
        assert rc == 0, stderr
        assert json.loads(stdout)["echo"] == "from stdin"

        rc, stdout, stderr = await _run_af("call", target, "--schema", "-o", "json")
        assert rc == 0, stderr
        schema = json.loads(stdout)
        assert "message" in schema.get("properties", {})

        rc, stdout, stderr = await _run_af("ls", "echo", "--node", agent.node_id, "-o", "json")
        assert rc == 0, stderr
        listing = json.loads(stdout)
        assert any(
            row["node"] == agent.node_id and row["reasoner"] == "echo"
            for row in listing["reasoners"]
        )

        rc, stdout, stderr = await _run_af(
            "call",
            target,
            "--in",
            '{"message":"async tail"}',
            "--async",
        )
        assert rc == 0, stderr
        run_id = stdout.strip()
        assert run_id

        rc, stdout, stderr = await _run_af("tail", run_id, "-o", "json")
        assert rc == 0, stderr
        events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
        assert events
        assert any(event.get("status") == "succeeded" for event in events)
