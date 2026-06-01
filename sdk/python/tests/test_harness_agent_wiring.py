import pytest
from unittest.mock import AsyncMock, patch
from agentfield.agent import Agent
from agentfield.types import HarnessConfig
from agentfield.harness._runner import HarnessRunner
from agentfield.harness._result import HarnessResult


class TestAgentHarnessConfig:
    def test_agent_accepts_harness_config(self):
        config = HarnessConfig(provider="claude-code")
        agent = Agent(node_id="test-agent", harness_config=config, auto_register=False)
        assert agent.harness_config is config

    def test_agent_harness_config_default_none(self):
        agent = Agent(node_id="test-agent", auto_register=False)
        assert agent.harness_config is None

    def test_harness_runner_lazy_init(self):
        config = HarnessConfig(provider="claude-code")
        agent = Agent(node_id="test-agent", harness_config=config, auto_register=False)
        assert agent._harness_runner is None
        runner = agent.harness_runner
        assert runner is not None
        assert isinstance(runner, HarnessRunner)
        assert agent.harness_runner is runner

    def test_harness_runner_without_config(self):
        agent = Agent(node_id="test-agent", auto_register=False)
        runner = agent.harness_runner
        assert isinstance(runner, HarnessRunner)


class TestAgentHarnessMethod:
    @pytest.mark.asyncio
    async def test_harness_delegates_to_runner(self):
        config = HarnessConfig(provider="claude-code")
        agent = Agent(node_id="test-agent", harness_config=config, auto_register=False)

        mock_result = HarnessResult(
            result="test output",
            parsed=None,
            is_error=False,
            num_turns=1,
            duration_ms=100,
            session_id="sess-1",
            messages=[],
        )

        with patch.object(
            HarnessRunner, "run", new_callable=AsyncMock, return_value=mock_result
        ) as mock_run:
            result = await agent.harness("Do something", provider="claude-code")
            assert result is mock_result
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args
            assert call_kwargs[0][0] == "Do something"
            assert call_kwargs[1]["provider"] == "claude-code"

    @pytest.mark.asyncio
    async def test_harness_passes_all_options(self):
        agent = Agent(node_id="test-agent", auto_register=False)

        mock_result = HarnessResult(
            result="ok",
            parsed=None,
            is_error=False,
            num_turns=1,
            duration_ms=50,
            session_id="s",
            messages=[],
        )

        with patch.object(
            HarnessRunner, "run", new_callable=AsyncMock, return_value=mock_result
        ) as mock_run:
            await agent.harness(
                "task",
                provider="codex",
                model="o3",
                max_turns=10,
                max_budget_usd=5.0,
                tools=["Read"],
                permission_mode="auto",
                system_prompt="Be helpful",
                env={"FOO": "bar"},
                cwd="/tmp",
            )
            _, kwargs = mock_run.call_args
            assert kwargs["provider"] == "codex"
            assert kwargs["model"] == "o3"
            assert kwargs["max_turns"] == 10
            assert kwargs["max_budget_usd"] == 5.0
            assert kwargs["tools"] == ["Read"]
            assert kwargs["permission_mode"] == "auto"
            assert kwargs["system_prompt"] == "Be helpful"
            assert kwargs["env"] == {"FOO": "bar"}
            assert kwargs["cwd"] == "/tmp"


class TestAgentInitExports:
    def test_harness_config_importable(self):
        from agentfield import HarnessConfig

        assert HarnessConfig is not None

    def test_harness_result_importable(self):
        from agentfield import HarnessResult

        assert HarnessResult is not None
