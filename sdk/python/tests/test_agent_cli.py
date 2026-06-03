"""
Tests for agent_cli.py CLI functionality.
"""

import pytest
from unittest.mock import MagicMock, patch, Mock
from agentfield.agent_cli import AgentCLI


@pytest.fixture
def mock_agent():
    """Create a mock agent instance."""
    agent = MagicMock()
    agent.node_id = "test-agent"
    agent.reasoners = [
        {"id": "reasoner1", "type": "reasoner"},
        {"id": "reasoner2", "type": "reasoner"},
    ]
    agent.skills = [
        {"id": "skill1", "type": "skill"},
    ]
    return agent


def test_agent_cli_init(mock_agent):
    """Test AgentCLI initialization."""
    cli = AgentCLI(mock_agent)
    assert cli.agent == mock_agent


def test_get_all_functions(mock_agent):
    """Test getting all available functions."""
    cli = AgentCLI(mock_agent)
    functions = cli._get_all_functions()

    assert "reasoner1" in functions
    assert "reasoner2" in functions
    assert "skill1" in functions
    assert len(functions) == 3
    # Should be sorted
    assert functions == sorted(functions)


def test_get_function(mock_agent):
    """Test getting function by name."""
    cli = AgentCLI(mock_agent)

    # Mock a function on the agent
    mock_func = MagicMock()
    setattr(mock_agent, "reasoner1", mock_func)

    result = cli._get_function("reasoner1")
    # Should return the function or its _original_func if wrapped
    assert result is not None


def test_get_function_not_found(mock_agent):
    """Test getting non-existent function."""
    cli = AgentCLI(mock_agent)
    # The function checks hasattr, so if it doesn't exist, it returns None
    # But we need to ensure the attribute doesn't exist
    if hasattr(mock_agent, "nonexistent"):
        delattr(mock_agent, "nonexistent")
    result = cli._get_function("nonexistent")
    assert result is None


def test_get_function_metadata(mock_agent):
    """Test getting function metadata."""
    cli = AgentCLI(mock_agent)

    metadata = cli._get_function_metadata("reasoner1")
    assert metadata is not None
    assert metadata["id"] == "reasoner1"
    assert metadata["type"] == "reasoner"

    metadata = cli._get_function_metadata("skill1")
    assert metadata is not None
    assert metadata["id"] == "skill1"
    assert metadata["type"] == "skill"

    metadata = cli._get_function_metadata("nonexistent")
    assert metadata is None


def test_parse_function_args_simple(mock_agent):
    """Test parsing simple function arguments."""
    cli = AgentCLI(mock_agent)

    def test_func(name: str, age: int):
        pass

    args = cli._parse_function_args(test_func, ["--name", "John", "--age", "30"])
    assert args["name"] == "John"
    assert args["age"] == 30


def test_parse_function_args_boolean(mock_agent):
    """Test parsing boolean arguments."""
    cli = AgentCLI(mock_agent)

    def test_func(verbose: bool, debug: bool):
        pass

    args = cli._parse_function_args(test_func, ["--verbose"])
    assert args["verbose"] is True
    assert args["debug"] is False  # Not provided, should be False for bool


def test_parse_function_args_with_defaults(mock_agent):
    """Test parsing arguments with default values."""
    cli = AgentCLI(mock_agent)

    def test_func(name: str, age: int = 25):
        pass

    args = cli._parse_function_args(test_func, ["--name", "John"])
    assert args["name"] == "John"
    assert args["age"] == 25  # Should use default


def test_parse_function_args_json(mock_agent):
    """Test parsing JSON arguments."""
    cli = AgentCLI(mock_agent)

    def test_func(data: dict, items: list):
        pass

    args = cli._parse_function_args(
        test_func, ["--data", '{"key": "value"}', "--items", '["a", "b"]']
    )
    assert args["data"] == {"key": "value"}
    assert args["items"] == ["a", "b"]


def test_call_function_sync(mock_agent):
    """Test calling a synchronous function."""
    cli = AgentCLI(mock_agent)

    def sync_func(name: str) -> dict:
        return {"result": f"Hello {name}"}

    mock_agent.sync_func = sync_func

    with patch("builtins.print") as mock_print:
        with patch.object(cli, "_get_function", return_value=sync_func):
            with patch.object(
                cli, "_parse_function_args", return_value={"name": "World"}
            ):
                cli._call_function("sync_func", [])

                # Should print JSON result
                mock_print.assert_called_once()
                call_args = mock_print.call_args[0][0]
                assert "Hello World" in call_args


def test_call_function_async(mock_agent):
    """Test calling an async function."""
    cli = AgentCLI(mock_agent)

    async def async_func(name: str) -> dict:
        return {"result": f"Hello {name}"}

    with patch("builtins.print"):
        with patch.object(cli, "_get_function", return_value=async_func):
            with patch.object(
                cli, "_parse_function_args", return_value={"name": "World"}
            ):
                with patch("asyncio.run") as mock_asyncio_run:
                    mock_asyncio_run.return_value = {"result": "Hello World"}
                    cli._call_function("async_func", [])

                    mock_asyncio_run.assert_called_once()


def test_call_function_not_found(mock_agent):
    """Test calling non-existent function."""
    cli = AgentCLI(mock_agent)

    with patch.object(cli, "_get_function", return_value=None):
        with patch("agentfield.agent_cli.log_error") as mock_log_error:
            with patch("sys.exit") as mock_exit:
                try:
                    cli._call_function("nonexistent", [])
                except SystemExit:
                    pass
                # Should log error and exit
                mock_log_error.assert_called()
                assert mock_exit.called


def test_show_function_help(mock_agent):
    """Test showing function help."""
    cli = AgentCLI(mock_agent)

    def test_func(name: str, age: int = 25) -> dict:
        """This is a test function."""
        pass

    with patch("builtins.print") as mock_print:
        with patch.object(cli, "_get_function", return_value=test_func):
            with patch.object(
                cli,
                "_get_function_metadata",
                return_value={"type": "reasoner", "id": "test_func"},
            ):
                cli._show_function_help("test_func")

                # Should print help information
                assert mock_print.call_count > 0
                printed = " ".join(str(call) for call in mock_print.call_args_list)
                assert "test_func" in printed
                assert "reasoner" in printed


def test_list_functions(mock_agent):
    """Test listing all functions."""
    cli = AgentCLI(mock_agent)

    def func1():
        """Function 1 docstring."""
        pass

    def func2():
        """Function 2 docstring."""
        pass

    with patch("builtins.print") as mock_print:
        with patch.object(cli, "_get_function") as mock_get_func:
            mock_get_func.side_effect = [func1, func2, func1]
            cli._list_functions()

            # Should print function list
            assert mock_print.call_count > 0
            printed = " ".join(str(call) for call in mock_print.call_args_list)
            assert "test-agent" in printed
            assert "reasoner1" in printed or "reasoner2" in printed


def test_run_cli_list_command(mock_agent):
    """Test running CLI with list command."""
    cli = AgentCLI(mock_agent)

    with patch.object(cli, "_list_functions") as mock_list:
        with patch("sys.argv", ["script", "list"]):
            with patch(
                "argparse.ArgumentParser.parse_known_args",
                return_value=(Mock(command="list"), []),
            ):
                cli.run_cli()
                mock_list.assert_called_once()


def test_run_cli_call_command(mock_agent):
    """Test running CLI with call command."""
    cli = AgentCLI(mock_agent)

    with patch.object(cli, "_call_function") as mock_call:
        with patch("sys.argv", ["script", "call", "func1", "--arg", "value"]):
            with patch(
                "argparse.ArgumentParser.parse_known_args",
                return_value=(
                    Mock(command="call", function="func1"),
                    ["--arg", "value"],
                ),
            ):
                cli.run_cli()
                mock_call.assert_called_once_with("func1", ["--arg", "value"])


def test_run_cli_help_command(mock_agent):
    """Test running CLI with help command."""
    cli = AgentCLI(mock_agent)

    with patch.object(cli, "_show_function_help") as mock_help:
        with patch("sys.argv", ["script", "help", "func1"]):
            with patch(
                "argparse.ArgumentParser.parse_known_args",
                return_value=(Mock(command="help", function="func1"), []),
            ):
                cli.run_cli()
                mock_help.assert_called_once_with("func1")


@pytest.mark.skip(
    reason="Complex argparse mocking - functionality tested in integration"
)
def test_run_cli_no_command(mock_agent):
    """Test running CLI with no command shows help."""
    # This test requires complex argparse mocking that's brittle
    # The functionality is better tested in integration tests
    pass
