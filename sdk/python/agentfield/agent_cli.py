"""
CLI functionality for AgentField Agent class.

Provides native command-line interface support for running agent functions
directly from the terminal without starting a server.
"""

import argparse
import asyncio
import inspect
import json
import sys
from typing import Any, Callable, Dict, List, Optional, get_type_hints

from agentfield.logger import log_error, log_warn


class AgentCLI:
    """CLI handler for Agent class"""

    def __init__(self, agent_instance):
        """
        Initialize CLI handler with agent instance.

        Args:
            agent_instance: The Agent instance to provide CLI for
        """
        self.agent = agent_instance

    def _get_all_functions(self) -> List[str]:
        """Get list of all available reasoners and skills"""
        functions = []

        # Add reasoners
        for reasoner in self.agent.reasoners:
            functions.append(reasoner["id"])

        # Add skills
        for skill in self.agent.skills:
            functions.append(skill["id"])

        return sorted(functions)

    def _get_function(self, func_name: str) -> Optional[Callable]:
        """
        Get function by name from agent.

        Args:
            func_name: Name of the function to retrieve

        Returns:
            The function if found, None otherwise
        """
        if hasattr(self.agent, func_name):
            func = getattr(self.agent, func_name)
            # Get the original function if it's a tracked wrapper
            if hasattr(func, "_original_func"):
                return func._original_func
            return func
        return None

    def _get_function_metadata(self, func_name: str) -> Optional[Dict]:
        """
        Get metadata for a function.

        Args:
            func_name: Name of the function

        Returns:
            Metadata dict if found, None otherwise
        """
        # Check reasoners
        for reasoner in self.agent.reasoners:
            if reasoner["id"] == func_name:
                return {"type": "reasoner", **reasoner}

        # Check skills
        for skill in self.agent.skills:
            if skill["id"] == func_name:
                return {"type": "skill", **skill}

        return None

    def _parse_function_args(
        self, func: Callable, cli_args: List[str]
    ) -> Dict[str, Any]:
        """
        Parse CLI arguments for a specific function.

        Args:
            func: The function to parse arguments for
            cli_args: List of CLI arguments

        Returns:
            Dictionary of parsed arguments
        """
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        # Create argument parser for this function
        parser = argparse.ArgumentParser(
            description=f"Arguments for {func.__name__}", add_help=False
        )

        # Add arguments based on function signature
        for param_name, param in sig.parameters.items():
            if param_name in ["self", "execution_context"]:
                continue

            param_type = type_hints.get(param_name, str)

            # Handle different parameter types
            if param_type is bool:
                # Boolean flags
                parser.add_argument(
                    f"--{param_name}",
                    action="store_true",
                    help=f"{param_name} (boolean flag)",
                )
            elif param_type is int:
                parser.add_argument(
                    f"--{param_name}",
                    type=int,
                    required=param.default == inspect.Parameter.empty,
                    default=(
                        param.default
                        if param.default != inspect.Parameter.empty
                        else None
                    ),
                    help=f"{param_name} (integer)",
                )
            elif param_type is float:
                parser.add_argument(
                    f"--{param_name}",
                    type=float,
                    required=param.default == inspect.Parameter.empty,
                    default=(
                        param.default
                        if param.default != inspect.Parameter.empty
                        else None
                    ),
                    help=f"{param_name} (float)",
                )
            elif param_type in [list, List]:
                parser.add_argument(
                    f"--{param_name}",
                    type=str,
                    required=param.default == inspect.Parameter.empty,
                    default=(
                        param.default
                        if param.default != inspect.Parameter.empty
                        else None
                    ),
                    help=f"{param_name} (JSON list)",
                )
            elif param_type in [dict, Dict]:
                parser.add_argument(
                    f"--{param_name}",
                    type=str,
                    required=param.default == inspect.Parameter.empty,
                    default=(
                        param.default
                        if param.default != inspect.Parameter.empty
                        else None
                    ),
                    help=f"{param_name} (JSON object)",
                )
            else:
                # Default to string
                parser.add_argument(
                    f"--{param_name}",
                    type=str,
                    required=param.default == inspect.Parameter.empty,
                    default=(
                        param.default
                        if param.default != inspect.Parameter.empty
                        else None
                    ),
                    help=f"{param_name} (string)",
                )

        # Parse arguments
        try:
            parsed_args = parser.parse_args(cli_args)
            kwargs = vars(parsed_args)

            # Convert JSON strings to objects
            for param_name, param in sig.parameters.items():
                if param_name in kwargs and kwargs[param_name] is not None:
                    param_type = type_hints.get(param_name, str)
                    if param_type in [list, List, dict, Dict]:
                        try:
                            kwargs[param_name] = json.loads(kwargs[param_name])
                        except json.JSONDecodeError:
                            log_warn(
                                f"Failed to parse JSON for {param_name}, using as string"
                            )

            return kwargs
        except SystemExit:
            # argparse calls sys.exit on error, catch it
            raise ValueError("Invalid arguments")

    def _call_function(self, func_name: str, cli_args: List[str]) -> None:
        """
        Call a function with parsed CLI arguments.

        Args:
            func_name: Name of the function to call
            cli_args: List of CLI arguments
        """
        func = self._get_function(func_name)
        if not func:
            log_error(f"Function '{func_name}' not found")
            sys.exit(1)

        try:
            # Parse arguments
            kwargs = self._parse_function_args(func, cli_args)

            # Call function
            if inspect.iscoroutinefunction(func):
                result = asyncio.run(func(**kwargs))
            else:
                result = func(**kwargs)

            print(json.dumps(result, indent=2, default=str))

        except ValueError as e:
            log_error(f"Argument parsing failed: {e}")
            self._show_function_help(func_name)
            sys.exit(1)
        except Exception as e:
            log_error(f"Execution failed: {e}")
            sys.exit(1)

    def _show_function_help(self, func_name: str) -> None:
        """
        Show help for a specific function.

        Args:
            func_name: Name of the function
        """
        func = self._get_function(func_name)
        metadata = self._get_function_metadata(func_name)

        if not func or not metadata:
            log_error(f"Function '{func_name}' not found")
            return

        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or "No description available"

        print(f"\n{func_name} ({metadata['type']})")
        print("=" * 60)
        print(f"\n{doc}\n")
        print("Arguments:")

        for param_name, param in sig.parameters.items():
            if param_name in ["self", "execution_context"]:
                continue

            required = param.default == inspect.Parameter.empty
            default = "" if required else f" (default: {param.default})"
            req_str = "required" if required else "optional"

            print(f"  --{param_name:<20} {req_str}{default}")

        print("\nExample:")
        example_args = []
        for param_name, param in sig.parameters.items():
            if param_name in ["self", "execution_context"]:
                continue
            if param.default == inspect.Parameter.empty:
                example_args.append(f'--{param_name} "value"')

        print(f"  python main.py call {func_name} {' '.join(example_args)}")
        print()

    def _list_functions(self) -> None:
        """List all available functions with their signatures"""
        print(f"\n📋 Agent: {self.agent.node_id}\n")

        if self.agent.reasoners:
            print("Reasoners (AI-powered):")
            for reasoner in self.agent.reasoners:
                func = self._get_function(reasoner["id"])
                if func:
                    sig = inspect.signature(func)
                    doc = inspect.getdoc(func) or "No description"
                    # Get first line of docstring
                    doc_first_line = doc.split("\n")[0]
                    print(f"  • {reasoner['id']}{sig}")
                    print(f"    {doc_first_line}\n")

        if self.agent.skills:
            print("Skills (deterministic):")
            for skill in self.agent.skills:
                func = self._get_function(skill["id"])
                if func:
                    sig = inspect.signature(func)
                    doc = inspect.getdoc(func) or "No description"
                    # Get first line of docstring
                    doc_first_line = doc.split("\n")[0]
                    print(f"  • {skill['id']}{sig}")
                    print(f"    {doc_first_line}\n")

        print(
            f"Total: {len(self.agent.reasoners)} reasoners, {len(self.agent.skills)} skills\n"
        )

    def _interactive_shell(self) -> None:
        """Launch interactive shell with agent context"""
        try:
            from IPython import embed

            # Prepare namespace with all functions
            namespace = {
                "agent": self.agent,
                "asyncio": asyncio,
            }

            # Add all skills and reasoners to namespace
            for reasoner in self.agent.reasoners:
                func = self._get_function(reasoner["id"])
                if func:
                    namespace[reasoner["id"]] = func

            for skill in self.agent.skills:
                func = self._get_function(skill["id"])
                if func:
                    namespace[skill["id"]] = func

            print(f"🚀 Agent Shell: {self.agent.node_id}")
            print(f"Available functions: {', '.join(self._get_all_functions())}")
            print("\nTip: Use 'await function_name(args)' for async functions")
            print("     Use 'function_name(args)' for sync functions\n")

            embed(user_ns=namespace)
        except ImportError:
            log_error("IPython not installed. Install with: pip install ipython")
            sys.exit(1)

    def run_cli(self) -> None:
        """
        Main CLI entry point - parses commands and executes.
        """
        parser = argparse.ArgumentParser(
            description=f"Agent CLI: {self.agent.node_id}",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # 'call' command
        call_parser = subparsers.add_parser("call", help="Call a function")
        call_parser.add_argument("function", help="Function name to call")

        # 'list' command
        subparsers.add_parser("list", help="List all functions")

        # 'shell' command
        subparsers.add_parser("shell", help="Interactive shell")

        # 'help' command
        help_parser = subparsers.add_parser("help", help="Show help for a function")
        help_parser.add_argument("function", help="Function name")

        # Parse known args to separate command from function args
        args, unknown = parser.parse_known_args()

        if not args.command:
            parser.print_help()
            sys.exit(0)

        if args.command == "call":
            self._call_function(args.function, unknown)
        elif args.command == "list":
            self._list_functions()
        elif args.command == "shell":
            self._interactive_shell()
        elif args.command == "help":
            self._show_function_help(args.function)
        else:
            parser.print_help()
            sys.exit(1)
