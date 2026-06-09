"""AC-12: mypy infers str when simulate_trigger gets a str-returning reasoner."""

from agentfield.testing import simulate_trigger


def handler(input: object) -> str:
    return "hello"


handler._reasoner_triggers = []  # type: ignore[attr-defined]

reveal_type(simulate_trigger(handler, source="test", body={}))