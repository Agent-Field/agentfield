"""AC-13: mypy infers str when simulate_schedule gets a str-returning reasoner."""

from agentfield.testing import simulate_schedule


def handler(input: object) -> str:
    return "ok"


handler._reasoner_triggers = []  # type: ignore[attr-defined]

reveal_type(simulate_schedule(handler))