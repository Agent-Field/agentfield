"""AC-12: mypy infers str when simulate_trigger gets a str-returning reasoner.

Checked with ``mypy tests/mypy_reveal_simulate_trigger.py`` — the revealed
types must be ``str`` for both the sync and async handlers (simulate_trigger
awaits coroutines transparently, so the async case must NOT reveal a
Coroutine type).
"""

from typing_extensions import reveal_type

from agentfield.testing import simulate_trigger


def handler(input: object) -> str:
    return "hello"


async def async_handler(input: object) -> str:
    return "hello"

handler._reasoner_triggers = []  # type: ignore[attr-defined]
async_handler._reasoner_triggers = []  # type: ignore[attr-defined]

reveal_type(simulate_trigger(handler, source="test", body={}))
reveal_type(simulate_trigger(async_handler, source="test", body={}))
