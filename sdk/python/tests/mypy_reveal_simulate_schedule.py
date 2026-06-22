"""AC-13: mypy infers str when simulate_schedule gets a str-returning reasoner.

Checked with ``mypy tests/mypy_reveal_simulate_schedule.py`` — the revealed
types must be ``str`` for both the sync and async handlers (simulate_schedule
awaits coroutines transparently, so the async case must NOT reveal a
Coroutine type).
"""

from typing_extensions import reveal_type

from agentfield.testing import simulate_schedule


def handler(input: object) -> str:
    return "ok"


async def async_handler(input: object) -> str:
    return "ok"

handler._reasoner_triggers = []  # type: ignore[attr-defined]
async_handler._reasoner_triggers = []  # type: ignore[attr-defined]

reveal_type(simulate_schedule(handler))
reveal_type(simulate_schedule(async_handler))
