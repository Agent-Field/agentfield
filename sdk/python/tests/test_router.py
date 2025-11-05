import pytest

from haxen_sdk.router import AgentRouter


class DummyAgent:
    def __init__(self):
        self.calls = []

    async def ai(self, *args, **kwargs):
        self.calls.append(("ai", args, kwargs))
        return "ai-called"

    async def call(self, target, *args, **kwargs):
        self.calls.append((target, args, kwargs))
        return "call-result"

    @property
    def memory(self):
        return "memory-client"


@pytest.mark.asyncio
async def test_router_requires_agent_before_use():
    router = AgentRouter()

    with pytest.raises(RuntimeError):
        await router.call("node.skill")

    agent = DummyAgent()
    router._attach_agent(agent)

    result = await router.call("node.skill", 1, mode="fast")
    assert result == "call-result"
    assert agent.calls == [("node.skill", (1,), {"mode": "fast"})]

    ai_result = await router.ai("gpt")
    assert ai_result == "ai-called"

    assert router.memory == "memory-client"


def test_reasoner_and_skill_registration():
    router = AgentRouter(prefix="/api/v1", tags=["base"])

    @router.reasoner(path="/foo")
    def sample_reasoner():
        return "reasoner"

    @router.skill(tags=["extra"], path="tool")
    def sample_skill():
        return "skill"

    assert router.reasoners[0]["func"] is sample_reasoner
    assert router.reasoners[0]["path"] == "/foo"

    skill_entry = router.skills[0]
    assert skill_entry["func"] is sample_skill
    assert skill_entry["tags"] == ["base", "extra"]
    assert skill_entry["path"] == "tool"


@pytest.mark.parametrize(
    "prefix,default,custom,expected",
    [
        ("", None, None, None),
        ("/api", "/items", None, "/api/items"),
        ("api/", None, "detail", "/api/detail"),
        ("/root/", "default", "custom", "/root/custom"),
        ("", "default", None, "/default"),
        ("group", "/reasoners/foo", None, "/reasoners/group/foo"),
    ],
)
def test_combine_path(prefix, default, custom, expected):
    router = AgentRouter(prefix=prefix)
    assert router._combine_path(default, custom) == expected
