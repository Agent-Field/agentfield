"""
Tests for agentfield.types — data models, enums, and configuration classes.
"""
from __future__ import annotations

import asyncio
import sys
import types as stdlib_types
import pytest

from agentfield.types import (
    AgentCapability,
    AgentStatus,
    AIConfig,
    CompactCapability,
    CompactDiscoveryResponse,
    DiscoveryPagination,
    DiscoveryResponse,
    DiscoveryResult,
    ExecutionHeaders,
    HarnessConfig,
    HeartbeatData,

    MemoryChangeEvent,
    MemoryConfig,
    MemoryValue,
    ReasonerCapability,
    ReasonerDefinition,
    SkillCapability,
    SkillDefinition,
    WebhookConfig,
)


# ---------------------------------------------------------------------------
# AgentStatus enum
# ---------------------------------------------------------------------------


class TestAgentStatus:
    def test_values(self):
        assert AgentStatus.STARTING == "starting"
        assert AgentStatus.READY == "ready"
        assert AgentStatus.DEGRADED == "degraded"
        assert AgentStatus.OFFLINE == "offline"

    def test_is_str_subclass(self):
        assert isinstance(AgentStatus.READY, str)

    def test_from_value(self):
        assert AgentStatus("ready") is AgentStatus.READY

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            AgentStatus("nonexistent")

    def test_all_members(self):
        assert len(AgentStatus) == 4


# ---------------------------------------------------------------------------
# HeartbeatData
# ---------------------------------------------------------------------------


class TestHeartbeatData:
    def test_construction_and_defaults(self):
        hb = HeartbeatData(
            status=AgentStatus.READY,
            timestamp="2024-01-01T00:00:00Z",
        )
        assert hb.version == ""

    def test_to_dict_serializes_status_value(self):
        hb = HeartbeatData(
            status=AgentStatus.READY,
            timestamp="2024-01-01T00:00:00Z",
            version="1.0",
        )
        d = hb.to_dict()
        assert d["status"] == "ready"
        assert d["version"] == "1.0"
        assert d["timestamp"] == "2024-01-01T00:00:00Z"

    def test_to_dict_offline(self):
        hb = HeartbeatData(
            status=AgentStatus.OFFLINE, timestamp="t"
        )
        d = hb.to_dict()
        assert d["status"] == "offline"


# ---------------------------------------------------------------------------
# MemoryConfig
# ---------------------------------------------------------------------------


class TestMemoryConfig:
    def test_construction(self):
        mc = MemoryConfig(
            auto_inject=["session"], memory_retention="run", cache_results=True
        )
        assert mc.auto_inject == ["session"]
        assert mc.cache_results is True

    def test_to_dict(self):
        mc = MemoryConfig(
            auto_inject=[], memory_retention="session", cache_results=False
        )
        d = mc.to_dict()
        assert d == {
            "auto_inject": [],
            "memory_retention": "session",
            "cache_results": False,
        }


# ---------------------------------------------------------------------------
# ReasonerDefinition
# ---------------------------------------------------------------------------


class TestReasonerDefinition:
    def test_without_memory_config(self):
        rd = ReasonerDefinition(
            id="r1",
            input_schema={"type": "object"},
            output_schema={"type": "string"},
        )
        assert rd.memory_config is None
        d = rd.to_dict()
        assert d["id"] == "r1"
        assert d["input_schema"] == {"type": "object"}

    def test_with_memory_config(self):
        mc = MemoryConfig(
            auto_inject=["global"], memory_retention="run", cache_results=False
        )
        rd = ReasonerDefinition(
            id="r1", input_schema={}, output_schema={}, memory_config=mc
        )
        d = rd.to_dict()
        assert d["memory_config"]["auto_inject"] == ["global"]


# ---------------------------------------------------------------------------
# SkillDefinition
# ---------------------------------------------------------------------------


class TestSkillDefinition:
    def test_construction_and_to_dict(self):
        sd = SkillDefinition(
            id="skill-1", input_schema={"type": "object"}, tags=["tag1", "tag2"]
        )
        d = sd.to_dict()
        assert d["id"] == "skill-1"
        assert d["tags"] == ["tag1", "tag2"]

    def test_empty_tags(self):
        sd = SkillDefinition(id="skill-2", input_schema={}, tags=[])
        assert sd.to_dict()["tags"] == []


# ---------------------------------------------------------------------------
# ExecutionHeaders
# ---------------------------------------------------------------------------


class TestExecutionHeaders:
    def test_minimal(self):
        eh = ExecutionHeaders(run_id="run-1")
        headers = eh.to_headers()
        assert headers == {"X-Run-ID": "run-1"}

    def test_full(self):
        eh = ExecutionHeaders(
            run_id="run-1",
            session_id="sess-1",
            actor_id="actor-1",
            parent_execution_id="exec-parent",
        )
        headers = eh.to_headers()
        assert headers["X-Run-ID"] == "run-1"
        assert headers["X-Session-ID"] == "sess-1"
        assert headers["X-Actor-ID"] == "actor-1"
        assert headers["X-Parent-Execution-ID"] == "exec-parent"

    def test_optional_fields_omitted(self):
        eh = ExecutionHeaders(run_id="run-1", session_id="sess-1")
        headers = eh.to_headers()
        assert "X-Actor-ID" not in headers
        assert "X-Parent-Execution-ID" not in headers

    def test_defaults(self):
        eh = ExecutionHeaders(run_id="r")
        assert eh.session_id is None
        assert eh.actor_id is None
        assert eh.parent_execution_id is None


# ---------------------------------------------------------------------------
# WebhookConfig
# ---------------------------------------------------------------------------


class TestWebhookConfig:
    def test_minimal_payload(self):
        wc = WebhookConfig(url="https://example.com/hook")
        p = wc.to_payload()
        assert p == {"url": "https://example.com/hook"}

    def test_full_payload(self):
        wc = WebhookConfig(
            url="https://example.com/hook",
            secret="s3cret",
            headers={"X-Custom": "val"},
        )
        p = wc.to_payload()
        assert p["secret"] == "s3cret"
        assert p["headers"]["X-Custom"] == "val"

    def test_optional_fields_omitted_from_payload(self):
        wc = WebhookConfig(url="https://example.com/hook")
        p = wc.to_payload()
        assert "secret" not in p
        assert "headers" not in p

    def test_defaults(self):
        wc = WebhookConfig(url="u")
        assert wc.secret is None
        assert wc.headers is None


# ---------------------------------------------------------------------------
# DiscoveryPagination
# ---------------------------------------------------------------------------


class TestDiscoveryPagination:
    def test_from_dict(self):
        dp = DiscoveryPagination.from_dict(
            {"limit": 10, "offset": 5, "has_more": True}
        )
        assert dp.limit == 10
        assert dp.offset == 5
        assert dp.has_more is True

    def test_from_dict_defaults(self):
        dp = DiscoveryPagination.from_dict({})
        assert dp.limit == 0
        assert dp.offset == 0
        assert dp.has_more is False

    def test_from_dict_coerces_types(self):
        dp = DiscoveryPagination.from_dict(
            {"limit": "20", "offset": "3", "has_more": 1}
        )
        assert dp.limit == 20
        assert dp.offset == 3
        assert dp.has_more is True


# ---------------------------------------------------------------------------
# ReasonerCapability
# ---------------------------------------------------------------------------


class TestReasonerCapability:
    def test_from_dict_full(self):
        data = {
            "id": "r1",
            "description": "A reasoner",
            "tags": ["t1"],
            "input_schema": {"type": "object"},
            "output_schema": {"type": "string"},
            "examples": [{"input": "hi"}],
            "invocation_target": "node.r1",
        }
        rc = ReasonerCapability.from_dict(data)
        assert rc.id == "r1"
        assert rc.description == "A reasoner"
        assert rc.tags == ["t1"]
        assert rc.examples == [{"input": "hi"}]

    def test_from_dict_minimal(self):
        rc = ReasonerCapability.from_dict({})
        assert rc.id == ""
        assert rc.description is None
        assert rc.tags == []
        assert rc.examples is None
        assert rc.invocation_target == ""

    def test_from_dict_empty_examples_becomes_none(self):
        rc = ReasonerCapability.from_dict({"examples": []})
        assert rc.examples is None

    def test_from_dict_none_tags_becomes_empty_list(self):
        rc = ReasonerCapability.from_dict({"tags": None})
        assert rc.tags == []


# ---------------------------------------------------------------------------
# SkillCapability
# ---------------------------------------------------------------------------


class TestSkillCapability:
    def test_from_dict(self):
        sc = SkillCapability.from_dict(
            {
                "id": "sk1",
                "description": "A skill",
                "tags": ["x"],
                "input_schema": {},
                "invocation_target": "node.sk1",
            }
        )
        assert sc.id == "sk1"
        assert sc.tags == ["x"]

    def test_from_dict_defaults(self):
        sc = SkillCapability.from_dict({})
        assert sc.id == ""
        assert sc.tags == []
        assert sc.description is None


# ---------------------------------------------------------------------------
# AgentCapability
# ---------------------------------------------------------------------------


class TestAgentCapability:
    def test_from_dict_with_nested(self):
        data = {
            "agent_id": "a1",
            "base_url": "http://localhost:8001",
            "version": "1.0",
            "health_status": "healthy",
            "deployment_type": "local",
            "last_heartbeat": "2024-01-01T00:00:00Z",
            "reasoners": [{"id": "r1", "invocation_target": "a1.r1"}],
            "skills": [{"id": "s1", "invocation_target": "a1.s1"}],
        }
        ac = AgentCapability.from_dict(data)
        assert ac.agent_id == "a1"
        assert len(ac.reasoners) == 1
        assert ac.reasoners[0].id == "r1"
        assert len(ac.skills) == 1

    def test_from_dict_empty(self):
        ac = AgentCapability.from_dict({})
        assert ac.agent_id == ""
        assert ac.reasoners == []
        assert ac.skills == []

    def test_from_dict_none_reasoners_skills(self):
        ac = AgentCapability.from_dict({"reasoners": None, "skills": None})
        assert ac.reasoners == []
        assert ac.skills == []


# ---------------------------------------------------------------------------
# DiscoveryResponse
# ---------------------------------------------------------------------------


class TestDiscoveryResponse:
    def test_from_dict(self):
        data = {
            "discovered_at": "2024-01-01",
            "total_agents": 2,
            "total_reasoners": 3,
            "total_skills": 1,
            "pagination": {"limit": 10, "offset": 0, "has_more": False},
            "capabilities": [
                {
                    "agent_id": "a1",
                    "base_url": "http://localhost",
                    "version": "1.0",
                    "health_status": "healthy",
                    "deployment_type": "local",
                    "last_heartbeat": "now",
                }
            ],
        }
        dr = DiscoveryResponse.from_dict(data)
        assert dr.total_agents == 2
        assert dr.total_skills == 1
        assert len(dr.capabilities) == 1
        assert dr.pagination.has_more is False

    def test_from_dict_empty(self):
        dr = DiscoveryResponse.from_dict({})
        assert dr.total_agents == 0
        assert dr.capabilities == []
        assert dr.pagination.limit == 0

    def test_from_dict_none_pagination(self):
        dr = DiscoveryResponse.from_dict({"pagination": None})
        assert dr.pagination.limit == 0


# ---------------------------------------------------------------------------
# CompactCapability / CompactDiscoveryResponse
# ---------------------------------------------------------------------------


class TestCompactCapability:
    def test_from_dict(self):
        cc = CompactCapability.from_dict(
            {"id": "c1", "agent_id": "a1", "target": "a1.c1", "tags": ["t"]}
        )
        assert cc.id == "c1"
        assert cc.tags == ["t"]

    def test_from_dict_defaults(self):
        cc = CompactCapability.from_dict({})
        assert cc.id == ""
        assert cc.tags == []
        assert cc.target == ""


class TestCompactDiscoveryResponse:
    def test_from_dict(self):
        data = {
            "discovered_at": "2024-01-01",
            "reasoners": [
                {"id": "r1", "agent_id": "a1", "target": "t", "tags": []}
            ],
            "skills": [],
        }
        cdr = CompactDiscoveryResponse.from_dict(data)
        assert len(cdr.reasoners) == 1
        assert cdr.skills == []

    def test_from_dict_empty(self):
        cdr = CompactDiscoveryResponse.from_dict({})
        assert cdr.reasoners == []
        assert cdr.skills == []


# ---------------------------------------------------------------------------
# DiscoveryResult
# ---------------------------------------------------------------------------


class TestDiscoveryResult:
    def test_construction(self):
        dr = DiscoveryResult(format="json", raw='{"ok": true}')
        assert dr.format == "json"
        assert dr.json is None
        assert dr.compact is None
        assert dr.xml is None

    def test_with_json_response(self):
        resp = DiscoveryResponse.from_dict({"total_agents": 1})
        dr = DiscoveryResult(format="json", raw="{}", json=resp)
        assert dr.json is not None
        assert dr.json.total_agents == 1


# ---------------------------------------------------------------------------
# HarnessConfig (Pydantic BaseModel)
# ---------------------------------------------------------------------------


class TestHarnessConfig:
    def test_defaults(self):
        hc = HarnessConfig(provider="claude-code")
        assert hc.provider == "claude-code"
        assert hc.model == "sonnet"
        assert hc.max_turns == 30
        assert hc.max_budget_usd is None
        assert hc.max_retries == 3
        assert hc.initial_delay == 1.0
        assert hc.max_delay == 30.0
        assert hc.backoff_factor == 2.0
        assert "Read" in hc.tools

    def test_custom_values(self):
        hc = HarnessConfig(
            provider="codex",
            model="gpt-4o",
            max_turns=10,
            max_budget_usd=5.0,
            tools=["Bash"],
            permission_mode="auto",
        )
        assert hc.provider == "codex"
        assert hc.max_budget_usd == 5.0
        assert hc.tools == ["Bash"]
        assert hc.permission_mode == "auto"

    def test_provider_required(self):
        with pytest.raises(Exception):
            HarnessConfig()  # type: ignore[call-arg]

    def test_json_roundtrip(self):
        hc = HarnessConfig(provider="gemini", model="gemini-2.5-flash")
        data = hc.model_dump()
        hc2 = HarnessConfig(**data)
        assert hc2.provider == "gemini"
        assert hc2.model == "gemini-2.5-flash"

    def test_binary_paths_defaults(self):
        hc = HarnessConfig(provider="codex")
        assert hc.codex_bin == "codex"
        assert hc.gemini_bin == "gemini"
        assert hc.opencode_bin == "opencode"

    def test_optional_fields(self):
        hc = HarnessConfig(provider="p")
        assert hc.system_prompt is None
        assert hc.cwd is None
        assert hc.project_dir is None
        assert hc.env == {}


# ---------------------------------------------------------------------------
# AIConfig (Pydantic BaseModel)
# ---------------------------------------------------------------------------


class TestAIConfig:
    def test_defaults(self):
        cfg = AIConfig()
        assert cfg.model == "gpt-4o"
        assert cfg.temperature is None
        assert cfg.response_format == "auto"
        assert cfg.image_quality == "high"
        assert cfg.enable_rate_limit_retry is True
        assert cfg.preserve_context is True
        assert cfg.context_window == 10
        assert cfg.avg_chars_per_token == 4

    def test_computed_image_model(self):
        cfg = AIConfig(vision_model="dall-e-3")
        assert cfg.image_model == "dall-e-3"

    def test_temperature_validation(self):
        cfg = AIConfig(temperature=1.5)
        assert cfg.temperature == 1.5
        with pytest.raises(Exception):
            AIConfig(temperature=3.0)  # exceeds max 2.0

    def test_temperature_negative_raises(self):
        with pytest.raises(Exception):
            AIConfig(temperature=-0.1)

    def test_top_p_validation(self):
        cfg = AIConfig(top_p=0.5)
        assert cfg.top_p == 0.5
        with pytest.raises(Exception):
            AIConfig(top_p=1.5)

    def test_from_env(self):
        cfg = AIConfig.from_env(model="claude-3-opus", temperature=0.2)
        assert cfg.model == "claude-3-opus"
        assert cfg.temperature == 0.2

    def test_to_dict(self):
        cfg = AIConfig(model="gpt-4o", temperature=0.5)
        d = cfg.to_dict()
        assert d["model"] == "gpt-4o"
        assert d["temperature"] == 0.5
        # Should include computed field
        assert "image_model" in d

    def test_copy_with_update(self):
        cfg = AIConfig(model="gpt-4o")
        cfg2 = cfg.copy(update={"model": "claude-3-opus"})
        assert cfg2.model == "claude-3-opus"
        assert cfg.model == "gpt-4o"  # original unchanged

    def test_get_litellm_params_basic(self):
        cfg = AIConfig(model="gpt-4o", temperature=0.7, max_tokens=100)
        params = cfg.get_litellm_params()
        assert params["model"] == "gpt-4o"
        assert params["temperature"] == 0.7
        assert params["max_tokens"] == 100
        assert "api_key" not in params

    def test_get_litellm_params_strips_none(self):
        cfg = AIConfig()
        params = cfg.get_litellm_params()
        # temperature is None by default so should be stripped
        assert "temperature" not in params

    def test_get_litellm_params_with_overrides(self):
        cfg = AIConfig(model="gpt-4o")
        params = cfg.get_litellm_params(model="claude-3-opus", temperature=0.1)
        assert params["model"] == "claude-3-opus"
        assert params["temperature"] == 0.1

    def test_get_litellm_params_response_format_non_auto(self):
        cfg = AIConfig(response_format="json")
        params = cfg.get_litellm_params()
        assert params["response_format"] == {"type": "json"}

    def test_get_litellm_params_response_format_auto_omitted(self):
        cfg = AIConfig(response_format="auto")
        params = cfg.get_litellm_params()
        assert "response_format" not in params

    def test_get_litellm_params_openai_provider_rewrites_max_tokens(self):
        cfg = AIConfig(model="openai/gpt-4o", max_tokens=500)
        params = cfg.get_litellm_params()
        assert "max_completion_tokens" in params
        assert "max_tokens" not in params
        assert params["max_completion_tokens"] == 500

    def test_get_litellm_params_non_openai_keeps_max_tokens(self):
        cfg = AIConfig(model="anthropic/claude-3-opus", max_tokens=500)
        params = cfg.get_litellm_params()
        assert params["max_tokens"] == 500
        assert "max_completion_tokens" not in params

    def test_get_litellm_params_with_api_key(self):
        cfg = AIConfig(api_key="test-key", api_base="http://localhost:1234")
        params = cfg.get_litellm_params()
        assert params["api_key"] == "test-key"
        assert params["api_base"] == "http://localhost:1234"

    def test_get_litellm_params_with_org_and_version(self):
        cfg = AIConfig(organization="org-1", api_version="2024-01-01")
        params = cfg.get_litellm_params()
        assert params["organization"] == "org-1"
        assert params["api_version"] == "2024-01-01"

    def test_get_litellm_params_litellm_extras(self):
        cfg = AIConfig(litellm_params={"seed": 42})
        params = cfg.get_litellm_params()
        assert params["seed"] == 42

    def test_trim_by_chars_short_text(self):
        cfg = AIConfig()
        text = "short"
        assert cfg.trim_by_chars(text, 100) == "short"

    def test_trim_by_chars_long_text(self):
        cfg = AIConfig()
        text = "A" * 200
        result = cfg.trim_by_chars(text, 100, head_ratio=0.3)
        assert len(result) < 200
        assert "TRIMMED" in result
        assert result.startswith("A" * 30)
        assert result.endswith("A" * 70)

    def test_trim_by_chars_exact_limit(self):
        cfg = AIConfig()
        text = "A" * 100
        assert cfg.trim_by_chars(text, 100) == text

    def test_get_safe_prompt_chars_uncached(self):
        cfg = AIConfig()
        chars = cfg.get_safe_prompt_chars()
        assert chars >= 1000

    def test_get_safe_prompt_chars_cached(self):
        cfg = AIConfig()
        cfg.model_limits_cache["gpt-4o"] = {
            "context_length": 128000,
            "max_output_tokens": 4096,
        }
        chars = cfg.get_safe_prompt_chars()
        expected = (128000 - 4096) * 4
        assert chars == expected

    def test_get_safe_prompt_chars_with_override(self):
        cfg = AIConfig()
        cfg.model_limits_cache["gpt-4o"] = {
            "context_length": 128000,
            "max_output_tokens": 4096,
        }
        chars = cfg.get_safe_prompt_chars(max_output_tokens=8000)
        expected = (128000 - 8000) * 4
        assert chars == expected

    def test_get_safe_prompt_chars_minimum(self):
        """Ensure minimum of 1000 chars returned."""
        cfg = AIConfig()
        cfg.model_limits_cache["gpt-4o"] = {
            "context_length": 100,
            "max_output_tokens": 100,
        }
        chars = cfg.get_safe_prompt_chars()
        assert chars == 1000

    def test_get_model_limits_cached(self):
        cfg = AIConfig()
        cfg.model_limits_cache["gpt-4o"] = {
            "context_length": 128000,
            "max_output_tokens": 4096,
        }
        result = asyncio.get_event_loop().run_until_complete(cfg.get_model_limits())
        assert result["context_length"] == 128000

    def test_get_model_limits_fallback_known_model(self):
        """When litellm raises, falls back to hardcoded limits."""
        cfg = AIConfig(model="gpt-4o")

        fake = stdlib_types.ModuleType("litellm")
        fake.suppress_debug_info = True

        def bad_get_model_info(model):
            raise Exception("no litellm")

        fake.get_model_info = bad_get_model_info
        old = sys.modules.get("litellm")
        sys.modules["litellm"] = fake
        try:
            result = asyncio.get_event_loop().run_until_complete(
                cfg.get_model_limits("gpt-4o")
            )
            assert result["context_length"] == 128000
        finally:
            if old is not None:
                sys.modules["litellm"] = old
            else:
                sys.modules.pop("litellm", None)

    def test_get_model_limits_unknown_model_fallback(self):
        """Unknown model with no litellm falls back to 8192."""
        cfg = AIConfig(model="unknown-model-xyz")
        fake = stdlib_types.ModuleType("litellm")
        fake.suppress_debug_info = True
        fake.get_model_info = lambda m: (_ for _ in ()).throw(Exception("fail"))
        old = sys.modules.get("litellm")
        sys.modules["litellm"] = fake
        try:
            result = asyncio.get_event_loop().run_until_complete(
                cfg.get_model_limits("unknown-model-xyz")
            )
            assert result["context_length"] == 8192
        finally:
            if old is not None:
                sys.modules["litellm"] = old
            else:
                sys.modules.pop("litellm", None)

    def test_model_context_limits_dict_has_entries(self):
        """Sanity: the fallback mapping is populated."""
        # Access on instance to bypass Pydantic private attr handling
        cfg = AIConfig()
        limits = cfg._MODEL_CONTEXT_LIMITS
        assert len(limits) > 10
        assert "gpt-4o" in limits

    def test_protected_namespaces(self):
        cfg = AIConfig()
        assert cfg.model_config == {"protected_namespaces": ()}

    def test_rate_limit_defaults(self):
        cfg = AIConfig()
        assert cfg.rate_limit_max_retries == 5
        assert cfg.rate_limit_base_delay == 0.5
        assert cfg.rate_limit_max_delay == 30.0
        assert cfg.rate_limit_jitter_factor == 0.25
        assert cfg.rate_limit_circuit_breaker_threshold == 5
        assert cfg.rate_limit_circuit_breaker_timeout == 30

    def test_fallback_models_default_empty(self):
        cfg = AIConfig()
        assert cfg.fallback_models == []

    def test_auto_inject_memory_default_empty(self):
        cfg = AIConfig()
        assert cfg.auto_inject_memory == []

    def test_audio_defaults(self):
        cfg = AIConfig()
        assert cfg.audio_model == "tts-1"
        assert cfg.audio_format == "wav"


# ---------------------------------------------------------------------------
# MemoryValue
# ---------------------------------------------------------------------------


class TestMemoryValue:
    def test_construction_and_to_dict(self):
        mv = MemoryValue(
            key="k1",
            data={"nested": True},
            scope="agent",
            scope_id="a1",
            created_at="2024-01-01",
            updated_at="2024-01-02",
        )
        d = mv.to_dict()
        assert d["key"] == "k1"
        assert d["data"] == {"nested": True}

    def test_from_dict(self):
        data = {
            "key": "k1",
            "data": 42,
            "scope": "session",
            "scope_id": "s1",
            "created_at": "2024-01-01",
            "updated_at": "2024-01-02",
        }
        mv = MemoryValue.from_dict(data)
        assert mv.key == "k1"
        assert mv.data == 42

    def test_roundtrip(self):
        data = {
            "key": "k1",
            "data": [1, 2, 3],
            "scope": "run",
            "scope_id": "r1",
            "created_at": "t1",
            "updated_at": "t2",
        }
        mv = MemoryValue.from_dict(data)
        assert mv.to_dict() == data


# ---------------------------------------------------------------------------
# MemoryChangeEvent
# ---------------------------------------------------------------------------


class TestMemoryChangeEvent:
    def test_defaults(self):
        evt = MemoryChangeEvent()
        assert evt.id is None
        assert evt.type is None
        assert evt.timestamp is None
        assert evt.scope == ""
        assert evt.key == ""
        assert evt.action == ""
        assert evt.data is None
        assert evt.previous_data is None
        assert evt.metadata == {}

    def test_full_construction(self):
        evt = MemoryChangeEvent(
            id="ev1",
            type="update",
            timestamp="2024-01-01",
            scope="agent",
            scope_id="a1",
            key="k1",
            action="set",
            data={"new": True},
            previous_data={"old": True},
            metadata={"source": "test"},
        )
        assert evt.data == {"new": True}
        assert evt.previous_data == {"old": True}

    def test_backward_compat_aliases(self):
        evt = MemoryChangeEvent(data="new", previous_data="old")
        assert evt.new_value == "new"
        assert evt.old_value == "old"

    def test_new_value_none(self):
        evt = MemoryChangeEvent()
        assert evt.new_value is None
        assert evt.old_value is None

    def test_from_dict(self):
        data = {
            "id": "ev1",
            "type": "create",
            "timestamp": "now",
            "scope": "session",
            "scope_id": "s1",
            "key": "k1",
            "action": "set",
            "data": 42,
            "previous_data": None,
            "metadata": {"tag": "x"},
        }
        evt = MemoryChangeEvent.from_dict(data)
        assert evt.id == "ev1"
        assert evt.data == 42
        assert evt.metadata == {"tag": "x"}

    def test_from_dict_defaults(self):
        evt = MemoryChangeEvent.from_dict({})
        assert evt.scope == ""
        assert evt.metadata == {}
        assert evt.id is None

    def test_from_dict_none_metadata(self):
        evt = MemoryChangeEvent.from_dict({"metadata": None})
        assert evt.metadata == {}

    def test_to_dict(self):
        evt = MemoryChangeEvent(id="e1", scope="agent", key="k")
        d = evt.to_dict()
        assert d["id"] == "e1"
        assert d["scope"] == "agent"
        assert d["key"] == "k"
