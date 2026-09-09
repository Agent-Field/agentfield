from typing import Any, Dict, List, Optional, Sequence, Union

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from agentfield import Agent
from agentfield.execution_context import get_current_context
from agentfield.pydantic_utils import should_convert_args


pytestmark = pytest.mark.unit


class M1(BaseModel):
    a: int


class M2(BaseModel):
    b: int


def make_agent(node_id: str) -> Agent:
    # An empty server keeps X-Execution-ID requests on the synchronous path,
    # allowing the cancellation-aware await branch to be tested directly.
    return Agent(node_id=node_id, agentfield_server="", auto_register=False)


def test_union_of_models_coerces_over_http():
    # Contract item 1: every valid union branch reaches the handler as a model.
    app = make_agent("pydantic-union-http")
    seen = []

    @app.reasoner()
    def choose_model(item: Union[M1, M2, None] = None) -> dict:
        seen.append(item)
        return {"model": type(item).__name__}

    client = TestClient(app, raise_server_exceptions=False)
    first = client.post("/reasoners/choose_model", json={"item": {"a": 1}})
    second = client.post("/reasoners/choose_model", json={"item": {"b": 2}})

    assert first.status_code == 200
    assert first.json() == {"model": "M1"}
    assert second.status_code == 200
    assert second.json() == {"model": "M2"}
    assert seen == [M1(a=1), M2(b=2)]


def test_optional_list_of_models_coerces_over_http():
    # Contract item 2: model dictionaries nested in an optional list are
    # converted before the handler runs.
    app = make_agent("pydantic-list-http")

    @app.reasoner()
    def collect(items: Optional[List[M1]] = None) -> dict:
        return {
            "all_models": all(isinstance(item, M1) for item in items or []),
            "values": [item.a for item in items or []],
        }

    response = TestClient(app, raise_server_exceptions=False).post(
        "/reasoners/collect", json={"items": [{"a": 1}, {"a": 2}]}
    )

    assert response.status_code == 200
    assert response.json() == {"all_models": True, "values": [1, 2]}


def test_sequence_of_optional_models_coerces_over_http():
    # Contract item 3: nested None remains valid while adjacent dictionaries
    # are converted into models.
    app = make_agent("pydantic-sequence-http")

    @app.reasoner()
    def collect(seq: Sequence[Union[M1, None]] = ()) -> dict:
        return {
            "first_is_model": isinstance(seq[0], M1),
            "values": [item.a if item is not None else None for item in seq],
        }

    response = TestClient(app, raise_server_exceptions=False).post(
        "/reasoners/collect", json={"seq": [{"a": 1}, None]}
    )

    assert response.status_code == 200
    assert response.json() == {"first_is_model": True, "values": [1, None]}


def test_plain_parameters_remain_plain_over_http():
    # Contract item 4: signatures without models do not enter Pydantic
    # conversion, and plain values retain their contents and types.
    app = make_agent("pydantic-plain-http")
    original_payload = {"nested": {"value": 1}}

    @app.reasoner()
    def plain(payload: Dict[str, Any], n: int, s: str) -> dict:
        return {
            "payload": payload,
            "is_dict": type(payload) is dict,
            "n": n,
            "s": s,
        }

    assert should_convert_args(plain) is False
    response = TestClient(app, raise_server_exceptions=False).post(
        "/reasoners/plain",
        json={"payload": original_payload, "n": 4, "s": "unchanged"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "payload": original_payload,
        "is_dict": True,
        "n": 4,
        "s": "unchanged",
    }


@pytest.mark.parametrize("send_null", [False, True])
def test_implicit_optional_model_defaults_preserve_none_over_http(send_null):
    # Contract item 5: absent and explicit-null values both preserve implicit
    # Optional defaults for bare models and containers of models.
    app = make_agent(f"pydantic-defaults-http-{send_null}")

    @app.reasoner()
    def bare_default(m: M1 = None) -> dict:
        return {"is_none": m is None}

    @app.reasoner()
    def list_default(items: List[M1] = None) -> dict:
        return {"is_none": items is None}

    client = TestClient(app, raise_server_exceptions=False)
    bare_body = {"m": None} if send_null else {}
    list_body = {"items": None} if send_null else {}
    bare_response = client.post("/reasoners/bare_default", json=bare_body)
    list_response = client.post("/reasoners/list_default", json=list_body)

    assert bare_response.status_code == 200
    assert bare_response.json() == {"is_none": True}
    assert list_response.status_code == 200
    assert list_response.json() == {"is_none": True}


def test_invalid_composite_reasoner_inputs_return_safe_422():
    # Contract item 6: composite reasoner validation failures are safe 422s,
    # including both synchronous endpoint await branches.
    app = make_agent("pydantic-invalid-reasoner-http")
    entered = []

    @app.reasoner()
    def union_reasoner(item: Union[M1, M2, None] = None) -> dict:
        entered.append("union")
        return {"ok": True}

    @app.reasoner()
    def list_reasoner(items: List[M1]) -> dict:
        entered.append("list")
        return {"ok": True}

    client = TestClient(app, raise_server_exceptions=False)
    union_response = client.post(
        "/reasoners/union_reasoner", json={"item": {"x": "offending-union"}}
    )
    list_response = client.post(
        "/reasoners/list_reasoner",
        json={"items": [{"a": "offending-list"}]},
        headers={"X-Execution-ID": "sync-validation-error"},
    )

    assert union_response.status_code == 422
    assert union_response.json() == {
        "detail": "Pydantic validation failed for reasoner 'union_reasoner'"
    }
    assert list_response.status_code == 422
    assert list_response.json() == {
        "detail": "Pydantic validation failed for reasoner 'list_reasoner'"
    }
    assert "offending-union" not in union_response.text
    assert "offending-list" not in list_response.text
    assert entered == []


def test_invalid_composite_skill_input_returns_safe_422_without_context_leak():
    # Contract item 6: skill conversion errors are safe 422s, do not enter the
    # handler, and return before any execution context is established.
    app = make_agent("pydantic-invalid-skill-http")
    entered = []

    @app.skill()
    def list_skill(items: List[M1]) -> dict:
        entered.append(True)
        return {"ok": True}

    initial_agent_context = app._current_execution_context
    response = TestClient(app, raise_server_exceptions=False).post(
        "/skills/list_skill", json={"items": [{"a": "offending-skill"}]}
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "Pydantic validation failed for skill 'list_skill'"
    }
    assert "offending-skill" not in response.text
    assert entered == []
    assert app._current_execution_context is initial_agent_context
    assert get_current_context() is None


def test_valid_model_and_plain_skill_conversion_paths():
    # Contract items 2 and 4: skill endpoints coerce model-bearing signatures
    # and leave signatures without models on the plain-input path.
    app = make_agent("pydantic-valid-skills-http")

    @app.skill()
    def model_skill(items: List[M1]) -> dict:
        return {
            "all_models": all(isinstance(item, M1) for item in items),
            "values": [item.a for item in items],
        }

    @app.skill()
    def plain_skill(value: str) -> dict:
        return {"value": value}

    client = TestClient(app, raise_server_exceptions=False)
    model_response = client.post(
        "/skills/model_skill", json={"items": [{"a": 1}, {"a": 2}]}
    )
    plain_response = client.post("/skills/plain_skill", json={"value": "unchanged"})

    assert model_response.status_code == 200
    assert model_response.json() == {"all_models": True, "values": [1, 2]}
    assert plain_response.status_code == 200
    assert plain_response.json() == {"value": "unchanged"}


def test_non_validation_skill_conversion_failure_uses_legacy_fallback(monkeypatch):
    # Contract item 10: an unexpected conversion-inspection failure still logs
    # in dev mode and passes the original keyword arguments to the skill.
    app = Agent(
        node_id="pydantic-skill-fallback-http",
        agentfield_server="",
        auto_register=False,
        dev_mode=True,
    )
    warnings = []

    @app.skill()
    def fallback_skill(value: str) -> dict:
        return {"value": value}

    def fail_conversion_check(func):
        raise RuntimeError("conversion inspection failed")

    monkeypatch.setattr("agentfield.agent.should_convert_args", fail_conversion_check)
    monkeypatch.setattr("agentfield.agent.log_warn", warnings.append)
    response = TestClient(app, raise_server_exceptions=False).post(
        "/skills/fallback_skill", json={"value": "original"}
    )

    assert response.status_code == 200
    assert response.json() == {"value": "original"}
    assert warnings == [
        "Failed to convert arguments for skill 'fallback_skill': "
        "conversion inspection failed"
    ]


def test_invalid_bare_model_input_still_returns_422():
    # Contract item 7: the pre-conversion input validator retains its existing
    # 422 behaviour for a bare model parameter.
    app = make_agent("pydantic-invalid-bare-http")
    entered = []

    @app.reasoner()
    def bare_model(m: M1) -> dict:
        entered.append(True)
        return {"ok": True}

    response = TestClient(app, raise_server_exceptions=False).post(
        "/reasoners/bare_model", json={"m": {"x": "offending-bare"}}
    )

    assert response.status_code == 422
    assert isinstance(response.json()["detail"], str)
    assert entered == []
