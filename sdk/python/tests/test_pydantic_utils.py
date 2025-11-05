from pydantic import BaseModel
from haxen_sdk.pydantic_utils import (
    is_pydantic_model,
    is_optional_type,
    get_optional_inner_type,
    convert_dict_to_model,
    convert_function_args,
    should_convert_args,
)


class Inner(BaseModel):
    x: int


def test_is_pydantic_and_optional_helpers():
    assert is_pydantic_model(Inner) is True
    from typing import Optional

    opt = Optional[Inner]
    assert is_optional_type(opt) is True
    assert get_optional_inner_type(opt) is Inner


def test_convert_dict_to_model_success_and_error():
    out = convert_dict_to_model({"x": 5}, Inner)
    assert isinstance(out, Inner)
    assert out.x == 5
    # Should raise some validation-related exception from conversion
    raised = False
    try:
        convert_dict_to_model({"x": "bad"}, Inner)
    except Exception:
        raised = True
    assert raised


class WithModel(BaseModel):
    inner: Inner


def func_with_model(inner: Inner, y: int):
    return inner.x + y


def test_convert_function_args_and_should_convert():
    assert should_convert_args(func_with_model) is True
    # Provide dict for inner; expect conversion
    args, kwargs = convert_function_args(
        func_with_model, tuple(), {"inner": {"x": 2}, "y": 3}
    )
    assert isinstance(kwargs["inner"], Inner)
    assert kwargs["inner"].x == 2
