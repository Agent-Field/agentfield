import pytest
from typing import List, Optional, Sequence, Union
from pydantic import BaseModel, ValidationError
from agentfield.pydantic_utils import (
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


class MyModel(BaseModel):
    x: int


def test_convert_positional_args():
    def my_func(m: MyModel):
        return m

    args, kwargs = convert_function_args(my_func, ({"x": 1},), {})

    assert kwargs == {}
    assert len(args) == 1
    assert isinstance(args[0], MyModel)
    assert args[0].x == 1


def test_convert_optional_model_none():
    def my_func(m: Optional[MyModel]):
        return m

    args, kwargs = convert_function_args(my_func, (), {"m": None})

    assert args == ()
    assert kwargs["m"] is None


def test_convert_skips_self_and_context():
    class DummyCallable:
        def method(self, execution_context, m: MyModel):
            return execution_context, m

    instance = DummyCallable()
    raw_context = {"x": 99}
    raw_model = {"x": 5}

    args, kwargs = convert_function_args(
        instance.method, (), {"execution_context": raw_context, "m": raw_model}
    )

    assert args == ()
    assert kwargs["execution_context"] is raw_context
    assert isinstance(kwargs["m"], MyModel)
    assert kwargs["m"].x == 5


def test_convert_retains_untyped_params():
    def my_func(untyped, typed: MyModel):
        return untyped, typed

    untyped_value = {"left": "as-is"}
    args, kwargs = convert_function_args(
        my_func, (), {"untyped": untyped_value, "typed": {"x": 7}}
    )

    assert args == ()
    assert kwargs["untyped"] is untyped_value
    assert isinstance(kwargs["typed"], MyModel)
    assert kwargs["typed"].x == 7


def test_convert_validation_error_propagation():
    def my_func(m: MyModel):
        return m

    # A model parameter that fails validation must surface a pydantic
    # ValidationError (callers intercept that type specifically), not be
    # silently returned as the raw dict.
    with pytest.raises(ValidationError):
        convert_function_args(my_func, (), {"m": {"x": "not-an-int"}})


# --- #1034: complex type hints (unions of models, containers of models) ---


class M1(BaseModel):
    a: int


class M2(BaseModel):
    b: int


def test_convert_multi_arg_union_of_models():
    def f(item: Union[M1, M2, None] = None):
        return item

    assert should_convert_args(f) is True
    _, kwargs = convert_function_args(f, (), {"item": {"a": 1}})
    assert isinstance(kwargs["item"], M1)
    assert kwargs["item"].a == 1

    # None still passes through untouched.
    _, kwargs = convert_function_args(f, (), {"item": None})
    assert kwargs["item"] is None


def test_convert_list_of_models():
    def f(items: Optional[List[M1]] = None):
        return items

    assert should_convert_args(f) is True
    _, kwargs = convert_function_args(f, (), {"items": [{"a": 1}, {"a": 2}]})
    assert all(isinstance(x, M1) for x in kwargs["items"])
    assert [x.a for x in kwargs["items"]] == [1, 2]


def test_convert_union_of_list_of_models():
    def f(items: Union[List[M1], List[M2], None] = None):
        return items

    assert should_convert_args(f) is True
    _, kwargs = convert_function_args(f, (), {"items": [{"a": 1}]})
    assert all(isinstance(x, M1) for x in kwargs["items"])
    assert kwargs["items"][0].a == 1


def test_convert_sequence_of_optional_models():
    def f(seq: Sequence[Union[M1, None]] = ()):
        return seq

    assert should_convert_args(f) is True
    _, kwargs = convert_function_args(f, (), {"seq": [{"a": 1}, None]})
    assert isinstance(kwargs["seq"][0], M1)
    assert kwargs["seq"][0].a == 1
    assert kwargs["seq"][1] is None


def test_convert_nested_model_roundtrip():
    class Outer(BaseModel):
        inner: M1
        tags: List[str] = []

    def f(outer: Outer):
        return outer

    _, kwargs = convert_function_args(
        f, (), {"outer": {"inner": {"a": 5}, "tags": ["x"]}}
    )
    assert isinstance(kwargs["outer"], Outer)
    assert isinstance(kwargs["outer"].inner, M1)
    assert kwargs["outer"].inner.a == 5
    assert kwargs["outer"].tags == ["x"]


def test_container_of_models_validation_error_propagates():
    def f(items: List[M1]):
        return items

    with pytest.raises(ValidationError):
        convert_function_args(f, (), {"items": [{"a": "bad"}]})


def test_non_model_params_untouched_for_complex_hints():
    def f(nums: List[int], flag: Optional[str] = None):
        return nums, flag

    # No pydantic model anywhere: conversion should not trigger and values
    # pass through unchanged.
    assert should_convert_args(f) is False
    original = [1, 2, 3]
    _, kwargs = convert_function_args(f, (), {"nums": original, "flag": "hi"})
    assert kwargs["nums"] is original
    assert kwargs["flag"] == "hi"
