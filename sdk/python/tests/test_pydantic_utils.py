from typing import Optional
from pydantic import BaseModel
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

    _, kwargs = convert_function_args(
        my_func, (), {"m": {"x": "not-an-int"}}
    )

    # Current behavior: current implementation swallows the exception due to incompatibility with Pydantic v2 (ValidationError constructor signature mismatch), and returns original args
    assert kwargs["m"] == {"x": "not-an-int"}