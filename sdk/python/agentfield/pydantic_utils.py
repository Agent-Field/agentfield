"""
Utility functions for automatic Pydantic model conversion in AgentField SDK.
Provides FastAPI-like automatic conversion of dictionary arguments to Pydantic model instances.
"""

import inspect
from typing import Any, Tuple, Union, get_args, get_origin, get_type_hints

from agentfield.logger import log_warn
from pydantic import BaseModel, ValidationError


def is_pydantic_model(type_hint: Any) -> bool:
    """
    Check if a type hint represents a Pydantic model.

    Args:
        type_hint: The type hint to check

    Returns:
        True if the type hint is a Pydantic model class
    """
    try:
        return inspect.isclass(type_hint) and issubclass(type_hint, BaseModel)
    except TypeError:
        return False


def is_optional_type(type_hint: Any) -> bool:
    """
    Check if a type hint represents an Optional type (Union[T, None]).

    Args:
        type_hint: The type hint to check

    Returns:
        True if the type hint is Optional[T]
    """
    origin = get_origin(type_hint)
    if origin is Union:
        args = get_args(type_hint)
        return len(args) == 2 and type(None) in args
    return False


def get_optional_inner_type(type_hint: Any) -> Any:
    """
    Extract the inner type from an Optional[T] type hint.

    Args:
        type_hint: The Optional type hint

    Returns:
        The inner type T from Optional[T]
    """
    if is_optional_type(type_hint):
        args = get_args(type_hint)
        return args[0] if args[0] is not type(None) else args[1]
    return type_hint


def convert_dict_to_model(data: Any, model_class: type) -> Any:
    """
    Convert a dictionary to a Pydantic model instance.

    Args:
        data: The data to convert (usually a dict)
        model_class: The Pydantic model class to convert to

    Returns:
        The converted Pydantic model instance, or the original data if conversion fails

    Raises:
        ValidationError: If the data doesn't match the model schema
    """
    if not isinstance(data, dict):
        # If it's already the correct type or not a dict, return as-is
        return data

    if not is_pydantic_model(model_class):
        # Not a Pydantic model, return original data
        return data

    try:
        return model_class(**data)
    except ValidationError as e:
        # Re-raise with more context
        raise ValidationError(
            f"Failed to convert dictionary to {model_class.__name__}: {e}",
            model=model_class,
        ) from e
    except Exception as e:
        # For any other errors, provide helpful context
        raise ValueError(
            f"Unexpected error converting dictionary to {model_class.__name__}: {e}"
        ) from e


def convert_function_args(
    func: callable, args: tuple, kwargs: dict
) -> Tuple[tuple, dict]:
    """
    Convert function arguments to Pydantic models based on the function's type hints.
    This mimics FastAPI's automatic request body parsing behavior.

    Args:
        func: The function whose arguments should be converted
        args: Positional arguments passed to the function
        kwargs: Keyword arguments passed to the function

    Returns:
        Tuple of (converted_args, converted_kwargs)

    Raises:
        ValidationError: If any argument fails Pydantic validation
    """
    try:
        # Get function signature and type hints
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        # Convert args to kwargs for easier processing
        bound_args = sig.bind_partial(*args, **kwargs)
        bound_args.apply_defaults()

        converted_kwargs = {}

        for param_name, value in bound_args.arguments.items():
            # Skip special parameters
            if param_name in ["self", "execution_context"]:
                converted_kwargs[param_name] = value
                continue

            # Get the type hint for this parameter
            type_hint = type_hints.get(param_name)
            if type_hint is None:
                # No type hint, keep original value
                converted_kwargs[param_name] = value
                continue

            # Handle Optional types
            actual_type = type_hint
            if is_optional_type(type_hint):
                if value is None:
                    converted_kwargs[param_name] = None
                    continue
                actual_type = get_optional_inner_type(type_hint)

            # Convert if it's a Pydantic model
            if is_pydantic_model(actual_type):
                try:
                    converted_kwargs[param_name] = convert_dict_to_model(
                        value, actual_type
                    )
                except ValidationError as e:
                    # Add parameter context to the error
                    raise ValidationError(
                        f"Validation error for parameter '{param_name}': {e}",
                        model=actual_type,
                    ) from e
            else:
                # Not a Pydantic model, keep original value
                converted_kwargs[param_name] = value

        # Convert back to args and kwargs based on original call pattern
        final_args = []
        final_kwargs = {}

        param_names = list(sig.parameters.keys())

        # Rebuild args for positional parameters
        for i, param_name in enumerate(param_names[: len(args)]):
            if param_name in converted_kwargs:
                final_args.append(converted_kwargs[param_name])
                del converted_kwargs[param_name]

        # Remaining parameters go to kwargs
        final_kwargs.update(converted_kwargs)

        return tuple(final_args), final_kwargs

    except Exception as e:
        # If conversion fails completely, return original args
        # This ensures backward compatibility
        if isinstance(e, ValidationError):
            raise  # Re-raise validation errors

        # For other errors, log and return original
        log_warn(f"Failed to convert arguments for {func.__name__}: {e}")
        return args, kwargs


def should_convert_args(func: callable) -> bool:
    """
    Determine if a function's arguments should be automatically converted.

    Args:
        func: The function to check

    Returns:
        True if the function has Pydantic model parameters that could benefit from conversion
    """
    try:
        type_hints = get_type_hints(func)
        sig = inspect.signature(func)

        for param_name, param in sig.parameters.items():
            if param_name in ["self", "execution_context"]:
                continue

            type_hint = type_hints.get(param_name)
            if type_hint is None:
                continue

            # Check if it's a Pydantic model or Optional Pydantic model
            actual_type = type_hint
            if is_optional_type(type_hint):
                actual_type = get_optional_inner_type(type_hint)

            if is_pydantic_model(actual_type):
                return True

        return False

    except Exception:
        # If we can't determine, err on the side of not converting
        return False
