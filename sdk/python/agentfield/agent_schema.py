from typing import Any, Dict, Union

class _AgentSchemaMixin:
    def _types_to_json_schema(self, input_types: Dict[str, tuple]) -> Dict:
        """Convert Python types dict to JSON schema (on-demand generation)."""
        properties = {}
        required = []

        for name, (typ, default) in input_types.items():
            properties[name] = self._type_to_json_schema(typ)
            if default is ...:  # Required field (no default)
                required.append(name)

        schema = {
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required
        return schema
    
    def _type_to_json_schema(self, typ: type) -> Dict:
        """Convert a Python type to JSON schema."""
        # Handle None/NoneType
        if typ is None or typ is type(None):
            return {"type": "null"}

        # Handle basic types
        type_map = {
            str: {"type": "string"},
            int: {"type": "integer"},
            float: {"type": "number"},
            bool: {"type": "boolean"},
            list: {"type": "array"},
            dict: {"type": "object"},
            bytes: {"type": "string", "format": "binary"},
        }

        if typ in type_map:
            return type_map[typ]

        # Handle Pydantic models
        if hasattr(typ, "model_json_schema"):
            return typ.model_json_schema()

        # Handle typing constructs (List, Dict, Optional, etc.)
        origin = getattr(typ, "__origin__", None)
        if origin is list:
            args = getattr(typ, "__args__", (Any,))
            return {
                "type": "array",
                "items": self._type_to_json_schema(args[0]) if args else {},
            }
        if origin is dict:
            return {"type": "object", "additionalProperties": True}
        if origin is Union:
            args = getattr(typ, "__args__", ())
            # Handle Optional (Union with None)
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return self._type_to_json_schema(non_none[0])
            return {"anyOf": [self._type_to_json_schema(a) for a in args]}

        # Default fallback
        return {"type": "object"}

    def _validate_handler_input(
        self, data: dict, input_types: Dict[str, tuple]
    ) -> dict:
        """
        Validate input data against expected types at runtime.

        Replaces Pydantic model validation with lightweight runtime validation.
        Saves ~1.5-2 KB per handler by not creating Pydantic classes.

        Args:
            data: Raw input dict from request body
            input_types: Dict mapping field names to (type, default) tuples

        Returns:
            Validated dict with type coercion applied

        Raises:
            ValueError: If required field is missing or type conversion fails
        """
        result = {}

        for name, (expected_type, default) in input_types.items():
            # Check if field is present
            if name not in data:
                if default is ...:  # Required field (no default)
                    raise ValueError(f"Missing required field: {name}")
                result[name] = default
                continue

            value = data[name]

            # Handle None values
            if value is None:
                # Check if Optional type
                origin = getattr(expected_type, "__origin__", None)
                if origin is Union:
                    args = getattr(expected_type, "__args__", ())
                    if type(None) in args:
                        result[name] = None
                        continue
                # Not Optional, use default if available
                if default is not ...:
                    result[name] = default
                    continue
                raise ValueError(f"Field '{name}' cannot be None")

            # Type coercion for basic types
            try:
                # Get the actual type (unwrap Optional)
                actual_type = expected_type
                origin = getattr(expected_type, "__origin__", None)
                if origin is Union:
                    args = getattr(expected_type, "__args__", ())
                    non_none = [a for a in args if a is not type(None)]
                    if len(non_none) == 1:
                        actual_type = non_none[0]

                # Basic type coercion
                if actual_type is int:
                    result[name] = int(value)
                elif actual_type is float:
                    result[name] = float(value)
                elif actual_type is str:
                    result[name] = str(value)
                elif actual_type is bool:
                    if isinstance(value, bool):
                        result[name] = value
                    elif isinstance(value, str):
                        result[name] = value.lower() in ("true", "1", "yes")
                    else:
                        result[name] = bool(value)
                elif (
                    actual_type is dict
                    or getattr(actual_type, "__origin__", None) is dict
                ):
                    if not isinstance(value, dict):
                        raise ValueError(f"Field '{name}' must be a dict")
                    result[name] = dict(value)
                elif (
                    actual_type is list
                    or getattr(actual_type, "__origin__", None) is list
                ):
                    if not isinstance(value, list):
                        raise ValueError(f"Field '{name}' must be a list")
                    result[name] = list(value)
                elif hasattr(actual_type, "model_validate"):
                    # Pydantic model - use its validation
                    result[name] = actual_type.model_validate(value)
                else:
                    # Pass through for complex/unknown types
                    result[name] = value
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid value for field '{name}': {e}")

        return result