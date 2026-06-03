"""
Unit tests for Pydantic model support in skills.

Tests that skills can now accept Pydantic models as parameters,
providing the same developer experience as reasoners.
"""

import pytest
from typing import Optional
from pydantic import BaseModel
from agentfield import Agent
from agentfield.pydantic_utils import should_convert_args, convert_function_args


# Test models
class UserRequest(BaseModel):
    """Test input model."""

    user_id: int
    name: str
    email: Optional[str] = None


class UserResponse(BaseModel):
    """Test output model."""

    user_id: int
    created: bool


class TestSkillPydanticModels:
    """Test suite for Pydantic model support in skills."""

    def test_should_convert_args_with_pydantic_model(self):
        """Test that should_convert_args identifies Pydantic model parameters."""

        def skill_with_model(request: UserRequest) -> UserResponse:
            return UserResponse(user_id=request.user_id, created=True)

        assert should_convert_args(skill_with_model) is True

    def test_should_convert_args_with_plain_params(self):
        """Test that should_convert_args returns False for plain parameters."""

        def skill_with_params(user_id: int, name: str) -> dict:
            return {"user_id": user_id, "name": name}

        assert should_convert_args(skill_with_params) is False

    def test_convert_function_args_with_pydantic_model(self):
        """Test that convert_function_args properly converts dict to Pydantic model."""

        def skill_with_model(request: UserRequest) -> UserResponse:
            return UserResponse(user_id=request.user_id, created=True)

        # Simulate input from control plane
        input_dict = {
            "request": {"user_id": 123, "name": "John Doe", "email": "john@example.com"}
        }

        args, kwargs = convert_function_args(skill_with_model, (), input_dict)

        # Verify conversion
        assert "request" in kwargs
        assert isinstance(kwargs["request"], UserRequest)
        assert kwargs["request"].user_id == 123
        assert kwargs["request"].name == "John Doe"
        assert kwargs["request"].email == "john@example.com"

    def test_convert_function_args_with_plain_params(self):
        """Test that plain parameters are passed through unchanged."""

        def skill_with_params(user_id: int, name: str, email: str = None) -> dict:
            return {"user_id": user_id, "name": name}

        input_dict = {"user_id": 456, "name": "Jane Doe", "email": "jane@example.com"}

        args, kwargs = convert_function_args(skill_with_params, (), input_dict)

        # Verify parameters are preserved
        assert "user_id" in kwargs or len(args) > 0
        if "user_id" in kwargs:
            assert kwargs["user_id"] == 456
            assert kwargs["name"] == "Jane Doe"
            assert kwargs["email"] == "jane@example.com"

    def test_skill_registration_with_pydantic_model(self):
        """Test that skills can be registered with Pydantic model parameters."""

        app = Agent(
            node_id="test-pydantic-skill",
            agentfield_server="http://localhost:8080",
        )

        @app.skill()
        async def create_user(request: UserRequest) -> UserResponse:
            """Skill with Pydantic model parameter."""
            return UserResponse(user_id=request.user_id, created=True)

        # Verify skill is registered
        assert len(app.skills) == 1
        skill = app.skills[0]
        assert skill["id"] == "create_user"

    def test_skill_registration_with_plain_params(self):
        """Test backward compatibility: skills with plain parameters still work."""

        app = Agent(
            node_id="test-plain-skill",
            agentfield_server="http://localhost:8080",
        )

        @app.skill()
        async def create_user(
            user_id: int, name: str, email: str = None
        ) -> UserResponse:
            """Skill with plain parameters."""
            return UserResponse(user_id=user_id, created=True)

        # Verify skill is registered
        assert len(app.skills) == 1
        skill = app.skills[0]
        assert skill["id"] == "create_user"

    def test_optional_pydantic_model(self):
        """Test that Optional[PydanticModel] parameters work correctly."""

        def skill_with_optional(request: Optional[UserRequest] = None) -> dict:
            if request is None:
                return {"status": "no request"}
            return {"user_id": request.user_id}

        # Test with None
        args, kwargs = convert_function_args(skill_with_optional, (), {"request": None})
        assert kwargs["request"] is None

        # Test with actual model
        input_dict = {"request": {"user_id": 789, "name": "Test User"}}
        args, kwargs = convert_function_args(skill_with_optional, (), input_dict)
        assert isinstance(kwargs["request"], UserRequest)
        assert kwargs["request"].user_id == 789


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
