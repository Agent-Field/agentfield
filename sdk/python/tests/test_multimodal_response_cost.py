"""Functional tests for cost/usage extraction in MultimodalResponse."""
import types
from unittest.mock import MagicMock, patch

from agentfield.multimodal_response import MultimodalResponse, detect_multimodal_response


class TestMultimodalResponseCostFields:
    """Test that MultimodalResponse exposes cost and usage data."""

    def test_cost_usd_defaults_to_none(self):
        resp = MultimodalResponse(text="hello")
        assert resp.cost_usd is None

    def test_usage_defaults_to_empty_dict(self):
        resp = MultimodalResponse(text="hello")
        assert resp.usage == {}

    def test_cost_usd_set_explicitly(self):
        resp = MultimodalResponse(text="hello", cost_usd=0.0042)
        assert resp.cost_usd == 0.0042

    def test_usage_set_explicitly(self):
        usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        resp = MultimodalResponse(text="hello", usage=usage)
        assert resp.usage == usage

    def test_backward_compat_no_cost_args(self):
        """Existing callers that don't pass cost/usage still work."""
        resp = MultimodalResponse(text="hello", raw_response=None)
        assert resp.text == "hello"
        assert resp.cost_usd is None
        assert resp.usage == {}
        assert str(resp) == "hello"


def _make_litellm_response(
    text="test response",
    model="gpt-4o",
    prompt_tokens=100,
    completion_tokens=50,
):
    """Build a fake litellm ModelResponse-like object."""
    usage = types.SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    message = types.SimpleNamespace(content=text, audio=None, images=None)
    choice = types.SimpleNamespace(message=message)
    return types.SimpleNamespace(
        choices=[choice],
        model=model,
        usage=usage,
        data=None,
    )


class TestDetectMultimodalResponseCostExtraction:
    """Test that detect_multimodal_response extracts cost/usage from responses."""

    def test_extracts_usage_from_response(self):
        resp = _make_litellm_response(prompt_tokens=200, completion_tokens=80)
        result = detect_multimodal_response(resp)
        assert result.usage["prompt_tokens"] == 200
        assert result.usage["completion_tokens"] == 80
        assert result.usage["total_tokens"] == 280

    def test_extracts_cost_when_litellm_available(self):
        resp = _make_litellm_response()
        mock_litellm = MagicMock()
        mock_litellm.completion_cost.return_value = 0.0035
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            result = detect_multimodal_response(resp)
        assert result.cost_usd == 0.0035
        mock_litellm.completion_cost.assert_called_once_with(
            completion_response=resp
        )

    def test_cost_is_none_when_litellm_raises(self):
        resp = _make_litellm_response()
        mock_litellm = MagicMock()
        mock_litellm.completion_cost.side_effect = Exception("unknown model")
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            result = detect_multimodal_response(resp)
        assert result.cost_usd is None
        # Usage should still be extracted even if cost estimation fails
        assert result.usage["total_tokens"] == 150

    def test_no_usage_for_string_response(self):
        result = detect_multimodal_response("plain text")
        assert result.usage == {}
        assert result.cost_usd is None

    def test_no_usage_when_response_lacks_usage_field(self):
        resp = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content="hi", audio=None, images=None
                    )
                )
            ],
            model="gpt-4o",
            data=None,
        )
        result = detect_multimodal_response(resp)
        assert result.usage == {}
        assert result.cost_usd is None

    def test_no_cost_when_response_lacks_model(self):
        """Cost estimation requires a model name."""
        usage = types.SimpleNamespace(
            prompt_tokens=100, completion_tokens=50, total_tokens=150
        )
        resp = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content="hi", audio=None, images=None
                    )
                )
            ],
            model=None,
            usage=usage,
            data=None,
        )
        result = detect_multimodal_response(resp)
        # Usage extracted, but cost not estimated without model
        assert result.usage["total_tokens"] == 150
        assert result.cost_usd is None

    def test_text_still_extracted_with_cost(self):
        resp = _make_litellm_response(text="important analysis result")
        result = detect_multimodal_response(resp)
        assert result.text == "important analysis result"
        assert result.usage["prompt_tokens"] == 100

    def test_backward_compat_all_existing_fields_preserved(self):
        """Adding cost/usage doesn't break existing field access."""
        resp = _make_litellm_response(text="hello world")
        result = detect_multimodal_response(resp)
        assert result.text == "hello world"
        assert result.raw_response is resp
        assert result.has_audio is False
        assert result.has_images is False
        assert result.has_files is False
        assert result.is_multimodal is False
        assert str(result) == "hello world"

    def test_zero_token_usage_still_extracted(self):
        """Edge case: response with zero tokens should still populate usage."""
        resp = _make_litellm_response(prompt_tokens=0, completion_tokens=0)
        result = detect_multimodal_response(resp)
        assert result.usage == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
