"""Regression test for issue #587 -- ImageOutput.save() with data URLs."""

import base64

from agentfield.multimodal_response import ImageOutput


def test_issue_587(tmp_path):
    """Gemini-style data URLs should save locally without an HTTP request."""
    image_bytes = b"\x89PNG\r\n\x1a\nissue-587"
    image = ImageOutput(
        url=f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"
    )
    output_path = tmp_path / "gemini-image.png"

    image.save(output_path)

    assert output_path.read_bytes() == image_bytes
