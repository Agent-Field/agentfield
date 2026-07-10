"""Regression test for ImageOutput.save() handling Gemini data URLs."""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

sdk_package = Path(__file__).resolve().parents[1] / "sdk" / "python" / "agentfield"
agentfield_package = ModuleType("agentfield")
agentfield_package.__path__ = [str(sdk_package)]
sys.modules.setdefault("agentfield", agentfield_package)

from agentfield.multimodal_response import ImageOutput


def test_issue_587(tmp_path, monkeypatch):
    """
    ImageOutput.save() should decode data: URLs locally.

    Gemini image models returned through OpenRouter can provide image payloads as
    data:image/png;base64,... URLs. These must not be passed to requests.get().
    """

    def fail_if_called(url):
        raise AssertionError(f"requests.get should not be called for data URLs: {url}")

    monkeypatch.setitem(
        __import__("sys").modules,
        "requests",
        SimpleNamespace(get=fail_if_called),
    )

    image_bytes = b"gemini image bytes"
    data_url = f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"
    image = ImageOutput(url=data_url)
    output_path = tmp_path / "out.png"

    image.save(output_path)

    assert output_path.read_bytes() == image_bytes
