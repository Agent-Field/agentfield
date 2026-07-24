import base64
import builtins
from types import ModuleType, SimpleNamespace

import pytest

from agentfield.multimodal_response import (
    AudioOutput,
    FileOutput,
    ImageOutput,
    MultimodalResponse,
    VideoOutput,
    _extract_image_from_data,
    _find_images_recursive,
    detect_multimodal_response,
)


@pytest.fixture
def fake_requests_module():
    class DummyResponse:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            return None

    def make_module(content):
        return SimpleNamespace(get=lambda url: DummyResponse(content))

    return make_module


def test_image_and_file_outputs_support_url_downloads(tmp_path, monkeypatch, fake_requests_module):
    monkeypatch.setitem(__import__("sys").modules, "requests", fake_requests_module(b"remote"))

    image = ImageOutput(url="https://example.com/test.png")
    image_path = tmp_path / "image.png"
    image.save(image_path)

    file_output = FileOutput(url="https://example.com/test.bin")
    file_path = tmp_path / "file.bin"
    file_output.save(file_path)

    assert image.get_bytes() == b"remote"
    assert file_output.get_bytes() == b"remote"
    assert image_path.read_bytes() == b"remote"
    assert file_path.read_bytes() == b"remote"


def test_output_objects_raise_for_missing_data():
    with pytest.raises(ValueError, match="No audio data available to save"):
        AudioOutput().save("unused.wav")

    with pytest.raises(ValueError, match="No audio data available"):
        AudioOutput().get_bytes()

    with pytest.raises(ValueError, match="No image data or URL available to save"):
        ImageOutput().save("unused.png")

    with pytest.raises(ValueError, match="No image data or URL available"):
        ImageOutput().get_bytes()

    with pytest.raises(ValueError, match="No file data or URL available to save"):
        FileOutput().save("unused.bin")

    with pytest.raises(ValueError, match="No file data or URL available"):
        FileOutput().get_bytes()


def test_audio_play_and_image_show_log_failures(monkeypatch):
    logged = {"warn": [], "error": []}
    monkeypatch.setattr("agentfield.multimodal_response.log_warn", logged["warn"].append)
    monkeypatch.setattr("agentfield.multimodal_response.log_error", logged["error"].append)

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pygame":
            raise ImportError("missing pygame")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    AudioOutput(data=base64.b64encode(b"abc").decode(), format="mp3").play()

    class FakeImageModule:
        @staticmethod
        def open(_):
            raise RuntimeError("bad image")

    monkeypatch.setattr(builtins, "__import__", original_import)
    monkeypatch.setitem(__import__("sys").modules, "PIL", ModuleType("PIL"))
    monkeypatch.setitem(__import__("sys").modules, "PIL.Image", FakeImageModule)
    monkeypatch.setattr(
        ImageOutput,
        "get_bytes",
        lambda self: b"img",
    )
    ImageOutput(url="https://example.com/test.png").show()

    assert logged["warn"] == ["Audio playback requires pygame: pip install pygame"]
    assert logged["error"] == ["Could not display image: bad image"]


def test_multimodal_response_save_all_and_flags(tmp_path):
    audio = AudioOutput(data=base64.b64encode(b"aud").decode(), format="wav")
    image = ImageOutput(b64_json=base64.b64encode(b"img").decode(), url="https://cdn/test.jpg")
    file_output = FileOutput(
        data=base64.b64encode(b"file").decode(),
        filename="artifact.bin",
        mime_type="application/octet-stream",
    )
    response = MultimodalResponse(
        text="hello",
        audio=audio,
        images=[image],
        files=[file_output],
    )

    saved = response.save_all(tmp_path, prefix="sample")

    assert str(response) == "hello"
    assert "audio=wav" in repr(response)
    assert "images=1" in repr(response)
    assert "files=1" in repr(response)
    assert response.has_audio is True
    assert response.has_images is True
    assert response.has_files is True
    assert response.is_multimodal is True
    assert (tmp_path / "sample_audio.wav").read_bytes() == b"aud"
    assert (tmp_path / "sample_image_0.jpg").read_bytes() == b"img"
    assert (tmp_path / "artifact.bin").read_bytes() == b"file"
    assert (tmp_path / "sample_text.txt").read_text() == "hello"
    assert set(saved) == {"audio", "image_0", "file_0", "text"}


def test_multimodal_response_save_all_sanitizes_video_filename(tmp_path):
    escaped_filename = f"{tmp_path.name}-escape.mp4"
    video = VideoOutput(
        data=base64.b64encode(b"video").decode(),
        filename=f"../../{escaped_filename}",
        mime_type="video/mp4",
    )
    response = MultimodalResponse(videos=[video])

    saved = response.save_all(tmp_path, prefix="sample")

    assert saved == {"video_0": str(tmp_path / escaped_filename)}
    assert (tmp_path / escaped_filename).read_bytes() == b"video"
    assert not (tmp_path.parent.parent / escaped_filename).exists()


def test_extract_and_find_images_cover_supported_shapes():
    data_url = "data:image/png;base64,Zm9v"
    object_image = SimpleNamespace(image_url=SimpleNamespace(url=data_url))
    direct_image = SimpleNamespace(url="https://example.com/a.png", revised_prompt="rp")

    assert _extract_image_from_data(None) is None
    assert _extract_image_from_data(direct_image).revised_prompt == "rp"
    assert _extract_image_from_data(object_image).b64_json == "Zm9v"
    assert _extract_image_from_data({"image_url": {"url": data_url}}).b64_json == "Zm9v"

    class Nested:
        def __init__(self):
            self.child = {"payload": {"url": "https://example.com/b.png"}}

        @property
        def broken(self):
            raise RuntimeError("ignore me")

    found = _find_images_recursive([Nested(), {"other": direct_image}], max_depth=5)
    assert [img.url for img in found] == ["https://example.com/b.png", "https://example.com/a.png"]
    assert _find_images_recursive({"url": "https://example.com/c.png"}, max_depth=0) == []


def test_detect_multimodal_response_completion_and_cost(monkeypatch):
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="hello",
                    audio=SimpleNamespace(data=base64.b64encode(b"aud").decode()),
                    images=[{"image_url": {"url": "data:image/png;base64,Zm9v"}}],
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        model="gpt-test",
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "litellm",
        SimpleNamespace(completion_cost=lambda completion_response: 1.25),
    )

    result = detect_multimodal_response(response)

    assert result.text == "hello"
    assert result.audio.get_bytes() == b"aud"
    assert result.images[0].b64_json == "Zm9v"
    assert result.usage == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    assert result.cost_usd == 1.25


def test_detect_multimodal_response_other_shapes_and_fallbacks():
    image_generation = SimpleNamespace(
        data=[SimpleNamespace(url="https://example.com/generated.png", revised_prompt="prompt")]
    )
    tts_response = SimpleNamespace(
        text="spoken",
        audio_data=base64.b64encode(b"wave").decode(),
        format="mp3",
    )

    class Dumpable:
        def model_dump(self):
            return {"ok": True}

    class BrokenDump:
        def model_dump(self):
            raise RuntimeError("boom")

    class RecursiveOnly:
        def __init__(self):
            self.payload = {"image_url": {"url": "https://example.com/nested.png"}}

    image_result = detect_multimodal_response(image_generation)
    assert image_result.images[0].url == "https://example.com/generated.png"

    tts_result = detect_multimodal_response(tts_response)
    assert tts_result.text == "spoken"
    assert tts_result.audio.format == "mp3"

    schema_result = detect_multimodal_response(Dumpable())
    assert '"ok": true' in schema_result.text

    broken_result = detect_multimodal_response(BrokenDump())
    assert "BrokenDump" in broken_result.text

    plain_result = detect_multimodal_response("plain text")
    assert plain_result.text == "plain text"

    recursive_result = detect_multimodal_response(RecursiveOnly())
    assert recursive_result.images[0].url == "https://example.com/nested.png"
