import pytest
from pydantic import BaseModel

from agentfield.agent_utils import AgentUtils


def test_detect_input_type_and_helpers(tmp_path):
    # URL detection
    assert AgentUtils.detect_input_type("https://example.com/img.png") == "image_url"
    assert AgentUtils.detect_input_type("https://example.com/") == "url"

    # File detection
    img = tmp_path / "a.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0fakejpg")
    assert AgentUtils.detect_input_type(str(img)) == "image_file"

    # Dict/list detection
    assert (
        AgentUtils.detect_input_type({"role": "user", "content": "hi"})
        == "message_dict"
    )
    assert (
        AgentUtils.detect_input_type([{"role": "user", "content": "hi"}])
        == "conversation_list"
    )

    # Helpers
    assert AgentUtils.is_image_url("x.PNG") is True
    assert AgentUtils.is_audio_url("a.mp3") is True
    assert AgentUtils.get_mime_type(".png") == "image/png"


def test_generate_skill_name_and_schema():
    name = AgentUtils.generate_skill_name("srv-1", "tool name!")
    assert name.startswith("srv_1_tool_name")

    tool = {
        "input_schema": {
            "properties": {
                "q": {"type": "string"},
                "n": {"type": "integer", "default": 1},
            },
            "required": ["q"],
        }
    }
    Model = AgentUtils.create_input_schema_from_mcp_tool("skill", tool)
    inst = Model(q="text")
    assert inst.q == "text"


def test_detect_input_type_bytes_and_structured(tmp_path):
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"RIFFxxxxWAVEfmt ")
    assert AgentUtils.detect_input_type(b"RIFFxxxxWAVEfmt ") == "audio_bytes"
    assert AgentUtils.detect_input_type(str(audio_path)) == "audio_file"

    payload = {"image": "https://example.com/foo.png"}
    assert AgentUtils.detect_input_type(payload) == "structured_input"
    assert (
        AgentUtils.detect_input_type(["hello", {"image": "data"}]) == "multimodal_list"
    )


def test_serialize_result_handles_complex_objects():
    class Nested(BaseModel):
        value: int

    class Wrapper:
        def __init__(self):
            self.model = Nested(value=3)
            self.items = [Nested(value=4), {"misc": Nested(value=5)}]

    serialized = AgentUtils.serialize_result(Wrapper())
    assert serialized["model"]["value"] == 3
    assert serialized["items"][0]["value"] == 4
    assert serialized["items"][1]["misc"]["value"] == 5


@pytest.mark.parametrize("available", [True, False])
def test_is_port_available(monkeypatch, available):
    calls = {}

    class DummySocket:
        def __init__(self, *args, **kwargs):
            calls["created"] = True

        def bind(self, addr):
            if not available:
                raise OSError("in use")
            calls["bind"] = addr

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("socket.socket", lambda *a, **k: DummySocket())

    result = AgentUtils.is_port_available(4321)
    assert result is available
    assert calls.get("created") is True


def test_detect_input_type_additional_branches(tmp_path):
    assert AgentUtils.detect_input_type("data:image/png;base64,AAA") == "image_base64"
    assert AgentUtils.detect_input_type("data:audio/wav;base64,AAA") == "audio_base64"

    doc = tmp_path / "doc.pdf"
    doc.write_bytes(b"%PDF-1.4")
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"ftypisom")

    assert AgentUtils.detect_input_type(str(doc)) == "document_file"
    assert AgentUtils.detect_input_type(str(video)) == "video_file"
    assert AgentUtils.detect_input_type(b"%PDF-1.4 more bytes") == "document_bytes"
    assert AgentUtils.detect_input_type(12345) == "unknown"


def test_create_input_schema_handles_empty_properties():
    Model = AgentUtils.create_input_schema_from_mcp_tool("skill", {"input_schema": {}})
    instance = Model()
    assert hasattr(instance, "data")
    assert instance.data is None


def test_serialize_result_falls_back_to_string():
    class BadDict(dict):
        def items(self):  # type: ignore[override]
            raise RuntimeError("cannot iterate")

    class Problem:
        def __init__(self):
            self.__dict__ = BadDict()

    problem = Problem()
    serialized = AgentUtils.serialize_result(problem)
    assert serialized == str(problem)
