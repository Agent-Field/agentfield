import base64
from haxen_sdk.multimodal_response import AudioOutput, ImageOutput, FileOutput


def test_audio_output_save_and_get_bytes(tmp_path):
    data = base64.b64encode(b"abc").decode()
    ao = AudioOutput(data=data, format="wav")
    p = tmp_path / "a.wav"
    ao.save(p)
    assert p.read_bytes() == b"abc"
    assert ao.get_bytes() == b"abc"


def test_image_output_save_from_b64(tmp_path):
    b64 = base64.b64encode(b"img").decode()
    io = ImageOutput(b64_json=b64)
    p = tmp_path / "i.png"
    io.save(p)
    assert p.read_bytes() == b"img"


def test_file_output_save_from_b64(tmp_path):
    b64 = base64.b64encode(b"file").decode()
    fo = FileOutput(data=b64)
    p = tmp_path / "f.bin"
    fo.save(p)
    assert p.read_bytes() == b"file"
