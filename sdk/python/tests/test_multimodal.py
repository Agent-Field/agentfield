from agentfield.multimodal import (
    audio_from_file,
    file_from_path,
    image_from_file,
    video_from_file,
)


def test_image_from_file_and_audio_from_file(tmp_path):
    # Create tiny fake image bytes
    img = tmp_path / "x.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    im = image_from_file(img)
    assert im.type == "image_url"
    assert isinstance(im.image_url, dict)
    assert im.image_url["url"].startswith("data:image/")

    # Fake wav header
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFFxxxxWAVEfmt ")
    au = audio_from_file(wav)
    assert au.type == "input_audio"
    assert "data" in au.input_audio

    mp4 = tmp_path / "v.mp4"
    mp4.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    vi = video_from_file(mp4)
    assert vi.type == "video_url"
    assert vi.video_url["url"].startswith("data:video/mp4;base64,")


def test_file_from_path(tmp_path):
    f = tmp_path / "d.txt"
    f.write_text("hello")
    fo = file_from_path(f)
    assert fo.file["url"].startswith("file://")
