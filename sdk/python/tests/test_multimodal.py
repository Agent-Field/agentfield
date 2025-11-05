from haxen_sdk.multimodal import image_from_file, audio_from_file, file_from_path


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


def test_file_from_path(tmp_path):
    f = tmp_path / "d.txt"
    f.write_text("hello")
    fo = file_from_path(f)
    assert fo.file["url"].startswith("file://")
