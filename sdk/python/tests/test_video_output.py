import base64

import pytest

from agentfield.multimodal_response import MultimodalResponse, VideoOutput


class TestVideoOutput:
    def test_create_with_url(self):
        v = VideoOutput(url="https://example.com/video.mp4")
        assert v.url == "https://example.com/video.mp4"
        assert v.mime_type == "video/mp4"

    def test_create_with_metadata(self):
        v = VideoOutput(
            url="https://example.com/video.mp4",
            duration=8.0,
            resolution="1080p",
            aspect_ratio="16:9",
            has_audio=True,
            cost_usd=0.40,
        )
        assert v.duration == 8.0
        assert v.resolution == "1080p"
        assert v.has_audio is True

    def test_get_bytes_from_base64(self):
        data = base64.b64encode(b"fake video data").decode()
        v = VideoOutput(data=data)
        assert v.get_bytes() == b"fake video data"

    def test_get_bytes_no_data_raises(self):
        v = VideoOutput()
        with pytest.raises(ValueError, match="No video data"):
            v.get_bytes()

    def test_save_from_base64(self, tmp_path):
        data = base64.b64encode(b"fake video").decode()
        v = VideoOutput(data=data)
        path = tmp_path / "test.mp4"
        v.save(path)
        assert path.read_bytes() == b"fake video"


class TestMultimodalResponseVideo:
    def test_has_videos_false_by_default(self):
        r = MultimodalResponse(text="hello")
        assert r.has_videos is False
        assert r.videos == []

    def test_has_videos_true(self):
        v = VideoOutput(url="https://example.com/video.mp4")
        r = MultimodalResponse(text="hello", videos=[v])
        assert r.has_videos is True
        assert len(r.videos) == 1

    def test_is_multimodal_with_video(self):
        v = VideoOutput(url="https://example.com/video.mp4")
        r = MultimodalResponse(text="hello", videos=[v])
        assert r.is_multimodal is True

    def test_backward_compat_no_videos_param(self):
        # Existing code that doesn't pass videos= should still work
        r = MultimodalResponse(text="hi", audio=None, images=[], files=[])
        assert r.videos == []
        assert r.has_videos is False

    def test_repr_includes_videos(self):
        v = VideoOutput(url="https://example.com/video.mp4")
        r = MultimodalResponse(text="hello", videos=[v])
        assert "videos=1" in repr(r)
