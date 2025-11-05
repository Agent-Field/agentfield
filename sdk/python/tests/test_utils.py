import socket
import pytest

from haxen_sdk.utils import get_free_port


def test_get_free_port_iterates_until_success(monkeypatch):
    attempts = [OSError, None]

    class DummySocket:
        def __init__(self, outcome):
            self._outcome = outcome

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def bind(self, addr):
            if self._outcome is OSError:
                raise OSError("port in use")

    def fake_socket(*args, **kwargs):
        outcome = attempts.pop(0)
        return DummySocket(outcome)

    monkeypatch.setattr(socket, "socket", fake_socket)

    port = get_free_port(start_port=1000, end_port=1001)
    assert port == 1001


def test_get_free_port_raises_when_exhausted(monkeypatch):
    class DummySocket:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def bind(self, addr):
            raise OSError("busy")

    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: DummySocket())

    with pytest.raises(RuntimeError):
        get_free_port(start_port=5, end_port=6)
