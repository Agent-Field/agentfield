import socket
import pytest

from agentfield.utils import get_free_port
from agentfield.exceptions import AgentFieldError


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

    real_socket = socket.socket

    def fake_socket(*args, **kwargs):
        if attempts:
            outcome = attempts.pop(0)
            return DummySocket(outcome)
        return real_socket(*args, **kwargs)

    monkeypatch.setattr("agentfield.utils.socket.socket", fake_socket)

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

    monkeypatch.setattr(
        "agentfield.utils.socket.socket", lambda *args, **kwargs: DummySocket()
    )

    with pytest.raises(AgentFieldError):
        get_free_port(start_port=5, end_port=6)
