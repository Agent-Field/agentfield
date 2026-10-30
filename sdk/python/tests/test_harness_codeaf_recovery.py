"""Review regressions: recovery, isolated state, read-only doctor and transport."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.parse import urlsplit

import pytest

from agentfield.exceptions import HarnessProviderUnavailable
from agentfield.harness import harness_doctor
from agentfield.harness import _codeaf_install as install
from agentfield.harness.providers import codeaf
from agentfield.harness.providers.codeaf import CodeafProvider


def write_cli(path: Path, *, legacy: bool = False, managed: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        {"text": "recovered"}
        if legacy
        else {"ok": True, "stop": "done", "answer": "recovered"}
    )
    path.write_text(f"#!{sys.executable}\nprint({json.dumps(payload)!r})\n")
    path.chmod(0o755)
    if managed:
        (path.parent / ".codeaf.version").write_text("v0.8.2\n")
    return str(path)


@pytest.fixture(autouse=True)
def isolated_resolution(monkeypatch, tmp_path):
    for key in (
        "CODEAF_BIN",
        "AFORGE_BIN",
        "CODEAF_HOME",
        "AGENTFIELD_SKIP_CODEAF",
        "AGENTFIELD_SKIP_AFORGE",
        "AGENTFIELD_CODEAF_COMMAND",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "all_proxy",
        "REQUEST_METHOD",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AGENTFIELD_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HOME", str(tmp_path / "user-home"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-codeaf-key")
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(install.tempfile, "gettempdir", lambda: str(tmp_path / "temp"))
    monkeypatch.setattr(install.platform, "system", lambda: "Linux")
    monkeypatch.setattr(install.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(urllib.request, "_opener", None)
    for name in (
        "_resolution_cache",
        "_resolution_lock",
        "_resolution_task",
        "_concurrency_sem",
    ):
        monkeypatch.setattr(CodeafProvider, name, None, raising=False)
    monkeypatch.setattr(codeaf, "_download_retry_at", 0.0, raising=False)
    monkeypatch.setattr(codeaf, "_last_download_error", None, raising=False)


@pytest.fixture
def release(monkeypatch):
    body = (
        f"#!{sys.executable}\nimport json,os\n"
        "print(json.dumps(dict(ok=True,stop='done',answer=json.dumps(dict(CODEAF_HOME=os.environ['CODEAF_HOME'])))))\n"
    ).encode()
    state = SimpleNamespace(mode="ok", body=body, requests=[], connects=[])

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state.requests.append(self.path)
            if state.mode == "503":
                self.send_response(503)
                self.end_headers()
                return
            if state.mode == "partial":
                self.send_response(200)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                self.wfile.write(b"4\r\nab")
                self.close_connection = True
                return
            if state.mode == "redirect" and urlsplit(self.path).path != "/payload":
                self.send_response(302)
                self.send_header("Location", "http://redirect-codeaf.invalid/payload")
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(state.body)

        def do_CONNECT(self):
            state.connects.append(self.path)
            self.send_error(502, "test proxy refuses tunnel")

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv("AGENTFIELD_CODEAF_BASE_URL", state.url)
    monkeypatch.setitem(
        install.CODEAF_SHA256, "codeaf-linux-amd64", hashlib.sha256(body).hexdigest()
    )
    yield state
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.mark.parametrize(
    "location", ["managed", "path", "legacy", "override", "env"]
)
async def test_failed_resolution_recovers_without_cache_reset(
    release, monkeypatch, tmp_path, location
):
    """#1/A1: repair any lookup location after a real HTTP 503."""
    release.mode = "503"
    provider = CodeafProvider()
    with pytest.raises(HarnessProviderUnavailable, match="503"):
        await provider.execute("p", {})
    (home,) = install.managed_paths()
    target = {"managed": home, "legacy": home.with_name("aforge")}.get(
        location, tmp_path / "path" / "codeaf"
    )
    binary = await asyncio.to_thread(
        write_cli,
        target,
        legacy=location == "legacy",
        managed=location == "managed",
    )
    if location == "path":
        monkeypatch.setenv("PATH", str(target.parent))
    elif location == "env":
        monkeypatch.setenv("CODEAF_BIN", binary)
    for current in (
        provider,
        CodeafProvider(binary if location == "override" else None),
    ):
        if location == "override":
            current = CodeafProvider(binary)
        result = await current.execute("p", {})
        assert result.result == "recovered"
        assert result.provider == ("aforge" if location == "legacy" else "codeaf")
    assert len(release.requests) == 1


async def test_failed_download_retry_is_throttled_but_resolution_is_not(
    release, monkeypatch
):
    """#1/A1: concurrent failures share an attempt; retry starts at 60 seconds."""
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(codeaf.time, "monotonic", lambda: clock.now)
    release.mode = "503"
    results = await asyncio.gather(
        *(CodeafProvider().execute("p", {}) for _ in range(12)), return_exceptions=True
    )
    assert all(isinstance(result, HarnessProviderUnavailable) for result in results)
    assert len(release.requests) == 1
    release.mode = "ok"
    clock.now = 159.9
    with pytest.raises(HarnessProviderUnavailable, match="503"):
        await CodeafProvider().execute("p", {})
    assert len(release.requests) == 1
    clock.now = 160.0
    assert not (await CodeafProvider().execute("p", {})).is_error
    assert len(release.requests) == 2


async def test_removing_skip_allows_immediate_download(release, monkeypatch):
    monkeypatch.setenv("AGENTFIELD_SKIP_CODEAF", "1")
    with pytest.raises(HarnessProviderUnavailable):
        await CodeafProvider().execute("p", {})
    monkeypatch.delenv("AGENTFIELD_SKIP_CODEAF")
    assert not (await CodeafProvider().execute("p", {})).is_error
    assert len(release.requests) == 1


async def test_cancelled_caller_keeps_one_install_in_flight(release, monkeypatch):
    started, finish = threading.Event(), threading.Event()
    calls = []
    download = codeaf.install_codeaf

    def blocked_install():
        calls.append("install")
        started.set()
        assert finish.wait(5)
        return download()

    monkeypatch.setattr(codeaf, "install_codeaf", blocked_install)
    first = asyncio.create_task(CodeafProvider().execute("p", {}))
    second = None
    try:
        assert await asyncio.to_thread(started.wait, 2)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(CodeafProvider().execute("p", {}))
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(second), timeout=0.05)
        finish.set()
        assert not (await second).is_error
        assert calls == ["install"]
        assert len(release.requests) == 1
    finally:
        finish.set()
        await asyncio.gather(
            first, *([second] if second else []), return_exceptions=True
        )


@pytest.mark.parametrize("source", ["caller", "overlay"])
@pytest.mark.parametrize("value", ["", "/explicit/readonly"])
async def test_explicit_state_directory_is_used_as_is(monkeypatch, source, value):
    monkeypatch.setattr(
        Path,
        "mkdir",
        Mock(side_effect=AssertionError("must not create explicit state")),
    )
    options = {"env": {"CODEAF_HOME": value}} if source == "overlay" else {}
    if source == "caller":
        monkeypatch.setenv("CODEAF_HOME", value)
    overlay = await asyncio.to_thread(codeaf._child_environment, options, None)
    assert {**os.environ, **overlay}["CODEAF_HOME"] == value


@pytest.mark.parametrize(
    "level", ["override", "managed", "path", "missing", "bad-override"]
)
async def test_doctor_uses_execution_lookup_order_without_downloading(
    monkeypatch, tmp_path, level
):
    """#6/A3: each earlier location wins even when later binaries exist."""
    paths = [
        tmp_path / "vendor/codeaf",
        *install.managed_paths(),
        tmp_path / "path/codeaf",
    ]
    index = {
        "override": 0,
        "managed": 1,
        "path": 2,
        "missing": 3,
        "bad-override": 1,
    }[level]
    for i, path in enumerate(paths):
        if i >= index:
            await asyncio.to_thread(write_cli, path, managed=i == 1)
    environment = {
        "AGENTFIELD_HOME": str(install.agentfield_home()),
        "PATH": str(paths[2].parent),
    }
    if level in {"override", "bad-override"}:
        environment["CODEAF_BIN"] = str(paths[0])
    # A supplied environment controls the lookup rather than ambient CODEAF_BIN/PATH.
    monkeypatch.setenv("CODEAF_BIN", "/ambient/missing")
    monkeypatch.setattr(
        install,
        "install_codeaf",
        Mock(side_effect=AssertionError("doctor must not download")),
    )
    commands = []

    async def version(command):
        commands.append(command)
        return "codeaf v0.8.2 built fixture"

    [report] = await harness_doctor(["codeaf"], env=environment, version_probe=version)
    expected = None if level in {"missing", "bad-override"} else str(paths[index])
    assert report.binary == expected
    assert report.usable == (expected is not None)
    assert commands == ([[expected, "version"]] if expected else [])


@pytest.mark.parametrize("fallback", ["codeaf", "aforge", "missing"])
async def test_truncated_chunked_download_continues_resolution(
    release, monkeypatch, tmp_path, fallback
):
    """#8: a real incomplete chunked response must reach PATH/legacy fallback."""
    release.mode = "partial"
    if fallback != "missing":
        await asyncio.to_thread(
            write_cli, tmp_path / fallback, legacy=fallback == "aforge"
        )
        monkeypatch.setenv("PATH", str(tmp_path))
        result = await CodeafProvider().execute("p", {})
        assert result.result == "recovered"
        assert result.provider == fallback
    else:
        with pytest.raises(
            HarnessProviderUnavailable, match="af codeaf ensure.*IncompleteRead"
        ):
            await CodeafProvider().execute("p", {})
    assert not await asyncio.to_thread(
        lambda: list(install.managed_paths()[0].parent.iterdir())
    )


@pytest.mark.parametrize("key", ["HTTP_PROXY", "http_proxy"])
@pytest.mark.parametrize("corrupt", [False, True])
def test_http_proxy_redirects_still_verify_embedded_hash(
    release, monkeypatch, key, corrupt
):
    """A4: only the proxy can reach these deliberately unresolvable origins."""
    release.mode = "redirect"
    monkeypatch.setenv(key, release.url)
    monkeypatch.setenv(
        "AGENTFIELD_CODEAF_BASE_URL", "http://codeaf-origin.invalid/v0.8.2"
    )
    if corrupt:
        release.body = b"untrusted binary"
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            install.install_codeaf()
        assert not install.managed_paths()[0].exists()
    else:
        assert install.install_codeaf() == str(install.managed_paths()[0])
    assert release.requests == [
        "http://codeaf-origin.invalid/v0.8.2/codeaf-linux-amd64",
        "http://redirect-codeaf.invalid/payload",
    ]


@pytest.mark.parametrize("key", ["HTTPS_PROXY", "https_proxy"])
def test_https_download_uses_configured_connect_proxy(release, monkeypatch, key):
    monkeypatch.setenv(key, release.url)
    monkeypatch.setenv(
        "AGENTFIELD_CODEAF_BASE_URL", "https://codeaf-origin.invalid/v0.8.2"
    )
    with pytest.raises(urllib.error.URLError, match="502"):
        install.install_codeaf()
    assert release.connects == ["codeaf-origin.invalid:443"]


@pytest.mark.parametrize("key", ["NO_PROXY", "no_proxy"])
def test_no_proxy_change_is_honored_after_an_earlier_download(
    release, monkeypatch, tmp_path, key
):
    # Prime urllib's global opener before adding proxy configuration.
    install.install_codeaf()
    monkeypatch.setenv("AGENTFIELD_HOME", str(tmp_path / "next-home"))
    monkeypatch.setenv("HTTP_PROXY", release.url)
    monkeypatch.setenv(
        "AGENTFIELD_CODEAF_BASE_URL", "http://codeaf-origin.invalid/v0.8.2"
    )
    release.mode = "503"
    with pytest.raises(urllib.error.HTTPError, match="503"):
        install.install_codeaf()
    release.mode = "ok"
    monkeypatch.setenv("AGENTFIELD_CODEAF_BASE_URL", release.url)
    monkeypatch.setenv(key, "127.0.0.1")
    assert install.install_codeaf() == str(install.managed_paths()[0])
    assert len(release.requests) == 3
    assert release.requests[-1] == "/codeaf-linux-amd64"


async def test_codeaf_bin_runs_with_unwritable_default_state(monkeypatch, tmp_path):
    """C2: state creation failure must reach the explicitly selected child."""
    binary = tmp_path / "explicit-codeaf"
    binary.write_text(
        f"#!{sys.executable}\nimport json,os\n"
        "print(json.dumps(dict(ok=True,stop='done',answer=os.environ['CODEAF_HOME'])))\n"
    )
    binary.chmod(0o755)
    home = install.agentfield_home()
    home.write_text("not a directory")
    monkeypatch.setenv("CODEAF_BIN", str(binary))
    result = await CodeafProvider().execute("p", {})
    assert not result.is_error
    assert result.result == str(home / "codeaf")
    assert not (tmp_path / "temp").exists()
