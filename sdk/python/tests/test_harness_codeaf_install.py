"""V5/V6: local HTTP downloads exercise the embedded-pin installation contract."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from agentfield.harness import _codeaf_install as install
from agentfield.harness.providers.codeaf import CodeafProvider


@pytest.fixture
def local_release(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(install.logger, "handlers", [caplog.handler])
    monkeypatch.setattr(install.logger, "propagate", False)
    caplog.set_level(logging.INFO, logger="agentfield.harness.codeaf")
    data = f'#!{sys.executable}\nprint(\'{{"ok":true,"stop":"done","answer":"downloaded"}}\')\n'.encode()
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    for key in (
        "AGENTFIELD_SKIP_CODEAF",
        "AGENTFIELD_SKIP_AFORGE",
        "CODEAF_BIN",
        "AFORGE_BIN",
        "AGENTFIELD_CODEAF_COMMAND",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AGENTFIELD_HOME", str(tmp_path / "home"))
    monkeypatch.setenv(
        "AGENTFIELD_CODEAF_BASE_URL", f"http://127.0.0.1:{server.server_port}/v0.8.2/"
    )
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path / "temp"))
    monkeypatch.setattr(install.platform, "system", lambda: "Linux")
    monkeypatch.setattr(install.platform, "machine", lambda: "x86_64")
    # Test-only hash injection: production always verifies its embedded pins.
    for asset in install.CODEAF_SHA256:
        monkeypatch.setitem(
            install.CODEAF_SHA256, asset, hashlib.sha256(data).hexdigest()
        )
    for attr in (
        "_resolution_cache",
        "_resolution_lock",
        "_resolution_task",
        "_concurrency_sem",
    ):
        monkeypatch.setattr(CodeafProvider, attr, None)
    monkeypatch.setattr("agentfield.harness.providers.codeaf._download_retry_at", 0.0)
    monkeypatch.setattr(
        "agentfield.harness.providers.codeaf._last_download_error", None
    )
    yield data, requests
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.mark.parametrize(
    "system,machine,asset",
    [
        ("Linux", "x86_64", "codeaf-linux-amd64"),
        ("Linux", "aarch64", "codeaf-linux-arm64"),
        ("Darwin", "amd64", "codeaf-darwin-amd64"),
        ("Darwin", "arm64", "codeaf-darwin-arm64"),
        ("Windows", "AMD64", "codeaf-windows-amd64.exe"),
        ("Windows", "ARM64", "codeaf-windows-arm64.exe"),
    ],
)
def test_download_asset_hash_marker_and_atomic_order(
    local_release, monkeypatch, system, machine, asset
):
    data, requests = local_release
    monkeypatch.setattr(install.platform, "system", lambda: system)
    monkeypatch.setattr(install.platform, "machine", lambda: machine)
    target = install.managed_paths()[0]
    replace = os.replace
    observed = []

    def checked_replace(src, dst):
        source = Path(src)
        assert source.parent == target.parent
        assert source != target
        assert source.read_bytes() == data
        assert source.stat().st_mode & 0o777 == 0o755
        assert not (target.parent / ".codeaf.version").exists()
        observed.append(source)
        replace(src, dst)

    monkeypatch.setattr(install.os, "replace", checked_replace)
    assert install.install_codeaf() == str(target)
    assert requests == [f"/v0.8.2/{asset}"]
    assert target.read_bytes() == data
    assert (target.parent / ".codeaf.version").read_text() == "v0.8.2\n"
    assert install.is_installed(target)
    assert install.install_codeaf() == str(target)
    assert len(requests) == len(observed) == 1
    assert not observed[0].exists()


@pytest.mark.parametrize("reason", ["hash", "size", "http"])
def test_rejected_download_leaves_no_binary_marker_or_temp(
    local_release, monkeypatch, reason
):
    _, requests = local_release
    if reason == "hash":
        monkeypatch.setitem(install.CODEAF_SHA256, "codeaf-linux-amd64", "0" * 64)
    elif reason == "size":
        monkeypatch.setattr(install, "MAX_DOWNLOAD_BYTES", 4)
    else:

        def fail(*_, **__):
            raise OSError("network unavailable")

        monkeypatch.setattr(install.urllib.request.OpenerDirector, "open", fail)
    with pytest.raises(
        (ValueError, OSError),
        match={
            "hash": "SHA-256 mismatch",
            "size": "128 MiB",
            "http": "network unavailable",
        }[reason],
    ):
        install.install_codeaf()
    target = install.managed_paths()[0]
    assert not target.exists()
    assert not list(target.parent.iterdir())
    assert len(requests) == (0 if reason == "http" else 1)


@pytest.mark.parametrize("failure", ["mkdir", "write"])
async def test_unwritable_home_reports_install_error_without_temp(local_release, monkeypatch, failure):
    (home,) = install.managed_paths()
    mkstemp = install.tempfile.mkstemp
    mkdir = Path.mkdir

    def create_directory(path, *args, **kwargs):
        if failure == "mkdir" and path == home.parent:
            raise PermissionError("read-only home")
        return mkdir(path, *args, **kwargs)

    def create(*args, **kwargs):
        if failure == "write" and kwargs.get("dir") == home.parent:
            raise PermissionError("read-only home")
        return mkstemp(*args, **kwargs)

    monkeypatch.setattr(install.tempfile, "mkstemp", create)
    monkeypatch.setattr(Path, "mkdir", create_directory)
    monkeypatch.setenv("PATH", "")
    from agentfield.exceptions import HarnessProviderUnavailable

    for _ in range(2):
        with pytest.raises(HarnessProviderUnavailable, match="read-only home") as error:
            await CodeafProvider().execute("p", {})
        assert "cannot install CodeAF into" in str(error.value)
        assert "af codeaf ensure" in str(error.value)
    assert not home.exists()
    assert not (home.parents[2] / "temp").exists()
    assert install.find_managed_binary() is None
    assert local_release[1] == []


@pytest.mark.parametrize("key", ["AGENTFIELD_SKIP_CODEAF", "AGENTFIELD_SKIP_AFORGE"])
@pytest.mark.parametrize("value", ["1", "TRUE", "Yes", "oN"])
def test_skip_switches_never_download(local_release, monkeypatch, key, value):
    _, requests = local_release
    monkeypatch.setenv(key, value)
    assert install.install_codeaf() is None
    assert not requests
    assert not install.managed_paths()[0].parent.exists()


@pytest.mark.parametrize(
    "system,machine", [("FreeBSD", "x86_64"), ("Linux", "riscv64")]
)
def test_unsupported_platform_is_not_installable(
    local_release, monkeypatch, system, machine
):
    _, requests = local_release
    monkeypatch.setattr(install.platform, "system", lambda: system)
    monkeypatch.setattr(install.platform, "machine", lambda: machine)
    assert install.install_codeaf() is None
    assert not requests


async def test_concurrent_first_calls_download_once(local_release, caplog):
    _, requests = local_release
    caplog.set_level(logging.INFO)
    results = await asyncio.gather(
        *(CodeafProvider().execute("p", {}) for _ in range(16))
    )
    assert all(result.result == "downloaded" for result in results)
    assert requests == ["/v0.8.2/codeaf-linux-amd64"]
    assert caplog.text.count("downloading CodeAF v0.8.2 (~70 MB) to") == 1


def test_managed_lookup_requires_executable_and_matching_marker(local_release):
    (home,) = install.managed_paths()
    for path in (home,):
        path.parent.mkdir(parents=True)
        path.write_text("binary")
        path.chmod(0o755)
        (path.parent / ".codeaf.version").write_text("v0.1.0")
    assert install.find_managed_binary() is None
    (home.parent / ".codeaf.version").write_text("v0.8.2\n")
    home.chmod(0o644)
    assert not install.is_installed(home)
    home.chmod(0o755)
    assert install.find_managed_binary() == str(home)
    home.unlink()
    home.mkdir()
    assert not install.is_installed(home)


def test_corrupt_or_missing_marker_is_not_installed(local_release):
    target = install.managed_paths()[0]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"binary")
    target.chmod(0o755)
    assert not install.is_installed(target)
    (target.parent / ".codeaf.version").write_bytes(b"\xff\xfe")
    assert not install.is_installed(target)


@pytest.mark.parametrize("matching", [True, False])
def test_windows_existing_binary_replace_failure(local_release, monkeypatch, matching):
    monkeypatch.setattr(install.platform, "system", lambda: "Windows")
    target = install.managed_paths()[0]

    def in_use(src, dst):
        # Simulate a second installer completing while we downloaded.
        target.write_bytes(b"existing")
        target.chmod(0o755)
        if matching:
            (target.parent / ".codeaf.version").write_text("v0.8.2\n")
        raise PermissionError("binary in use")

    monkeypatch.setattr(install.os, "replace", in_use)
    if matching:
        assert install.install_codeaf() == str(target)
        assert target.read_bytes() == b"existing"
    else:
        with pytest.raises(PermissionError, match="in use"):
            install.install_codeaf()
    assert not list(target.parent.glob(".codeaf-*"))
