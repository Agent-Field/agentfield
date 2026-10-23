"""A5: temporary binaries and state must use private, trusted directories."""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import os
import stat
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agentfield.harness import harness_doctor
from agentfield.harness import _codeaf_install as install
from agentfield.harness.providers import codeaf


@pytest.fixture(autouse=True)
def isolated_temp(monkeypatch, tmp_path, caplog):
    for key in ("AGENTFIELD_SKIP_CODEAF", "AGENTFIELD_SKIP_AFORGE", "CODEAF_BIN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AGENTFIELD_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(install.tempfile, "gettempdir", lambda: str(tmp_path / "temp"))
    monkeypatch.setattr(install.platform, "system", lambda: "Linux")
    monkeypatch.setattr(install.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(install.logger, "handlers", [caplog.handler])
    monkeypatch.setattr(install.logger, "propagate", False)
    caplog.set_level(logging.WARNING, logger="agentfield.harness.codeaf")
    body = b"#!/bin/sh\nexit 0\n"
    download = Mock(side_effect=lambda *args, **kwargs: io.BytesIO(body))
    monkeypatch.setattr(
        install.urllib.request,
        "build_opener",
        lambda: SimpleNamespace(open=download),
    )
    for asset in install.CODEAF_SHA256:
        monkeypatch.setitem(
            install.CODEAF_SHA256, asset, hashlib.sha256(body).hexdigest()
        )
    return download


def temp_base() -> Path:
    name = "agentfield-codeaf"
    if install.platform.system() != "Windows":
        name += f"-{os.getuid()}"
    return Path(install.tempfile.gettempdir()) / name


def plant_binary(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    (path.parent / ".codeaf.version").write_text("v0.8.2\n")


def select_fallback(operation: str) -> str | None:
    if operation == "lookup":
        return install.find_managed_binary()
    if operation == "install":
        return install.install_codeaf()
    if operation == "state":
        return install.default_state_directory(os.environ)

    async def probe(command):
        return "codeaf v0.8.2 built fixture"

    [report] = asyncio.run(harness_doctor(["codeaf"], version_probe=probe))
    return report.binary


@pytest.mark.parametrize("system", ["Linux", "Darwin", "Windows"])
def test_temp_paths_are_per_user(monkeypatch, system):
    monkeypatch.setattr(install.platform, "system", lambda: system)
    install.agentfield_home().write_text("unwritable home")
    name = "codeaf.exe" if system == "Windows" else "codeaf"
    assert install.managed_paths()[1] == temp_base() / "v0.8.2" / name
    assert select_fallback("state") == str(temp_base() / "state")


@pytest.mark.parametrize("operation", ["lookup", "install", "state", "doctor"])
def test_existing_public_directories_are_private_before_use(monkeypatch, operation):
    base = temp_base()
    directory = base / ("state" if operation == "state" else "v0.8.2")
    directory.mkdir(parents=True)
    if operation in {"lookup", "doctor"}:
        plant_binary(directory / "codeaf")
    if operation in {"install", "state"}:
        install.agentfield_home().write_text("unwritable home")
    for path in (base, directory):
        path.chmod(0o777)
    read_text = Path.read_text

    def checked_read(path, *args, **kwargs):
        if path.parent == directory:
            assert stat.S_IMODE(base.lstat().st_mode) == 0o700
            assert stat.S_IMODE(directory.lstat().st_mode) == 0o700
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", checked_read)
    expected = directory if operation == "state" else directory / "codeaf"
    assert select_fallback(operation) == str(expected)
    assert stat.S_IMODE(base.lstat().st_mode) == 0o700
    assert stat.S_IMODE(directory.lstat().st_mode) == 0o700


@pytest.mark.parametrize("operation", ["lookup", "install", "state", "doctor"])
@pytest.mark.parametrize("level", ["base", "child"])
@pytest.mark.parametrize(
    "defect", ["symlink", "foreign-owner", "file", "lstat", "chmod"]
)
def test_untrusted_temp_directories_are_skipped(
    isolated_temp, monkeypatch, tmp_path, caplog, operation, level, defect
):
    base = temp_base()
    directory = base / ("state" if operation == "state" else "v0.8.2")
    rejected = base if level == "base" else directory
    rejected.parent.mkdir(parents=True, exist_ok=True)
    external = tmp_path / "external"
    external.mkdir(mode=0o755)
    if defect == "symlink":
        rejected.symlink_to(external, target_is_directory=True)
    elif defect == "file":
        rejected.write_text("not a directory")
    else:
        rejected.mkdir(mode=0o777)
    if defect != "file":
        directory.mkdir(parents=True, exist_ok=True)
        if operation in {"lookup", "doctor"}:
            plant_binary(directory / "codeaf")
    if operation in {"install", "state"}:
        install.agentfield_home().write_text("unwritable home")
    lstat, chmod = Path.lstat, Path.chmod
    untouched_mode = rejected.stat().st_mode

    def checked_lstat(path, *args, **kwargs):
        if path == rejected:
            if defect == "lstat":
                raise PermissionError("cannot inspect temp directory")
            if defect == "foreign-owner":
                info = lstat(path, *args, **kwargs)
                return SimpleNamespace(st_mode=info.st_mode, st_uid=os.getuid() + 1)
        return lstat(path, *args, **kwargs)

    def checked_chmod(path, *args, **kwargs):
        if path == rejected:
            if defect == "chmod":
                raise PermissionError("cannot secure temp directory")
            if defect in {"symlink", "foreign-owner", "file", "lstat"}:
                pytest.fail("must reject an untrusted directory before chmod")
        return chmod(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", checked_lstat)
    monkeypatch.setattr(Path, "chmod", checked_chmod)
    if operation == "state":
        with pytest.raises(OSError):
            select_fallback(operation)
    else:
        assert select_fallback(operation) is None
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1
    isolated_temp.assert_not_called()
    assert rejected.stat().st_mode == untouched_mode


@pytest.mark.parametrize("operation", ["lookup", "install", "state", "doctor"])
def test_old_shared_binary_and_marker_are_never_used(
    isolated_temp, monkeypatch, tmp_path, operation
):
    old = Path(install.tempfile.gettempdir()) / "agentfield-codeaf/v0.8.2/codeaf"
    plant_binary(old)
    if operation in {"install", "state"}:
        install.agentfield_home().write_text("unwritable home")
        expected = temp_base() / ("state" if operation == "state" else "v0.8.2/codeaf")
        assert select_fallback(operation) == str(expected)
        assert isolated_temp.call_count == (operation == "install")
        assert old.read_bytes() == b"#!/bin/sh\nexit 0\n"
        return
    assert select_fallback(operation) is None
    path_binary = tmp_path / "path/codeaf"
    plant_binary(path_binary)
    monkeypatch.setenv("PATH", str(path_binary.parent))
    assert select_fallback(operation) == (
        None if operation == "lookup" else str(path_binary)
    )


def test_rejected_temp_fallback_continues_to_path(monkeypatch, tmp_path, caplog):
    install.agentfield_home().write_text("unwritable home")
    base = temp_base()
    base.parent.mkdir(parents=True)
    base.symlink_to(tmp_path, target_is_directory=True)
    path_binary = tmp_path / "path/codeaf"
    plant_binary(path_binary)
    monkeypatch.setenv("PATH", str(path_binary.parent))
    monkeypatch.setattr(codeaf, "_download_retry_at", 0.0)
    monkeypatch.setattr(codeaf, "_last_download_error", None)
    assert codeaf._resolve_binary()[:2] == (str(path_binary), "codeaf")
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_repaired_temp_directory_is_rechecked(monkeypatch, tmp_path):
    base = temp_base()
    base.parent.mkdir(parents=True)
    base.symlink_to(tmp_path, target_is_directory=True)
    assert install.find_managed_binary() is None
    base.unlink()
    binary = base / "v0.8.2/codeaf"
    plant_binary(binary)
    assert install.find_managed_binary() == str(binary)


def test_windows_rejects_reparse_points(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(install.platform, "system", lambda: "Windows")
    base = temp_base()
    plant_binary(base / "v0.8.2/codeaf.exe")
    lstat = Path.lstat

    def reparse(path, *args, **kwargs):
        info = lstat(path, *args, **kwargs)
        if path == base:
            return SimpleNamespace(
                st_mode=info.st_mode,
                st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT,
            )
        return info

    monkeypatch.setattr(Path, "lstat", reparse)
    assert install.find_managed_binary() is None
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_windows_does_not_require_unix_ownership_or_permissions(monkeypatch):
    monkeypatch.setattr(install.platform, "system", lambda: "Windows")
    base = temp_base()
    binary = base / "v0.8.2/codeaf.exe"
    plant_binary(binary)
    monkeypatch.setattr(
        install.os, "getuid", Mock(side_effect=AssertionError("Unix only"))
    )
    monkeypatch.setattr(Path, "chmod", Mock(side_effect=AssertionError("Unix only")))
    assert install.find_managed_binary() == str(binary)
