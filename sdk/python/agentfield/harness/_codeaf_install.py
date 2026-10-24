"""Pinned, isolated CodeAF installation using only the standard library."""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import shutil
import tempfile
import urllib.request
from collections.abc import Mapping
from pathlib import Path

CODEAF_VERSION = "v0.8.2"
CODEAF_BASE_URL = (
    f"https://github.com/Agent-Field/CodeAF/releases/download/{CODEAF_VERSION}"
)
CODEAF_SHA256 = {
    "codeaf-darwin-amd64": "a476f1ce44b21f8016583a65c49b21497d10713bcd72c33478b3bd12e52446d7",
    "codeaf-darwin-arm64": "e61adac060bf2345c08a3e23ae83239871c3050eadac046879f5f3176270c4f1",
    "codeaf-linux-amd64": "681d9813f2517addc86fb644d313e3f7cf0ab60be708c8e8b13d77fb16a6798c",
    "codeaf-linux-arm64": "c6375d58d198d9e1a84880d8b4324ee6c067a0e5e351af115ca5506b1ffa1c78",
    "codeaf-windows-amd64.exe": "ff80f6f0c89c209fc809e960c70a1eb3518ccfc06775d587d5a187edd5d5222d",
    "codeaf-windows-arm64.exe": "bfc0b386e5248e7e6756f097b372f72e4e955ee1e19e9216ccafee47064b2b80",
}
MAX_DOWNLOAD_BYTES = 128 * 1024 * 1024
logger = logging.getLogger("agentfield.harness.codeaf")


def truthy(value: str | None) -> bool:
    return bool(value and value.strip().lower() in {"1", "true", "yes", "on"})


def agentfield_home(env: Mapping[str, str] | None = None) -> Path:
    environment = os.environ if env is None else env
    return Path(environment.get("AGENTFIELD_HOME") or "~/.agentfield").expanduser()


def binary_name() -> str:
    return "codeaf.exe" if platform.system() == "Windows" else "codeaf"


def managed_paths(env: Mapping[str, str] | None = None) -> tuple[Path]:
    return (agentfield_home(env) / "bin" / binary_name(),)


def is_installed(binary: Path) -> bool:
    try:
        return (
            binary.is_file()
            and os.access(binary, os.X_OK)
            and (binary.parent / ".codeaf.version").read_text().strip()
            == CODEAF_VERSION
        )
    except (OSError, UnicodeError):
        return False


def find_managed_binary(env: Mapping[str, str] | None = None) -> str | None:
    (home,) = managed_paths(env)
    return str(home) if is_installed(home) else None


def find_codeaf_binary(env: Mapping[str, str] | None = None) -> str | None:
    """Lookup without downloading, with explicit overrides before the pin."""
    environment = os.environ if env is None else env
    override = environment.get("CODEAF_BIN")
    path = environment.get("PATH")
    if override:
        return shutil.which(override, path=path)
    return find_managed_binary(environment) or shutil.which("codeaf", path=path)


def default_state_directory(env: Mapping[str, str]) -> str:
    """Create isolated state best-effort; let CodeAF report write failures."""
    home = agentfield_home(env) / "codeaf"
    try:
        home.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return str(home)


def release_asset() -> str | None:
    system = platform.system().lower()
    machine = platform.machine().lower()
    arch = {
        "x86_64": "amd64",
        "amd64": "amd64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }.get(machine)
    suffix = ".exe" if system == "windows" else ""
    asset = f"codeaf-{system}-{arch}{suffix}"
    return asset if asset in CODEAF_SHA256 else None


def install_codeaf() -> str | None:
    """Download and verify the pin in AgentField's bin directory."""
    if any(
        truthy(os.environ.get(key))
        for key in ("AGENTFIELD_SKIP_CODEAF", "AGENTFIELD_SKIP_AFORGE")
    ):
        return None
    installed = find_managed_binary()
    if installed:
        return installed
    asset = release_asset()
    if asset is None:
        return None

    (target,) = managed_paths()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".codeaf-", dir=target.parent)
    except OSError as exc:
        raise OSError(f"cannot install CodeAF into {target.parent}: {exc}") from exc

    logger.info("downloading CodeAF %s (~70 MB) to %s", CODEAF_VERSION, target)
    base = os.environ.get("AGENTFIELD_CODEAF_BASE_URL", CODEAF_BASE_URL).rstrip("/")
    digest = hashlib.sha256()
    try:
        with os.fdopen(fd, "wb") as output:
            # Refresh proxy configuration for each attempt rather than retaining
            # urllib's process-global opener from an earlier environment.
            opener = urllib.request.build_opener()
            with opener.open(f"{base}/{asset}", timeout=60) as response:
                total = 0
                while chunk := response.read(
                    min(1024 * 1024, MAX_DOWNLOAD_BYTES - total + 1)
                ):
                    total += len(chunk)
                    if total > MAX_DOWNLOAD_BYTES:
                        raise ValueError("CodeAF download exceeds 128 MiB size limit")
                    digest.update(chunk)
                    output.write(chunk)
        if digest.hexdigest() != CODEAF_SHA256[asset]:
            raise ValueError(f"CodeAF SHA-256 mismatch for {asset}")
        os.chmod(temporary, 0o755)
        try:
            os.replace(temporary, target)
        except OSError:
            if platform.system() != "Windows" or not is_installed(target):
                raise
        # Only a verified, executable binary may receive the version marker.
        (target.parent / ".codeaf.version").write_text(f"{CODEAF_VERSION}\n")
        return str(target)
    finally:
        Path(temporary).unlink(missing_ok=True)
