"""Workspace artifacts: transport a caller's folder into a reasoner execution.

A caller may attach a local folder to any reasoner execution. The folder is
sealed into a content-addressed store (CAS), transported to the node as a
manifest plus blobs, materialized into a per-execution directory, and the
reasoner runs *inside* that directory. File changes flow back to the caller as
a staged diff. Reasoner authors write no folder-handling code — they use
relative paths (cwd is the workspace) or read ``AGENTFIELD_WORKSPACE`` for the
absolute path.

This module implements the node-side pieces:

* A content store at ``~/.agentfield/cas`` — whole-file blobs keyed by their
  sha256 hex digest.
* Manifest hashing and diffing (the frozen contract in
  ``docs/design/workspace-artifacts.md``).
* Materialization of a manifest into a workspace directory.
* Running a reasoner in a dedicated worker process whose cwd is the workspace,
  with a fresh asyncio event loop, so the parent server stays responsive and
  ``os.chdir`` never races across concurrent requests.

The HTTP endpoints (prepare / put-blob / get-blob) and the execution hook live
in ``agent.py`` and call into the helpers here.
"""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import json
import logging
import os
import shutil
import stat
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Imported at module scope (not inside install_workspace_routes) so FastAPI's
# get_type_hints can resolve the string annotations produced by
# ``from __future__ import annotations`` when it inspects the route handlers.
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response

logger = logging.getLogger("agentfield.workspace")

# Manifest schema version. Bump only on a breaking manifest-format change.
MANIFEST_VERSION = 1

# Directories and files never included when sealing or hashing a workspace.
# Mirrors the defaults in docs/design/workspace-artifacts.md. ``.gitignore``
# rules found at the workspace root are honored on top of these.
DEFAULT_IGNORE_DIRS = frozenset(
    {
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        "target",
        "dist",
        "build",
    }
)
DEFAULT_IGNORE_FILES = frozenset({".DS_Store"})


# ---------------------------------------------------------------------------
# Filesystem locations
# ---------------------------------------------------------------------------


def agentfield_home() -> Path:
    """Root of the node's on-disk state.

    Honors ``AGENTFIELD_HOME`` so tests (and multi-tenant hosts) can redirect
    the CAS and workspace roots without touching a shared home directory.
    """
    override = os.environ.get("AGENTFIELD_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".agentfield"


def cas_dir() -> Path:
    """Directory of the node content-addressed store (blobs keyed by sha256)."""
    return agentfield_home() / "cas"


def workspaces_dir() -> Path:
    """Directory holding per-execution materialized workspaces."""
    return agentfield_home() / "workspaces"


# ---------------------------------------------------------------------------
# Content-addressed store
# ---------------------------------------------------------------------------


class ContentStore:
    """Whole-file blob store keyed by sha256 hex.

    Layout: ``<root>/<sha256-hex>``. Blobs are immutable; writes are atomic
    (temp file + rename) so a concurrent reader never sees a partial blob.
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root) if root is not None else cas_dir()

    def _path(self, sha256: str) -> Path:
        return self.root / sha256

    def has(self, sha256: str) -> bool:
        return self._path(sha256).is_file()

    def read(self, sha256: str) -> bytes:
        return self._path(sha256).read_bytes()

    def path_for(self, sha256: str) -> Path:
        """Absolute path of a stored blob (may not exist)."""
        return self._path(sha256)

    def put_bytes(self, data: bytes) -> str:
        """Store raw bytes, returning the sha256 hex key. Idempotent."""
        sha256 = hashlib.sha256(data).hexdigest()
        dest = self._path(sha256)
        if dest.is_file():
            return sha256
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{sha256}.{os.getpid()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, dest)
        return sha256

    def put_verified(self, sha256: str, data: bytes) -> None:
        """Store bytes whose expected sha256 is known, verifying the hash.

        Raises ``ValueError`` if the content does not hash to ``sha256`` — the
        PUT-blob endpoint relies on this to reject corrupt uploads.
        """
        actual = hashlib.sha256(data).hexdigest()
        if actual != sha256:
            raise ValueError(f"blob hash mismatch: expected {sha256}, got {actual}")
        self.put_bytes(data)


# ---------------------------------------------------------------------------
# .gitignore matching (minimal, root-level for the POC)
# ---------------------------------------------------------------------------


class _GitignoreMatcher:
    """Small subset of gitignore semantics sufficient for the POC.

    Supports: blank/comment lines, ``!`` negation, directory-only patterns
    (trailing ``/``), root-anchored patterns (leading or embedded ``/``), and
    glob patterns matched against the basename at any depth. Last matching
    rule wins, mirroring git.
    """

    def __init__(self, patterns: List[str]) -> None:
        # Each rule: (pattern, negate, dir_only, anchored)
        self._rules: List[Tuple[str, bool, bool, bool]] = []
        for raw in patterns:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            negate = line.startswith("!")
            if negate:
                line = line[1:]
            dir_only = line.endswith("/")
            if dir_only:
                line = line[:-1]
            # A leading "/" anchors to root; an embedded "/" also anchors.
            anchored = line.startswith("/") or ("/" in line.rstrip("/"))
            line = line.lstrip("/")
            self._rules.append((line, negate, dir_only, anchored))

    def match(self, rel_path: str, is_dir: bool) -> bool:
        """Return True if ``rel_path`` (posix, relative) is ignored."""
        ignored = False
        basename = rel_path.rsplit("/", 1)[-1]
        for pattern, negate, dir_only, anchored in self._rules:
            if dir_only and not is_dir:
                # Directory-only patterns can still ignore files *under* the
                # directory; that is handled by pruning during the walk. Here
                # we only match the directory entry itself.
                continue
            hit = False
            if anchored:
                hit = fnmatch.fnmatch(rel_path, pattern) or rel_path.startswith(
                    pattern + "/"
                )
            else:
                hit = (
                    fnmatch.fnmatch(basename, pattern)
                    or fnmatch.fnmatch(rel_path, pattern)
                    or fnmatch.fnmatch(rel_path, "*/" + pattern)
                )
            if hit:
                ignored = not negate
        return ignored


def _load_gitignore(root: Path) -> _GitignoreMatcher:
    gi = root / ".gitignore"
    if gi.is_file():
        try:
            return _GitignoreMatcher(gi.read_text().splitlines())
        except OSError:
            pass
    return _GitignoreMatcher([])


# ---------------------------------------------------------------------------
# Manifest: hashing and canonical id
# ---------------------------------------------------------------------------


def _iter_workspace_files(root: Path) -> List[Path]:
    """Yield regular files under ``root``, honoring default + gitignore rules."""
    matcher = _load_gitignore(root)
    out: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        # Prune ignored directories in place so os.walk doesn't descend.
        kept = []
        for name in dirnames:
            if name in DEFAULT_IGNORE_DIRS:
                continue
            rel = (d / name).relative_to(root).as_posix()
            if matcher.match(rel, is_dir=True):
                continue
            kept.append(name)
        dirnames[:] = kept
        for name in filenames:
            if name in DEFAULT_IGNORE_FILES:
                continue
            fpath = d / name
            if fpath.is_symlink() or not fpath.is_file():
                continue
            rel = fpath.relative_to(root).as_posix()
            if matcher.match(rel, is_dir=False):
                continue
            out.append(fpath)
    return out


def hash_file(path: Path) -> str:
    """sha256 hex of a file's contents (streamed)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(root: Path) -> Dict[str, Any]:
    """Seal a directory into a manifest dict.

    Files are sorted ascending by path. ``mode`` is the permission bits
    (e.g. 420 == 0o644). ``mtime_ns`` is the file modification time in
    nanoseconds. ``sha256`` keys the blob content.
    """
    root = Path(root)
    files: List[Dict[str, Any]] = []
    for fpath in _iter_workspace_files(root):
        st = fpath.stat()
        files.append(
            {
                "path": fpath.relative_to(root).as_posix(),
                "size": st.st_size,
                "mode": stat.S_IMODE(st.st_mode),
                "mtime_ns": st.st_mtime_ns,
                "sha256": hash_file(fpath),
            }
        )
    files.sort(key=lambda e: e["path"])
    return {"version": MANIFEST_VERSION, "files": files}


def canonical_manifest_json(manifest: Dict[str, Any]) -> bytes:
    """Canonical JSON: sorted keys, sorted files, no whitespace."""
    files = sorted(manifest.get("files", []), key=lambda e: e["path"])
    canonical = {"version": manifest.get("version", MANIFEST_VERSION), "files": files}
    return json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")


def manifest_id(manifest: Dict[str, Any]) -> str:
    """sha256 hex of the canonical manifest JSON."""
    return hashlib.sha256(canonical_manifest_json(manifest)).hexdigest()


def _files_by_path(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {e["path"]: e for e in manifest.get("files", [])}


def diff_manifests(
    input_manifest: Dict[str, Any], output_manifest: Dict[str, Any]
) -> Dict[str, Any]:
    """Compute the staged diff of ``output`` relative to ``input``.

    ``changed`` = files new in output or whose sha256 differs from the input
    (each entry carries ``path``, ``sha256``, ``size``, ``mode``).
    ``deleted`` = paths present in input but absent from output.
    Both lists are sorted by path for determinism.
    """
    before = _files_by_path(input_manifest)
    after = _files_by_path(output_manifest)

    changed: List[Dict[str, Any]] = []
    for path, entry in after.items():
        prior = before.get(path)
        if prior is None or prior.get("sha256") != entry.get("sha256"):
            changed.append(
                {
                    "path": path,
                    "sha256": entry["sha256"],
                    "size": entry["size"],
                    "mode": entry["mode"],
                }
            )
    deleted = [path for path in before if path not in after]

    changed.sort(key=lambda e: e["path"])
    deleted.sort()
    return {"changed": changed, "deleted": deleted}


# ---------------------------------------------------------------------------
# prepare / materialize
# ---------------------------------------------------------------------------


def missing_blobs(manifest: Dict[str, Any], store: ContentStore) -> List[str]:
    """sha256 hexes referenced by ``manifest`` that the store does not have.

    Deduplicated and sorted. Backs the ``prepare`` endpoint.
    """
    wanted = {e["sha256"] for e in manifest.get("files", [])}
    return sorted(sha for sha in wanted if not store.has(sha))


class MissingBlobsError(RuntimeError):
    """Raised when materialization cannot proceed: blobs absent from the CAS."""

    def __init__(self, missing: List[str]) -> None:
        self.missing = missing
        super().__init__(
            "cannot materialize workspace; missing "
            f"{len(missing)} blob(s) from the node content store: " + ", ".join(missing)
        )


def materialize(manifest: Dict[str, Any], dest: Path, store: ContentStore) -> Path:
    """Write ``manifest`` into ``dest``, copying blob content from the store.

    Restores each file's permission bits. Raises :class:`MissingBlobsError`
    (listing every absent blob) if any referenced blob is not in the CAS,
    before writing anything.
    """
    missing = missing_blobs(manifest, store)
    if missing:
        raise MissingBlobsError(missing)

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for entry in manifest.get("files", []):
        rel = entry["path"]
        # Defensive: reject path traversal in an untrusted manifest.
        target = (dest / rel).resolve()
        if not str(target).startswith(str(dest.resolve()) + os.sep) and target != dest:
            raise ValueError(f"unsafe manifest path escapes workspace: {rel!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(store.read(entry["sha256"]))
        mode = entry.get("mode")
        if isinstance(mode, int):
            os.chmod(target, mode)
    return dest


def store_diff_blobs(
    workspace: Path, diff: Dict[str, Any], store: ContentStore
) -> None:
    """Persist the content of every changed file in ``diff`` into the store."""
    workspace = Path(workspace)
    for entry in diff.get("changed", []):
        fpath = workspace / entry["path"]
        if fpath.is_file():
            store.put_bytes(fpath.read_bytes())


# ---------------------------------------------------------------------------
# Worker process: run the reasoner inside the workspace
# ---------------------------------------------------------------------------


def _worker_main(conn, workspace: str, func, args, kwargs) -> None:
    """Child-process entry point. Runs ``func`` with cwd == workspace.

    A fresh asyncio event loop is created here (never inherited). The result
    or a serializable error is sent back over ``conn``.
    """
    try:
        os.chdir(workspace)
        os.environ["AGENTFIELD_WORKSPACE"] = workspace

        if asyncio.iscoroutinefunction(func):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(func(*args, **kwargs))
            finally:
                try:
                    loop.close()
                finally:
                    asyncio.set_event_loop(None)
        else:
            result = func(*args, **kwargs)

        try:
            conn.send(("ok", result))
        except Exception:  # result not picklable — surface a clear error
            conn.send(
                (
                    "error",
                    "WorkspaceResultError",
                    "reasoner result is not serializable across the worker boundary",
                    "",
                )
            )
    except BaseException as exc:  # noqa: BLE001 — must ferry any failure back
        tb = traceback.format_exc()
        try:
            conn.send(("error", type(exc).__name__, str(exc), tb))
        except Exception:
            conn.send(("error", "WorkspaceWorkerError", "worker failed", tb))
    finally:
        conn.close()


class WorkspaceExecutionError(RuntimeError):
    """A reasoner raised (or the worker died) while running in a workspace.

    ``worker_traceback`` carries the child-process traceback text when known.
    """

    def __init__(self, message: str, worker_traceback: str = "") -> None:
        self.worker_traceback = worker_traceback
        super().__init__(message)


def _run_worker_blocking(
    workspace: Path, func: Callable, args: tuple, kwargs: dict
) -> Any:
    """Fork a worker, run the reasoner, return its result (blocking).

    Uses the ``fork`` start method where available (so the target/args and the
    reasoner closure are inherited rather than pickled); falls back to the
    platform default otherwise. Meant to be called from a thread executor so
    the parent event loop is never blocked.
    """
    import multiprocessing as mp

    try:
        ctx = mp.get_context("fork")
    except ValueError:  # pragma: no cover - platform without fork
        ctx = mp.get_context()

    recv_conn, send_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_worker_main,
        args=(send_conn, str(workspace), func, args, kwargs),
        daemon=False,
    )
    proc.start()
    # Parent does not write; close its copy so a child crash yields EOF.
    send_conn.close()

    payload = None
    got = False
    try:
        # Drain the pipe first (results can exceed the OS buffer), then join.
        if recv_conn.poll(None):
            payload = recv_conn.recv()
            got = True
    except EOFError:
        got = False
    finally:
        recv_conn.close()
        proc.join()

    if not got:
        raise WorkspaceExecutionError(
            "workspace worker exited without returning a result "
            f"(exit code {proc.exitcode})"
        )

    status = payload[0]
    if status == "ok":
        return payload[1]

    _, exc_type, exc_msg, tb = payload
    raise WorkspaceExecutionError(f"{exc_type}: {exc_msg}", worker_traceback=tb)


async def run_reasoner_in_workspace(
    *,
    func: Callable,
    args: tuple,
    kwargs: dict,
    manifest: Dict[str, Any],
    execution_id: str,
    store: Optional[ContentStore] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Materialize ``manifest``, run ``func`` inside it, and stage the diff.

    Returns ``(result, workspace_diff)``. The reasoner runs in a dedicated
    worker process (cwd == the workspace) so ``os.chdir`` never races with
    other requests and a crash cannot take down the server. New blobs are
    stored in the node CAS; the workspace directory is removed on success and
    kept (with its path logged) on failure for inspection.
    """
    store = store or ContentStore()
    loop = asyncio.get_running_loop()
    workspace = workspaces_dir() / execution_id

    # 1. Materialize (blocking file I/O -> executor).
    await loop.run_in_executor(None, materialize, manifest, workspace, store)

    keep_on_error = False
    try:
        # 2. Run the reasoner in a worker process, off the event loop.
        result = await loop.run_in_executor(
            None, _run_worker_blocking, workspace, func, args, kwargs
        )
    except BaseException:
        keep_on_error = True
        logger.error(
            "workspace execution failed; preserving directory for inspection: %s",
            workspace,
        )
        raise

    # 3. Hash the workspace, diff against the input, persist new blobs.
    output_manifest = await loop.run_in_executor(None, build_manifest, workspace)
    diff = diff_manifests(manifest, output_manifest)
    await loop.run_in_executor(None, store_diff_blobs, workspace, diff, store)

    # 4. Clean up (only on success).
    if not keep_on_error:
        await loop.run_in_executor(
            None, lambda: shutil.rmtree(workspace, ignore_errors=True)
        )

    return result, diff


# ---------------------------------------------------------------------------
# Auto-registered node endpoints
# ---------------------------------------------------------------------------


def install_workspace_routes(agent: Any) -> None:
    """Register the three workspace endpoints on the agent's FastAPI app.

    * ``POST /api/v1/workspace/prepare`` — body ``{"manifest": <manifest>}``
      returns ``{"missing": ["<sha256>", ...]}``.
    * ``PUT  /api/v1/workspace/blobs/{sha256}`` — raw bytes, 204 on success,
      verifies the uploaded content hashes to ``{sha256}``.
    * ``GET  /api/v1/workspace/blobs/{sha256}`` — raw bytes, 404 if absent.

    All three are control-plane-initiated (they work when the control plane
    is behind NAT and the node is remote). The target gets one shared
    :class:`ContentStore` bound to ``~/.agentfield/cas``.

    ``agent`` only needs FastAPI's ``.post``/``.put``/``.get`` decorators, so
    this also mounts the routes on a standalone FastAPI app — see
    :func:`attach_workspace_routes`, the name serverless wrappers use.
    """
    store = getattr(agent, "_workspace_store", None)
    if store is None:
        store = ContentStore()
        agent._workspace_store = store

    @agent.post("/api/v1/workspace/prepare")
    async def _workspace_prepare(request: Request):
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="invalid JSON body")
        if not isinstance(body, dict) or not isinstance(body.get("manifest"), dict):
            raise HTTPException(
                status_code=400, detail='expected {"manifest": <manifest>}'
            )
        missing = missing_blobs(body["manifest"], store)
        return JSONResponse(status_code=200, content={"missing": missing})

    @agent.put("/api/v1/workspace/blobs/{sha256}")
    async def _workspace_put_blob(sha256: str, request: Request):
        if not _is_sha256_hex(sha256):
            raise HTTPException(status_code=400, detail="invalid sha256")
        data = await request.body()
        try:
            store.put_verified(sha256, data)
        except ValueError:
            raise HTTPException(status_code=400, detail="blob hash mismatch")
        return Response(status_code=204)

    @agent.get("/api/v1/workspace/blobs/{sha256}")
    async def _workspace_get_blob(sha256: str):
        if not _is_sha256_hex(sha256):
            raise HTTPException(status_code=400, detail="invalid sha256")
        if not store.has(sha256):
            raise HTTPException(status_code=404, detail="blob not found")
        return Response(
            content=store.read(sha256),
            status_code=200,
            media_type="application/octet-stream",
        )


def attach_workspace_routes(api: Any) -> None:
    """Mount the three workspace endpoints on an external FastAPI app.

    Public helper for serverless wrappers (nodes the control plane reaches
    purely by URL — e.g. a PaaS deployment — that can never dial back). The
    wrapper serves ``/discover`` + ``/execute`` itself; call this so the same
    server also serves the workspace transport the control plane needs:

        POST /api/v1/workspace/prepare
        PUT  /api/v1/workspace/blobs/{sha256}
        GET  /api/v1/workspace/blobs/{sha256}

    The routes share the node's content store at ``~/.agentfield/cas``, so blobs
    the control plane uploads here are exactly the ones a workspace-bearing
    ``/execute`` materializes from — no wrapper-side CAS code required.
    """
    install_workspace_routes(api)


def _is_sha256_hex(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def extract_workspace_manifest(artifacts: Any) -> Optional[Dict[str, Any]]:
    """Pull the inline workspace manifest out of an ``artifacts`` envelope.

    Returns the manifest dict, or ``None`` when no workspace is attached.
    Tolerant of missing/oddly-shaped envelopes so the normal (non-workspace)
    path is never disturbed.
    """
    if not isinstance(artifacts, dict):
        return None
    workspace = artifacts.get("workspace")
    if not isinstance(workspace, dict):
        return None
    manifest = workspace.get("manifest")
    if isinstance(manifest, dict) and isinstance(manifest.get("files"), list):
        return manifest
    return None
