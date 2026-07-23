"""Tests for workspace artifacts: CAS, manifest hash/diff, node endpoints,
and the in-worker execution hook.

Unit tests cover the pure pieces (content store, manifest hashing, canonical
id determinism, diffing, materialization). Integration tests (marked
``integration``) exercise the real fork-based worker: a reasoner runs inside a
materialized folder and the staged ``workspace_diff`` comes back on the result.
A concurrency test asserts two simultaneous workspace executions never see each
other's files.
"""

from __future__ import annotations

import copy
import hashlib
import threading

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from agentfield.workspace import (
    ContentStore,
    MissingBlobsError,
    build_manifest,
    diff_manifests,
    install_workspace_routes,
    manifest_id,
    materialize,
    missing_blobs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def af_home(tmp_path, monkeypatch):
    """Redirect the node CAS/workspace roots into a temp dir per test."""
    home = tmp_path / "afhome"
    monkeypatch.setenv("AGENTFIELD_HOME", str(home))
    return home


def _seal(src_dir, store):
    """Build a manifest of ``src_dir`` and stash its blobs in ``store``."""
    manifest = build_manifest(src_dir)
    for entry in manifest["files"]:
        store.put_bytes((src_dir / entry["path"]).read_bytes())
    return manifest


# ---------------------------------------------------------------------------
# Content store
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_content_store_put_get_roundtrip(af_home):
    store = ContentStore()
    data = b"hello world"
    sha = store.put_bytes(data)
    assert sha == hashlib.sha256(data).hexdigest()
    assert store.has(sha)
    assert store.read(sha) == data
    # idempotent
    assert store.put_bytes(data) == sha


@pytest.mark.unit
def test_content_store_put_verified_rejects_mismatch(af_home):
    store = ContentStore()
    with pytest.raises(ValueError):
        store.put_verified("0" * 64, b"not that content")
    assert not store.has("0" * 64)


# ---------------------------------------------------------------------------
# Manifest: hashing, ignores, canonical id
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_manifest_respects_default_and_gitignore(tmp_path):
    root = tmp_path / "proj"
    (root / "sub").mkdir(parents=True)
    (root / "__pycache__").mkdir()
    (root / "a.txt").write_text("hello")
    (root / "sub" / "b.py").write_text("x = 1")
    (root / "__pycache__" / "junk.pyc").write_text("nope")
    (root / ".gitignore").write_text("ignored.log\nsecrets/\n")
    (root / "ignored.log").write_text("secret")
    (root / "secrets").mkdir()
    (root / "secrets" / "key.pem").write_text("private")

    manifest = build_manifest(root)
    paths = [e["path"] for e in manifest["files"]]

    assert paths == [".gitignore", "a.txt", "sub/b.py"]  # sorted, filtered
    assert "__pycache__/junk.pyc" not in paths
    assert "ignored.log" not in paths
    assert "secrets/key.pem" not in paths
    # permission bits stored as int (0o644 == 420)
    a = next(e for e in manifest["files"] if e["path"] == "a.txt")
    assert a["size"] == 5
    assert isinstance(a["mode"], int)


@pytest.mark.unit
def test_manifest_id_is_order_independent(tmp_path):
    root = tmp_path / "proj"
    (root / "d").mkdir(parents=True)
    (root / "z.txt").write_text("z")
    (root / "a.txt").write_text("a")
    (root / "d" / "m.txt").write_text("m")

    manifest = build_manifest(root)
    shuffled = copy.deepcopy(manifest)
    shuffled["files"].reverse()

    assert manifest_id(manifest) == manifest_id(shuffled)
    # stable digest length
    assert len(manifest_id(manifest)) == 64


@pytest.mark.unit
def test_manifest_id_changes_with_content(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.txt").write_text("hello")
    m1 = build_manifest(root)
    (root / "a.txt").write_text("hello world")
    m2 = build_manifest(root)
    assert manifest_id(m1) != manifest_id(m2)


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_diff_reports_new_modified_and_deleted(tmp_path):
    root = tmp_path / "proj"
    (root / "keep.txt").parent.mkdir(parents=True, exist_ok=True)
    (root / "keep.txt").write_text("same")
    (root / "mod.txt").write_text("before")
    (root / "gone.txt").write_text("bye")
    before = build_manifest(root)

    (root / "mod.txt").write_text("after")  # modified
    (root / "gone.txt").unlink()  # deleted
    (root / "new.txt").write_text("fresh")  # new
    after = build_manifest(root)

    diff = diff_manifests(before, after)
    changed_paths = [e["path"] for e in diff["changed"]]

    assert changed_paths == ["mod.txt", "new.txt"]  # sorted, unchanged excluded
    assert diff["deleted"] == ["gone.txt"]
    # changed entries carry the contract fields
    for e in diff["changed"]:
        assert set(e.keys()) == {"path", "sha256", "size", "mode"}


@pytest.mark.unit
def test_diff_empty_when_unchanged(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.txt").write_text("x")
    m = build_manifest(root)
    diff = diff_manifests(m, build_manifest(root))
    assert diff == {"changed": [], "deleted": []}


# ---------------------------------------------------------------------------
# prepare / materialize
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_missing_blobs_and_materialize(af_home, tmp_path):
    src = tmp_path / "src"
    (src / "pkg").mkdir(parents=True)
    (src / "a.txt").write_text("alpha")
    (src / "pkg" / "b.txt").write_text("beta")
    manifest = build_manifest(src)

    store = ContentStore()
    # Before upload, every blob is missing.
    assert set(missing_blobs(manifest, store)) == {
        e["sha256"] for e in manifest["files"]
    }

    # Materialize without blobs -> clear error listing them.
    dest = tmp_path / "ws"
    with pytest.raises(MissingBlobsError) as exc:
        materialize(manifest, dest, store)
    assert len(exc.value.missing) == 2

    # Seal, then materialize succeeds and restores content + mode.
    for entry in manifest["files"]:
        store.put_bytes((src / entry["path"]).read_bytes())
    assert missing_blobs(manifest, store) == []
    materialize(manifest, dest, store)
    assert (dest / "a.txt").read_text() == "alpha"
    assert (dest / "pkg" / "b.txt").read_text() == "beta"


# ---------------------------------------------------------------------------
# Node endpoints
# ---------------------------------------------------------------------------


@pytest.fixture
def endpoint_client(af_home):
    app = FastAPI()
    install_workspace_routes(app)
    return TestClient(app)


@pytest.mark.unit
def test_prepare_reports_missing_then_none_after_upload(endpoint_client, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("alpha")
    manifest = build_manifest(src)
    sha = manifest["files"][0]["sha256"]

    r = endpoint_client.post("/api/v1/workspace/prepare", json={"manifest": manifest})
    assert r.status_code == 200
    assert r.json()["missing"] == [sha]

    up = endpoint_client.put(f"/api/v1/workspace/blobs/{sha}", content=b"alpha")
    assert up.status_code == 204

    r2 = endpoint_client.post("/api/v1/workspace/prepare", json={"manifest": manifest})
    assert r2.json()["missing"] == []


@pytest.mark.unit
def test_put_blob_rejects_hash_mismatch(endpoint_client):
    sha = hashlib.sha256(b"real").hexdigest()
    bad = endpoint_client.put(f"/api/v1/workspace/blobs/{sha}", content=b"tampered")
    assert bad.status_code == 400


@pytest.mark.unit
def test_get_blob_roundtrip_and_404(endpoint_client):
    data = b"payload-bytes"
    sha = hashlib.sha256(data).hexdigest()
    assert endpoint_client.get(f"/api/v1/workspace/blobs/{sha}").status_code == 404
    endpoint_client.put(f"/api/v1/workspace/blobs/{sha}", content=data)
    got = endpoint_client.get(f"/api/v1/workspace/blobs/{sha}")
    assert got.status_code == 200
    assert got.content == data


@pytest.mark.unit
def test_prepare_bad_body(endpoint_client):
    assert (
        endpoint_client.post("/api/v1/workspace/prepare", json={"nope": 1}).status_code
        == 400
    )
    assert endpoint_client.get("/api/v1/workspace/blobs/zzz").status_code == 400


# ---------------------------------------------------------------------------
# Integration: real reasoner running inside a materialized workspace
# ---------------------------------------------------------------------------


def _build_agent(node_id):
    from agentfield import Agent

    return Agent(node_id=node_id)


@pytest.mark.integration
def test_reasoner_runs_in_workspace_and_returns_diff(af_home, tmp_path):
    agent = _build_agent("ws-agent")

    @agent.reasoner()
    def transform() -> dict:
        # Runs with cwd == the materialized workspace; use relative paths.
        import os as _os

        data = open("input.txt").read()
        open("out.txt", "w").write(data.upper())  # new file
        open("existing.txt", "w").write("MODIFIED")  # modify existing
        _os.remove("stale.txt")  # delete existing
        return {"read": data, "cwd_env": _os.environ.get("AGENTFIELD_WORKSPACE")}

    # Seal a source folder into the node CAS.
    src = tmp_path / "src"
    src.mkdir()
    (src / "input.txt").write_text("hello")
    (src / "existing.txt").write_text("ORIGINAL")
    (src / "stale.txt").write_text("obsolete")
    store = ContentStore()
    manifest = _seal(src, store)

    client = TestClient(agent)
    resp = client.post(
        "/reasoners/transform",
        json={"artifacts": {"workspace": {"manifest": manifest}}},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Reasoner ran inside the folder and read the caller's file.
    assert body["read"] == "hello"
    assert body["cwd_env"] and "workspaces" in body["cwd_env"]

    diff = body["workspace_diff"]
    changed = {e["path"] for e in diff["changed"]}
    assert changed == {"out.txt", "existing.txt"}
    assert diff["deleted"] == ["stale.txt"]

    # New blobs landed in the node CAS.
    out_entry = next(e for e in diff["changed"] if e["path"] == "out.txt")
    assert store.read(out_entry["sha256"]) == b"HELLO"


@pytest.mark.integration
def test_non_workspace_execution_is_unchanged(af_home):
    agent = _build_agent("plain-agent")

    @agent.reasoner()
    def echo(msg: str) -> dict:
        return {"echo": msg}

    client = TestClient(agent)
    resp = client.post("/reasoners/echo", json={"msg": "hi"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["echo"] == "hi"
    assert "workspace_diff" not in body  # no diff attached on the normal path


@pytest.mark.integration
def test_concurrent_workspace_executions_are_isolated(af_home, tmp_path):
    agent = _build_agent("iso-agent")

    @agent.reasoner()
    def marker() -> dict:
        import os as _os
        import time

        seen = open("marker.txt").read()
        time.sleep(0.2)  # widen the overlap window
        open("wrote.txt", "w").write(seen)
        # List everything visible in this workspace to prove isolation.
        visible = sorted(_os.listdir("."))
        return {
            "seen": seen,
            "workspace": _os.environ.get("AGENTFIELD_WORKSPACE"),
            "visible": visible,
        }

    store = ContentStore()

    def seal_marker(value):
        d = tmp_path / f"src-{value}"
        d.mkdir()
        (d / "marker.txt").write_text(value)
        return _seal(d, store)

    m_a = seal_marker("AAA")
    m_b = seal_marker("BBB")

    client = TestClient(agent)
    results = {}

    def run(tag, manifest):
        r = client.post(
            "/reasoners/marker",
            json={"artifacts": {"workspace": {"manifest": manifest}}},
        )
        results[tag] = r.json()

    ta = threading.Thread(target=run, args=("a", m_a))
    tb = threading.Thread(target=run, args=("b", m_b))
    ta.start()
    tb.start()
    ta.join()
    tb.join()

    ra, rb = results["a"], results["b"]
    # Each execution saw only its own marker and its own folder.
    assert ra["seen"] == "AAA"
    assert rb["seen"] == "BBB"
    assert ra["workspace"] != rb["workspace"]
    # Neither workspace leaked the other's produced file.
    assert "wrote.txt" in ra["visible"] and "marker.txt" in ra["visible"]
    assert set(ra["visible"]) == {"marker.txt", "wrote.txt"}
    assert set(rb["visible"]) == {"marker.txt", "wrote.txt"}


@pytest.mark.integration
def test_workspace_reasoner_error_preserves_dir(af_home, tmp_path):
    from agentfield.workspace import workspaces_dir

    agent = _build_agent("err-agent")

    @agent.reasoner()
    def boom() -> dict:
        raise RuntimeError("kaboom")

    src = tmp_path / "src"
    src.mkdir()
    (src / "input.txt").write_text("data")
    store = ContentStore()
    manifest = _seal(src, store)

    client = TestClient(agent, raise_server_exceptions=False)
    resp = client.post(
        "/reasoners/boom",
        json={"artifacts": {"workspace": {"manifest": manifest}}},
    )
    assert resp.status_code == 500
    # Workspace dir kept for inspection on error.
    assert workspaces_dir().exists()
    leftovers = list(workspaces_dir().iterdir())
    assert leftovers, "expected the failed workspace to be preserved"


# ---------------------------------------------------------------------------
# Integration: serverless-mode round trip (GET /discover + POST /execute)
# ---------------------------------------------------------------------------


def _serverless_agent(node_id):
    from agentfield import Agent

    # Serverless nodes never dial back to the control plane, so no auto-register.
    return Agent(node_id=node_id, auto_register=False)


def _serverless_wrapper(agent):
    """Mirror examples/workspace-demo/main.py::_run_serverless as a TestClient app."""
    import asyncio

    from fastapi.responses import JSONResponse

    from agentfield import attach_workspace_routes

    api = FastAPI()
    attach_workspace_routes(api)

    @api.post("/execute")
    async def execute(request: Request):
        payload = await request.json()
        result = await asyncio.to_thread(
            agent.handle_serverless, {"path": "/execute", **payload}
        )
        return JSONResponse(
            content=result.get("body", result),
            status_code=result.get("statusCode", 200),
        )

    return api


@pytest.mark.integration
def test_serverless_workspace_round_trip(af_home, tmp_path):
    """prepare + upload via the wrapper, POST /execute with a workspace, and
    assert the reasoner ran with cwd == the workspace and the response body
    carries a correct top-level workspace_diff (the control plane parses it
    synchronously — the serverless node cannot call back)."""
    agent = _serverless_agent("ws-serverless")

    @agent.reasoner()
    def transform() -> dict:
        import os as _os

        data = open("input.txt").read()
        open("out.txt", "w").write(data.upper())  # new file
        open("existing.txt", "w").write("MODIFIED")  # modify existing
        _os.remove("stale.txt")  # delete existing
        return {"read": data, "cwd_env": _os.environ.get("AGENTFIELD_WORKSPACE")}

    # Build the sealed source and its blob bytes (as the CLI would).
    src = tmp_path / "src"
    src.mkdir()
    (src / "input.txt").write_text("hello")
    (src / "existing.txt").write_text("ORIGINAL")
    (src / "stale.txt").write_text("obsolete")
    manifest = build_manifest(src)
    blob_bytes = {
        e["sha256"]: (src / e["path"]).read_bytes() for e in manifest["files"]
    }

    client = TestClient(_serverless_wrapper(agent))

    # 1. Control plane asks the node which blobs it is missing...
    prep = client.post("/api/v1/workspace/prepare", json={"manifest": manifest})
    assert prep.status_code == 200
    missing = prep.json()["missing"]
    assert set(missing) == set(blob_bytes)  # node has nothing yet

    # 2. ...and uploads each missing blob to the node's CAS.
    for sha in missing:
        up = client.put(f"/api/v1/workspace/blobs/{sha}", content=blob_bytes[sha])
        assert up.status_code == 204

    # 3. Dispatch /execute in the serverless shape the control plane sends.
    exec_id = "exec_serverless_1"
    resp = client.post(
        "/execute",
        json={
            "path": "/execute/transform",
            "target": "transform",
            "reasoner": "transform",
            "input": {},
            "execution_context": {
                "execution_id": exec_id,
                "run_id": "run_serverless_1",
                "workflow_id": "run_serverless_1",
            },
            "artifacts": {"workspace": {"manifest": manifest}},
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Reasoner ran inside the materialized folder for THIS execution.
    assert body["read"] == "hello"
    assert body["cwd_env"] and body["cwd_env"].endswith(exec_id)

    # workspace_diff is a top-level sibling of the result (what execute.go parses).
    diff = body["workspace_diff"]
    changed = {e["path"] for e in diff["changed"]}
    assert changed == {"out.txt", "existing.txt"}
    assert diff["deleted"] == ["stale.txt"]

    # 4. Control plane can fetch the produced blob back from the same node URL.
    out_entry = next(e for e in diff["changed"] if e["path"] == "out.txt")
    got = client.get(f"/api/v1/workspace/blobs/{out_entry['sha256']}")
    assert got.status_code == 200
    assert got.content == b"HELLO"


@pytest.mark.integration
def test_serverless_non_workspace_execution_is_unchanged(af_home):
    agent = _serverless_agent("plain-serverless")

    @agent.reasoner()
    def echo(msg: str) -> dict:
        return {"echo": msg}

    client = TestClient(_serverless_wrapper(agent))
    resp = client.post(
        "/execute",
        json={"reasoner": "echo", "input": {"msg": "hi"}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["echo"] == "hi"
    assert "workspace_diff" not in body
