"""
Workspace Demo Agent

Demonstrates the workspace-artifacts feature (see docs/design/workspace-artifacts.md):
a caller attaches a local folder to a reasoner call, the platform transports it to the
node, the reasoner reads/writes it like an ordinary working directory, and any changes
come back to the caller as a staged diff. The reasoners below do no folder-handling
code themselves -- the SDK materializes the workspace as the process cwd before the
function runs.

Two deterministic reasoners, neither of which needs LLM credentials:

- `report`: walks the current directory, writes REPORT.md summarizing file count,
  size, and language breakdown, and appends a "reviewed" marker line to any *.py
  files that don't already have one.
- `apply_note`: writes or updates a single file with a note, given
  {"filename": ..., "note": ...}.

Runs in two modes, selected by the AGENTFIELD_SERVERLESS env var:
- Normal (default): a long-lived server that registers with the control plane and
  sends heartbeats, exactly like examples/python_agent_nodes/docker_hello_world.
  Used for local development and the workspace e2e check.
- Serverless (AGENTFIELD_SERVERLESS=true): a thin FastAPI wrapper around
  Agent.handle_serverless exposing GET /discover and POST /execute, with no
  heartbeat loop. Used for the fly.io deployment in this directory (see deploy.md),
  where the control plane discovers and invokes the node on demand via
  `af nodes register-serverless`.
"""

import fnmatch
import os
from pathlib import Path

from agentfield import Agent

# ============= CONFIG =============

SERVERLESS_MODE = os.getenv("AGENTFIELD_SERVERLESS", "false").strip().lower() in (
    "1",
    "true",
    "yes",
)


def _dev_mode_enabled() -> bool:
    return os.getenv("AGENTFIELD_DEV_MODE", "true").strip().lower() not in (
        "0",
        "false",
        "no",
    )


_agent_kwargs = dict(
    node_id=os.getenv("AGENT_NODE_ID", "workspace-demo"),
    agentfield_server=os.getenv(
        "AGENTFIELD_URL", os.getenv("AGENTFIELD_SERVER", "http://localhost:8080")
    ),
    dev_mode=_dev_mode_enabled(),
    # Serverless deployments (fly.io) don't run a heartbeat loop -- the control
    # plane discovers and invokes them on demand instead. See serverless_hello
    # in examples/python_agent_nodes for the reference pattern this mirrors.
    auto_register=not SERVERLESS_MODE,
)

if SERVERLESS_MODE:
    from agentfield.async_config import AsyncConfig

    _agent_kwargs["async_config"] = AsyncConfig(
        enable_async_execution=False, fallback_to_sync=True
    )

app = Agent(**_agent_kwargs)

# ============= REPORT REASONER =============

# Default ignore rules mirrored from docs/design/workspace-artifacts.md so the
# report stays sane even when run directly against a non-sealed directory
# (e.g. a plain `af call workspace-demo.report` with no --dir).
IGNORE_DIRS = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    "target",
    "dist",
    "build",
}
IGNORE_FILES = {".DS_Store"}

REVIEWED_MARKER = "# reviewed: workspace-demo"

LANGUAGE_BY_EXT = {
    ".py": "Python",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".go": "Go",
    ".rs": "Rust",
    ".java": "Java",
    ".rb": "Ruby",
    ".md": "Markdown",
    ".json": "JSON",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".toml": "TOML",
    ".sh": "Shell",
    ".html": "HTML",
    ".css": "CSS",
    ".sql": "SQL",
    ".c": "C",
    ".h": "C",
    ".cpp": "C++",
    ".hpp": "C++",
}

REPORT_FILENAME = "REPORT.md"


def _load_gitignore_patterns(root: Path) -> list:
    """Best-effort .gitignore reader (flat glob patterns only).

    The CLI/control plane are responsible for the authoritative ignore
    filtering when sealing a workspace for transport; this is a defense-in-
    depth pass so the report is still sensible when run against an
    un-sealed directory.
    """
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        return []
    patterns = []
    for line in gitignore.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(line.rstrip("/"))
    return patterns


def _is_ignored(rel_path: Path, patterns: list) -> bool:
    rel_str = rel_path.as_posix()
    for pattern in patterns:
        if fnmatch.fnmatch(rel_str, pattern) or fnmatch.fnmatch(rel_path.name, pattern):
            return True
    return False


@app.reasoner()
async def report() -> dict:
    """Walk the current directory and write a REPORT.md summary.

    Counts files, total size, and a per-language breakdown by extension.
    Any *.py file missing a "reviewed" marker line gets one appended.
    Intended to run against a workspace materialized via `--dir` (see
    docs/design/workspace-artifacts.md), but works against any cwd.
    """
    root = Path(".")
    gitignore_patterns = _load_gitignore_patterns(root)

    discovered = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".git")
        ]
        for name in filenames:
            if name in IGNORE_FILES:
                continue
            full = Path(dirpath) / name
            rel = full.relative_to(root)
            if rel.name == REPORT_FILENAME:
                continue
            if _is_ignored(rel, gitignore_patterns):
                continue
            discovered.append(rel)

    by_language: dict = {}
    total_size = 0
    reviewed_added = []

    for rel in sorted(discovered, key=lambda p: p.as_posix()):
        size = rel.stat().st_size
        language = LANGUAGE_BY_EXT.get(rel.suffix.lower(), "Other")
        bucket = by_language.setdefault(language, {"files": 0, "size": 0})
        bucket["files"] += 1
        bucket["size"] += size
        total_size += size

        if rel.suffix.lower() == ".py":
            text = rel.read_text(encoding="utf-8", errors="ignore")
            # Match a standalone line, not a substring -- otherwise a file that
            # merely *mentions* the marker (e.g. this script's own
            # REVIEWED_MARKER constant) would be mistaken for already-reviewed.
            already_reviewed = any(
                line.strip() == REVIEWED_MARKER for line in text.splitlines()
            )
            if not already_reviewed:
                if text and not text.endswith("\n"):
                    text += "\n"
                text += REVIEWED_MARKER + "\n"
                rel.write_text(text, encoding="utf-8")
                reviewed_added.append(rel.as_posix())

    lines = [
        "# Workspace Report",
        "",
        "Generated by `workspace-demo.report`.",
        "",
        "## Summary",
        f"- Files: {len(discovered)}",
        f"- Total size: {total_size} bytes",
        "",
        "## By language",
        "",
        "| Language | Files | Size (bytes) |",
        "|---|---|---|",
    ]
    for language in sorted(by_language):
        bucket = by_language[language]
        lines.append(f"| {language} | {bucket['files']} | {bucket['size']} |")

    lines.append("")
    lines.append("## Reviewed marker added to")
    if reviewed_added:
        for path in reviewed_added:
            lines.append(f"- {path}")
    else:
        lines.append("- (none -- all Python files already reviewed)")
    lines.append("")

    Path(REPORT_FILENAME).write_text("\n".join(lines), encoding="utf-8")

    return {
        "files": len(discovered),
        "total_size": total_size,
        "by_language": by_language,
        "reviewed_added": reviewed_added,
        "report_path": REPORT_FILENAME,
    }


# ============= APPLY_NOTE REASONER =============


@app.reasoner()
async def apply_note(filename: str, note: str) -> dict:
    """Write or update `filename` in the current directory with `note`.

    Creates the file (and any parent directories, within the current
    directory) if it doesn't exist. If it already exists, the note is
    appended as a new line so repeated calls accumulate history instead of
    clobbering prior content.
    """
    target = Path(filename)
    if target.is_absolute() or ".." in target.parts:
        raise ValueError(
            f"filename must be a relative path inside the workspace, got: {filename!r}"
        )

    resolved = target.resolve()
    cwd = Path(".").resolve()
    if cwd not in resolved.parents and resolved != cwd:
        raise ValueError(f"filename resolves outside the workspace: {filename!r}")

    if target.parent != Path("."):
        target.parent.mkdir(parents=True, exist_ok=True)

    existed = target.exists()
    prior = target.read_text(encoding="utf-8") if existed else ""
    if prior and not prior.endswith("\n"):
        prior += "\n"

    target.write_text(prior + note + "\n", encoding="utf-8")

    return {
        "filename": target.as_posix(),
        "created": not existed,
        "bytes_written": target.stat().st_size,
    }


# ============= SERVERLESS WRAPPER (fly.io / register-serverless) =============


def _run_serverless() -> None:
    """Expose GET /discover and POST /execute for `af nodes register-serverless`.

    Mirrors examples/python_agent_nodes/serverless_hello and the test fixture in
    tests/functional/tests/test_serverless_agents.py -- this is the documented
    shape a control plane expects when it discovers and invokes a node purely by
    URL (no heartbeat loop).
    """
    import asyncio

    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    from agentfield import attach_workspace_routes

    api = FastAPI()

    # Serve the workspace transport (prepare / put-blob / get-blob) on the same
    # URL the control plane reaches this node by. Without these, workspace-bearing
    # calls to a serverless node have nowhere to upload sealed blobs to or fetch
    # the diff back from. handle_serverless materializes + diffs the workspace
    # synchronously and returns workspace_diff on the /execute response body.
    attach_workspace_routes(api)

    @api.get("/discover")
    async def discover():
        return await asyncio.to_thread(app.handle_serverless, {"path": "/discover"})

    @api.post("/execute")
    async def execute(request: Request):
        payload = await request.json()
        result = await asyncio.to_thread(
            app.handle_serverless, {"path": "/execute", **payload}
        )
        status = result.get("statusCode", 200)
        body = result.get("body", result)
        return JSONResponse(content=body, status_code=status)

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(api, host=host, port=port)


# ============= ENTRY POINT =============

if __name__ == "__main__":
    if SERVERLESS_MODE:
        print("Workspace Demo Agent (serverless mode: /discover + /execute)")
        _run_serverless()
    else:
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8001"))
        print("Workspace Demo Agent")
        print(f"Node: {app.node_id}")
        print(f"Control plane: {os.getenv('AGENTFIELD_URL', 'http://localhost:8080')}")
        # For containerized runs, set AGENT_CALLBACK_URL so the control plane can
        # call back: AGENT_CALLBACK_URL=http://<service-name>:<port>
        app.run(host=host, port=port, auto_port=False)
