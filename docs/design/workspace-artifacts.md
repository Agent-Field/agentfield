# Workspace artifacts (POC design)

Lets a caller attach a local folder to any reasoner execution. The platform transports
the folder to the node, the reasoner runs *inside* it, and file changes come back as a
staged diff the caller applies explicitly. Reasoner authors write no folder-handling code.

## Manifest

```json
{"version":1,"files":[{"path":"src/a.py","size":123,"mode":420,"mtime_ns":170,"sha256":"<hex>"}]}
```
- `path`: relative, `/`-separated, sorted ascending in canonical form.
- `manifest_id` = sha256 of canonical JSON (sorted keys, sorted files, no whitespace).
- Blobs are whole files keyed by sha256 (file-level dedup; chunking is a later optimization).
- Ignore rules when sealing: `.gitignore` honored, plus defaults:
  `.git/ node_modules/ .venv/ venv/ __pycache__/ target/ dist/ build/ .DS_Store`.

## Node endpoints (added automatically by the SDK to every node server)

- `POST /api/v1/workspace/prepare`  body `{"manifest": <manifest>}` → `{"missing": ["<sha256>", ...]}`
- `POST /api/v1/workspace/blobs/batch`  body is a gzip-compressed (`Content-Encoding: gzip`)
  tar stream (`Content-Type: application/x-tar`) where each entry is a regular file named by
  the blob's sha256 hex with the raw blob bytes as contents → `{"stored": <int>, "rejected":
  [{"sha256","error"}]}`. Entries are streamed and hash-verified one at a time; a blob whose
  content does not match its claimed name is rejected individually while valid entries are
  still stored. This is the fast path for cold transfers: it collapses the per-blob
  round-trip latency that used to dominate into a handful of requests. Senders split very
  large uploads into multiple batches of roughly 16 MB compressed each to bound memory and
  allow retry.
- `PUT  /api/v1/workspace/blobs/{sha256}`  raw bytes → 204
- `GET  /api/v1/workspace/blobs/{sha256}` → raw bytes (404 if absent)

Both push hops (CLI→control-plane and control-plane→node) share one sender that uploads via
the batch endpoint and, on a 404/405 from a server that predates the route, falls back
automatically to bounded-parallel per-blob PUTs (never sequential).

All transport is control-plane-initiated (works when the control plane runs behind NAT
and the node is remote). Node keeps a content store at `~/.agentfield/cas`.

## Execute request / result extensions

Request (both sync and async execute) gains an optional field:
```json
"artifacts": {"workspace": {"manifest_id": "<hex>", "manifest": { ...inline for POC... }}}
```
Result gains:
```json
"workspace_diff": {"changed": [{"path","sha256","size","mode"}], "deleted": ["path"]}
```

## Flow

1. CLI seals `--dir` → manifest + blobs into the local content store (`~/.agentfield/cas`).
2. Control plane, before dispatch: `prepare` → upload missing blobs → dispatch execute
   with `artifacts.workspace`.
3. Node SDK materializes the manifest into `~/.agentfield/workspaces/<execution_id>/`
   (copy from node CAS), spawns a worker process with cwd = that directory, and runs the
   reasoner function in the worker. `AGENTFIELD_WORKSPACE` env var exposes the absolute
   path as an escape hatch; authors normally just use relative paths.
4. On return the SDK hashes the workspace, computes the diff vs the input manifest,
   stores new blobs in the node CAS, and attaches `workspace_diff` to the result.
5. Control plane fetches missing diff blobs (`GET .../blobs/{hash}`) into its CAS and
   records a staged entry `~/.agentfield/staged/<run_id>.json`.
6. `af apply <run_id>` writes changed/deleted paths onto the original folder,
   skipping any file whose current content differs from what was sealed (conflict),
   and lists conflicts. `--force` overrides. `af diff <run_id>` prints the summary.

## CLI surface

```
af call <node.reasoner> --in '{...}' --dir <path> [--async]
af diff <run_id>
af apply <run_id> [--force]
```

## Worker semantics

One worker process per workspace-bearing execution (cwd is process-global in a server,
so per-request chdir is impossible; a worker per call also gives crash isolation and
concurrency safety). Python: multiprocessing with a fresh event loop in the child;
results returned over a pipe. Non-workspace executions are unchanged.

## Out of scope for the POC

Content-defined chunking, lazy materialization, delta compression, workspace encryption,
apply-time three-way merge (POC = conflict-skip + list), reflink/clonefile fast paths.

Lazy on-demand materialization (transporting only the blobs a reasoner actually opens,
rather than the whole manifest up front) remains a compatible future optimization: it sits
behind the same manifest abstraction and the same content-store endpoints, so it can be
added without changing the wire contract above.
