# Workspace Demo

A minimal AgentField node demonstrating **workspace artifacts** -- the
feature that lets a caller attach a local folder to a reasoner call. The
platform transports the folder to the node, the reasoner reads and writes it
like an ordinary working directory, and any changes come back to the caller
as a diff that's applied explicitly. The reasoners below contain zero
folder-transport code; that's entirely the SDK's job.

Full design: [`docs/design/workspace-artifacts.md`](../../docs/design/workspace-artifacts.md).

Both reasoners are deterministic -- **no LLM API keys required**.

## Reasoners

### `report`

Walks the current directory and writes `REPORT.md` summarizing:
- total file count and size
- a per-language breakdown (by file extension)

Any `*.py` file missing a `# reviewed: workspace-demo` marker line gets one
appended, so a second run against the same workspace is a no-op for files
already touched.

```bash
af call workspace-demo.report --dir ./some-folder
af diff <run_id>     # see what would change
af apply <run_id>    # write REPORT.md + the reviewed markers back to ./some-folder
```

### `apply_note`

Writes or updates a single file with a note.

```bash
af call workspace-demo.apply_note --dir ./some-folder \
  --in '{"filename": "NOTES.md", "note": "checked by workspace-demo"}'
af apply <run_id>
```

If the file already exists, the note is appended as a new line (accumulating
history) rather than overwriting it.

## Running it locally

```bash
cd examples/workspace-demo
python main.py
```

Reads `AGENTFIELD_URL` (default `http://localhost:8080`) for the control
plane and `PORT` (default `8001`) for its own listener. Registers itself with
a running control plane the same way any other node does (see the repo's
`examples/python_agent_nodes/docker_hello_world` for the reference pattern
this mirrors) -- no `--dir` needed for that part.

See [`../../scripts/workspace-e2e.sh`](../../scripts/workspace-e2e.sh) for a
scripted, self-checking run of the full `--dir` / `af diff` / `af apply` flow
against a throwaway fixture folder.

## Deploying it

See [`deploy.md`](deploy.md) for building the container image, deploying it
somewhere with a public https URL, and registering it with a control plane
via `af nodes register-serverless`.
