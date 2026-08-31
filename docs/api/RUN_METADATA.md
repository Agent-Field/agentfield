# Run metadata

Trusted API callers can attach a display name, labels, and external links to an existing run:

```http
POST /api/v1/runs/{run_id}/metadata
Content-Type: application/json
X-Actor-ID: release-bot

{
  "display_name": "Release verification",
  "labels": ["release", "production"],
  "links": [{"label": "Pull request", "url": "https://github.com/o/r/pull/1"}]
}
```

The response is the merged `run_metadata` object, including informational `set_by` and RFC3339 UTC `updated_at`. `set_by` is the trimmed `X-Actor-ID` header when present and `api` otherwise; actor IDs are limited to 200 Unicode code points. Run metadata is trusted-caller-writable and is not scoped per actor.

Fields are patches: omission preserves a stored field, while JSON `null` clears only that field. Display names are limited to 200 Unicode code points. There may be at most 20 labels of at most 64 code points each, and at most 10 links. Link labels are limited to 64 code points and URLs to 2048 bytes. URLs must use `http` or `https`, include a host, and contain no embedded credentials. Invalid types, an oversized actor ID, any exceeded field bound, or an invalid URL returns 400 without writing. Request bodies are limited to 128 KiB; oversized fixed-length and chunked bodies return `413` with `{"error":"request body too large"}`. A run with no execution records returns 404 without creating metadata. Once retention deletes a run's executions, this endpoint starts returning 404 even if a `workflow_runs` row still exists.

Root execute requests may include the same object as `run_metadata`. It is validated before the execution record is created and is excluded from replay matching. Child executes (those with `X-Parent-Execution-ID`) ignore it; only the root establishes run identity. Metadata persistence after a valid root execute is best effort.

Metadata appears separately as `run_metadata` on the UI run list and detail responses, full and lightweight workflow DAG responses, and at `data.run_metadata` in `GET /api/v1/agentic/run/{run_id}`. The server intentionally does not replace the existing runs-list `display_name`; the web UI prefers `run_metadata.display_name`, while Desktop's recent-executions list remains unchanged. The run list's `?search` filter matches `run_metadata.display_name` and individual labels in addition to the existing run ID, agent ID, and reasoner fields. Link URLs, provenance, lineage, and golden metadata are not part of identity search.

Non-goals in v1:

- `external_status` is out of scope and metadata never affects derived run status or active/terminal classification.
- A restart mints a new `run_id`; display name, labels, and links are not inherited.
- No SDK sends `run_metadata`; v1 is HTTP-only.
