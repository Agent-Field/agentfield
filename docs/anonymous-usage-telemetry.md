# Anonymous token reconciliation

When anonymous telemetry is enabled, the control plane reports `usage_delta`
for each persisted SDK-native `source=llm` usage entry. Harness entries and
entries with unknown sources are excluded from anonymous totals, preventing
external coding-agent rollups from overlapping self-reported usage. Local
accounting retains those entries unchanged. It reports entries belonging to the current
execution only; parent workflow totals are never added over child usage.
The event contains bounded `routing_provider`, `model_family`, `usage_status`,
`accounting_source=control_plane`, usage context, release version and token counts.
Raw model names, prompts, execution IDs, reasoner names and endpoint URLs are
not included. Routing provider is explicit SDK request routing; older SDK
entries lacking it report `unknown`, even if their model vendor is known.

Python and TypeScript streaming accounting observes caller consumption and does
not drain an abandoned stream in the background. A missing final receipt is
marked `missing` on exhaustion, cancellation or explicit close and contributes no tokens in the anonymous event. Go similarly
observes final usage chunks. Missing receipt amounts cannot be reconstructed.

Sanitized usage events are written and synced under
`<agentfield-home>/telemetry/usage-outbox` before sending. Failed delivery retries
on a ten-second interval and survives control-plane restart. A stable HMAC of
installation, execution and entry index enables relay deduplication; acknowledgement
files also suppress callback replays locally. The existing terminal callback
no-op guard prevents reingesting the same terminal result. The SDK must retain
entry ordering when retrying an execution envelope.

The queue accepts at most 10,000 pending events, with a cached pending count
initialized from disk on first enqueue after restart. Acknowledgements live in
a separate directory so saving and flushing new receipts does not scan historical
acknowledgements. Cleanup runs at startup and every ten minutes, retaining at most
10,000 recent acknowledgements after each cleanup. Pending events and acknowledgements
expire after 30 days; rejected or expired pending events produce local warnings.
Telemetry opt-out disables both enqueue and delivery. Local accounting is unchanged.
These events are useful coverage measurements, not an exact billing ledger:
SDK entries still travel to the control plane in execution result envelopes,
so an agent process killed before returning that envelope can lose usage; direct
provider calls outside SDK accounting and opted-out installations are unobserved.
A missing receipt has no known token total. Usage timestamps currently reflect
control-plane ingestion, so long-running executions can move usage across days.
The hosted relay must accept this additive schema before deploying clients.

Request routing is separate from the model vendor and SDK adapter. An explicit
OpenRouter endpoint is classified as `openrouter` even through an OpenAI adapter.
Only exact known endpoint hosts identify those routes; a private or custom
endpoint is `other` (or `unknown` if invalid), rather than inferred from URL
path text or an OpenRouter model prefix. With no explicit endpoint, the SDK's
requested provider determines the bounded route. Endpoint URLs are never sent.
