# App link contract

This page describes how an app registers as a node over an outbound link and works with reasoners: reasoners can start work in the app, and the app can offload work to reasoners while watching, steering and answering their questions. One app built this way is WorkAF, which registers as a node, receives work through an `intake` reasoner, and watches and steers the runs it starts. Nothing below is specific to it.

## Connect an app

Register the app as a node with `POST /api/v1/nodes/register`, including its node ID and any reasoners it serves. Then open one outbound WebSocket to `GET /api/v1/nodes/:node_id/link`. The app can receive calls without exposing an inbound port. The Go SDK registers and connects for you. Set `AGENTFIELD_NODE_LINK_ONLY=1` for link-only operation. `AGENTFIELD_NODE_LINK=off` disables the SDK link. A link-only node can advertise a `base_url` on the reserved host `<node_id>.nodelink`.

Send the control plane's API key or internal token in `X-API-Key` or `Authorization: Bearer <credential>`. The upgrade accepts credentials only in headers and rejects requests carrying `Origin`. A control plane with neither credential configured allows an unauthenticated upgrade.

The control plane sends `welcome` with `link_id`, `heartbeat_s`, and `max_frame`. The node sends `hello` with its `instance_id` and the delivery kinds it `accepts`. An empty list means `dispatch` and `cancel`. The other kinds are `http` for carried HTTP requests and `steer` for dedicated note deliveries. Wait for the `ok` to `hello` before sending other frames. Respond to WebSocket pings and send `heartbeat` to renew the node lease and report in-flight executions.

## Frames

Each WebSocket message carries a JSON `Frame` with `v: 1` and the frame type in `t`. Use `exec` when the frame concerns an execution, `id` to correlate a delivery with its acknowledgement, and `seq` to correlate an up-frame with its `ok`. Relay entries carry a separate sequence in the body. The control plane caps a frame at 1 MiB.

| Frame | Direction | Body | Purpose |
| --- | --- | --- | --- |
| `welcome` | Control plane → node | `Welcome` | Gives the link ID, ping interval, and frame limit. |
| `hello` | Node → control plane | `Hello` | Identifies the instance and advertises the delivery kinds it serves. |
| `ok` | Control plane → node | `OKBody` | Answers an up-frame on its `seq` with a `code` and optional `error` or result fields. |
| `heartbeat` | Node → control plane | `Heartbeat` | Renews the lease and can report `inflight` and `health_score`. |
| `deliver` | Control plane → node | `DeliverBody` | Carries a reasoner dispatch, cancellation, HTTP request, or steer. |
| `ack` | Node → control plane | `Ack` | Settles a `deliver` by ID with its outcome and optional result or error. |
| `status` | Node → control plane | `executionStatusUpdateRequest` | Updates an execution using the body of `POST /api/v1/executions/:execution_id/status`. |
| `progress` | Node → control plane | `executionStatusUpdateRequest` | Updates an execution with `status` set to `running`. |
| `log` | Node → control plane | `ExecutionLogEntry`, array, or `executionLogEnvelope` | Sends structured execution logs using the body of `POST /api/v1/executions/:execution_id/logs`. |
| `subscribe` | Node → control plane | `Subscribe` | Requests snapshots and held entries for runs this node owns. |
| `run` | Control plane → node | `RunState` | Sends a tree snapshot or changed executions, including any pending approval. |
| `entries` | Both | `EntriesUp` or `EntriesDown` | Carries a worker's relay entries up and sends them down to the run owner. |
| `entries_ack` | Node → control plane | `EntriesAck` | Confirms the owner has stored the execution's entries through `through`. |
| `steer` | Node → control plane | `SteerUp` | Sends a person's `message` to an execution in an owned run. |
| `delegate` | Node → control plane | `DelegateUp` | Requests a proof of consent for this link's node. |

The control plane keeps pending deliveries in memory and can redeliver them until the node acknowledges them. Deduplicate by delivery `id`.

## Run ownership and proof of consent

On a root execute request, `X-AgentField-Run-Owner` names the node that should receive the run tree, transcript, and approval questions. The first recorded owner for a run ID wins. Nested calls inherit their run's owner and ignore any owner claim. Trigger dispatch assigns ownership to the target node directly.

Mint a proof over the owner's link by sending `delegate` with a non-empty `scopes` array, optional `ttl_seconds`, and optional `run` restriction. The control plane signs for the node attached to the link. The `ok` body includes `token`, RFC3339 `expires_at`, `expires_in` seconds, and the granted `scopes`. The lifetime defaults to one hour and is capped at 24 hours. Verification allows 60 seconds of leeway after expiry.

The token starts with `afo1` and contains a base64url JSON payload and HMAC-SHA256 signature. The payload includes the key ID, owner, granted scopes, optional run, issue and expiry times, and a random log handle. Send the token in `X-AgentField-Run-Owner-Proof` alongside `X-AgentField-Run-Owner`. The Go SDK can mint it with `Agent.RunOwnerProof` and exposes `RunOwnerProofHeader` for that header.

| Scope | Allows the named owner to |
| --- | --- |
| `own` | Claim ownership on a root execute. |
| `read` | Read relay entries from an owned run. |
| `act` | Acknowledge relay entries, steer an execution, or answer its approval in an owned run. |

A run-restricted proof applies only to executions of that run. For a root claim, send the matching `X-Run-ID`. On the legacy reasoner and skill execute routes, the stored run ID also comes from `X-Workflow-ID`, so both headers must agree for a run-restricted claim.

By default, owner claims without a proof still work and the control plane logs a warning at most once per owner and scope every 10 minutes. Set `agentfield.run_owner.require_proof` or `AGENTFIELD_RUN_OWNER_REQUIRE_PROOF=true` to reject an owner claim without a proof. Both modes reject an invalid, expired, or mismatched proof with HTTP 403. Traffic on a node link uses the link's own node ID for scope. Relay uploads and relay counts do not require an owner proof. A control plane without relay storage returns 501 for `delegate` and retains legacy owner-claim behavior.

Set `agentfield.run_owner.proof_key` or `AGENTFIELD_RUN_OWNER_PROOF_KEY` to a signing secret of at least 16 bytes. Otherwise, the control plane uses a random 32-byte secret persisted as `run_owner_proof.key` with mode 0600 in its data directory. If persistence is unavailable, it warns and uses a process-local key. When several control plane replicas do not share a data directory, set the signing secret explicitly so every replica verifies every proof. Rotating the signing key revokes prior proofs and their holders must mint new ones. The counter `agentfield_run_owner_claims_total{result}` reports `proven`, `unproven_accepted`, `refused_missing_proof`, and `refused_invalid_proof`.

**Trust model.** Strict mode prevents a caller using the shared API key from naming an app that did not consent and thereby pushing a run or its approval questions to that app or reading relay entries under its name. The link itself uses the shared API key or internal token, and a newer link for a node replaces the older link. The proof shows that a link with that node name consented. The person or machine behind the link is outside that proof. A caller may omit the owner header and act as the operator under the ordinary API authorization rules. Treat the proof as a bearer credential. There is no individual revocation list, so expiry and key rotation bound its lifetime.

## Watch an execution

When a run has an owner, the control plane sets `X-AgentField-Run-Watched: 1` on the execution delivery. The node can stream entries for that execution. Each execution numbers entries from 1. Deduplicate on execution ID and entry `seq`, including across control-plane restarts.

| Entry kind | Data |
| --- | --- |
| `started` | `{harness, steerable}` describes the live run and whether it takes notes. |
| `line` | A JSON transcript line supplied by the node. |
| `person` | `{text}` records a steer note the node accepted. |
| `ended` | `{}` marks the end of the live run. |

An owner sends `subscribe` to get run trees and relay entries. Its optional `after` map gives the last entry already stored per execution, and optional `runs` asks for fresh trees for specific runs the node owns. After storing the entries, acknowledge them with `entries_ack`. The control plane deletes acknowledged entries and holds unacknowledged entries for at most the execution-log retention period, `agentfield.execution_logs.retention_period`, which defaults to 24 hours.

The same relay is available over HTTP.

| Route | Behavior |
| --- | --- |
| `POST /api/v1/executions/:execution_id/relay` | Upload `{entries}` and receive `stored` and `through`. |
| `GET /api/v1/executions/:execution_id/relay` | Read entries after `after` with optional `limit`. The response includes `entries` and the `next` cursor. |
| `POST /api/v1/executions/:execution_id/relay/ack` | Acknowledge `{through}` and receive a `dropped` count. |
| `GET /api/v1/relay/count` | Return a `held` count, optionally scoped by the `execution_id` query parameter. |

For a plain Go reasoner, call `Agent.LiveRun(ctx)`. When it returns a live run, use `Begin` with a `harness.LiveInfo`, `Emit` for JSON lines, `Notes` to receive steering, and `End` to finish and flush. For a custom harness, implement `harness.LiveProvider` and register its factory with `harness.RegisterProvider`. CodeAF is the built-in live implementation.

## Steer

Send `POST /api/v1/executions/:execution_id/steer` with `{"message":"Check the figures"}`. The trimmed message must be 1 to 4,000 bytes. An HTTP call made as a named owner is scoped to that owner's runs and uses an `act` proof. An owner can also send a `steer` frame on its link.

The executing node receives a dedicated `steer` delivery or `POST /_internal/executions/:id/steer`. The latter can arrive as a carried `http` delivery or over HTTP. A successful response is HTTP 200 with `ok: true` and `via` set to `link` or `http`. An unknown execution returns 404. An ended execution, unavailable node, or refused note returns 409 with the reason.

## Pause with a question

A running execution sends `POST /api/v1/executions/:execution_id/request-approval` to enter the waiting state. Send `kind` as `plan_review`, `question`, or `approval`, plus `question` for the prompt, `body` for longer Markdown, and `options` for display choices. If `approval_request_id` is omitted, the control plane assigns one.

The owner's `run` frame includes an `approval` object on the waiting execution with `id`, `kind`, `question`, `body`, `options`, and `expires_at` when present. If the snapshot sets `body_omitted`, fetch the question with `GET /api/v1/executions/:execution_id/approval-status`.

Resolve it with `POST /api/v1/executions/:execution_id/approval-response`. The request takes `decision` (`approved`, `rejected`, or `request_changes`), optional `reason`, and optional JSON `response`. For an owner-scoped request use an `act` proof. An optional external approval service can post to `POST /api/v1/webhooks/approval-response` using the corresponding `requestId` and an HMAC-SHA256 signature in `X-Webhook-Signature`. Configure the shared signing secret with `agentfield.approval.webhook_secret` or `AGENTFIELD_APPROVAL_WEBHOOK_SECRET`.

Both endpoints share one resolution path, so the first answer wins: once the execution has left the waiting state, any later answer, from either endpoint, gets HTTP 200 with `status: already_processed` and changes nothing.

## Reasoner conventions

Use a tag such as `label:Intake` (`label:<text>`) if your app shows display labels. The app interprets the convention. The control plane treats it as an ordinary reasoner tag.

A reasoner starts work in an app by calling one of the app node's reasoners. For a link-only app accepting `dispatch`, the control plane sends a `deliver` with `kind: "dispatch"`, `reasoner: "intake"`, and the input and execution headers. If the app accepts `http`, it receives the same invocation as a carried HTTP request.

```http
POST /api/v1/execute/my-app.intake
Content-Type: application/json

{"input":{"task":"Prepare the report"}}
```

For a dispatch completed synchronously, the app returns an `ack` with the delivery's `id` and a body such as `{"outcome":"done","result":{"accepted":true}}`. The execute response includes `"status":"succeeded"` and `"result":{"accepted":true}`.

## Walkthrough

1. Register `my-app`, open its outbound link, and complete the `hello` exchange.
2. Send `delegate` with `{"scopes":["own","read","act"]}` and keep the proof from the `ok` response.
3. Send `subscribe`. Start a root execution with `X-AgentField-Run-Owner: my-app` and the proof header.
4. Read the `run` and `entries` frames and acknowledge stored relay entries. When a live execution reports that it takes notes, send a `steer` frame with its execution ID.
5. When a `run` frame carries an approval question, display it and send the answer to `approval-response` with the owner and proof headers.

## Follow-ups

- Add Python and TypeScript SDK parity for the link, relay, and steering.
- Give each node its own link credentials.
- Add an HTTP mint route.
- Add an individual proof revocation list.
- Make concurrent approval answers a single atomic update: two answers racing within milliseconds can both pass the waiting check today.
