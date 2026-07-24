# VC-Based Authorization Architecture

**Version:** 2.0
**Status:** Implemented
**Date:** February 2026

---

## Executive Summary

This document describes the Verifiable Credential (VC) based authorization system for AgentField. The system implements a two-step authorization model: **tag approval** (admin decides which tags agents get) and **policy evaluation** (policies decide which tagged agents can call which other tagged agents).

**Key Principles:**
- Agents propose tags at registration (agent-level and per-skill/per-reasoner)
- Control plane evaluates proposed tags against configurable approval rules (`auto`/`manual`/`forbidden`)
- Default mode is `auto` (all tags auto-approved) for zero-disruption backward compatibility
- Tags requiring `manual` approval put agents into `pending_approval` state until admin reviews
- `forbidden` tags reject registration outright (HTTP 403)
- Tag-based access policies control which agents can call which functions with parameter constraints
- Upon tag approval, control plane issues a signed `AgentTagVC` (W3C Verifiable Credential) per agent
- Permission middleware evaluates tag-based access policies; no policy match allows the request (backward compatible)
- `did:web` enables real-time revocation via control plane-hosted DID documents
- SDKs support decentralized verification: agents cache policies, revocation lists, and admin public keys locally
- Control plane is source of truth; agents can verify locally without hitting control plane on every call

---

## System Overview

### Flow 1: Registration with Tag Approval

When an agent registers, it proposes tags at both the agent level and per-skill/per-reasoner level. The control plane evaluates each proposed tag against configured approval rules.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     REGISTRATION WITH TAG APPROVAL                          │
└─────────────────────────────────────────────────────────────────────────────┘

  AGENT                         CONTROL PLANE                         ADMIN
     │                              │                                  │
     │  1. Register                 │                                  │
     │  ─────────────────────────►  │                                  │
     │  {                           │                                  │
     │    id: "finance-bot",        │                                  │
     │    skills: [{                │                                  │
     │      id: "charge",           │                                  │
     │      proposed_tags:          │                                  │
     │        ["finance","payment"] │                                  │
     │    }],                       │                                  │
     │    reasoners: [{             │                                  │
     │      id: "analyze",          │                                  │
     │      proposed_tags: ["nlp"]  │                                  │
     │    }]                        │                                  │
     │  }                           │                                  │
     │                              │                                  │
     │                              │  2. Evaluate tag approval rules: │
     │                              │     "finance" → manual ⏸        │
     │                              │     "payment" → manual ⏸        │
     │                              │     "nlp"     → auto ✓          │
     │                              │                                  │
     │                              │  3. Set status: pending_approval │
     │                              │     Auto-approve "nlp"           │
     │                              │                                  │
     │  4. Response                 │                                  │
     │  ◄─────────────────────────  │                                  │
     │  {                           │                                  │
     │    status: pending_approval, │                                  │
     │    pending_tags: [finance,   │                                  │
     │                   payment],  │                                  │
     │    auto_approved: [nlp]      │                                  │
     │  }                           │                                  │
     │                              │                                  │
     │                              │  5. Show in Admin UI             │
     │                              │  ─────────────────────────────►  │
     │                              │                                  │
     │                              │           6. Approve/Modify/Reject│
     │                              │  ◄─────────────────────────────  │
     │                              │                                  │
     │                              │  7. Issue AgentTagVC             │
     │                              │     Set status: starting         │
     │                              │                                  │
     │  8. Agent activates          │                                  │
     │  ◄─────────────────────────  │                                  │
```

### Flow 2: Runtime Permission Check (Policy Engine)

When an agent calls another agent, the permission middleware evaluates tag-based access policies to decide whether to allow or deny the request.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     RUNTIME PERMISSION CHECK (POLICY ENGINE)                │
└─────────────────────────────────────────────────────────────────────────────┘

  AGENT A                     CONTROL PLANE                         AGENT B
  (caller)                   (permission middleware)                (target)
     │                              │                                  │
     │  1. POST /execute/B.func     │                                  │
     │  Headers:                    │                                  │
     │    X-Caller-DID              │                                  │
     │    X-DID-Signature           │                                  │
     │    X-DID-Timestamp           │                                  │
     │  ─────────────────────────►  │                                  │
     │                              │                                  │
     │                              │  2. DID Auth Middleware:          │
     │                              │     Verify Ed25519 signature     │
     │                              │     Check replay protection      │
     │                              │     Check timestamp window       │
     │                              │                                  │
     │                              │  3. Is B in pending_approval?    │
     │                              │     YES → 503 Unavailable        │
     │                              │                                  │
     │                              │  4. Tag Policy Evaluation:       │
     │                              │     Load caller's AgentTagVC     │
     │                              │     Verify VC signature          │
     │                              │     Get caller tags from VC      │
     │                              │     Get target tags              │
     │                              │     Evaluate access policies:    │
     │                              │       caller_tags match?         │
     │                              │       target_tags match?         │
     │                              │       function allowed?          │
     │                              │       constraints satisfied?     │
     │                              │                                  │
     │                              │  5. If policy matched → decide   │
     │                              │     If no policy → allow         │
     │                              │     (backward compat)            │
     │                              │                                  │
     │                              │  6. ALLOW → forward to B         │
     │                              │  ───────────────────────────────►│
     │                              │                                  │
     │  7. Result                   │                                  │
     │  ◄─────────────────────────  │  ◄──────────────────────────────│
```

### Flow 3: Revocation

Admin can revoke an agent's tags at any time. The agent returns to `pending_approval` and subsequent calls fail.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              REVOCATION                                     │
└─────────────────────────────────────────────────────────────────────────────┘

  AGENT A                     CONTROL PLANE                         ADMIN
     │                              │                                  │
     │                              │     1. Revoke Agent B's tags     │
     │                              │  ◄─────────────────────────────  │
     │                              │                                  │
     │                              │  2. Clear approved_tags           │
     │                              │     Revoke AgentTagVC             │
     │                              │     Set status: pending_approval  │
     │                              │                                  │
     │  3. Call Agent B             │                                  │
     │  ─────────────────────────►  │                                  │
     │                              │                                  │
     │                              │  4. B is pending_approval → 503  │
     │                              │                                  │
     │  5. Error: Agent             │                                  │
     │     unavailable              │                                  │
     │  ◄─────────────────────────  │                                  │
```

### Flow 4: Decentralized Verification

Agents cache policies, revocation lists, and admin public keys locally. Verification happens without hitting the control plane on every call.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       DECENTRALIZED VERIFICATION                            │
└─────────────────────────────────────────────────────────────────────────────┘

  AGENT B                     CONTROL PLANE
  (target, verifying locally)
     │                              │
     │  1. On startup / every 5min  │
     │  GET /api/v1/policies        │
     │  ─────────────────────────►  │
     │  ◄─────────────────────────  │  (cache policies)
     │                              │
     │  GET /api/v1/revocations     │
     │  ─────────────────────────►  │
     │  ◄─────────────────────────  │  (cache revoked DIDs)
     │                              │
     │  GET /api/v1/admin/public-key│
     │  ─────────────────────────►  │
     │  ◄─────────────────────────  │  (cache issuer public key)
     │                              │
     │                              │
  AGENT A ──► AGENT B (direct call)
     │  2. Incoming request         │
     │     from Agent A             │
     │                              │
     │  3. Local verification:      │
     │     - Check caller DID       │
     │       not in revocation list │
     │     - Verify Ed25519 sig     │
     │       using cached pub key   │
     │     - Evaluate policies      │
     │       using cached rules     │
     │     - Check constraints      │
     │                              │
     │  4. ALLOW/DENY locally       │
     │     (no control plane call)  │
```

---

## Core Concepts

### 1. Agent Identity (Tags)

Agents propose tags at registration at both the agent level and per-skill/per-reasoner level. The control plane evaluates proposed tags against configurable approval rules.

```python
# Python SDK — per-skill tags
app = Agent(node_id="finance-bot")

@app.reasoner(tags=["pci-compliant", "finance"])
async def process_payment(input_data):
    ...

@app.skill(tags=["finance", "reporting"])
async def get_balance(customer_id: str):
    ...
```

```go
// Go SDK — per-reasoner tags
agent.RegisterReasoner("payment", handler,
    agent.WithReasonerTags("pci-compliant", "finance"),
)
```

```typescript
// TypeScript SDK — per-skill tags
agent.registerReasoner("payment", handler, {
    tags: ["pci-compliant", "finance"],
});
```

**Tag Data Model:**

Each `ReasonerDefinition` and `SkillDefinition` has three tag fields:

| Field | Purpose |
|-------|---------|
| `tags` | Original tags declared by developer (backward compatibility) |
| `proposed_tags` | Tags proposed for approval (copied from `tags` if not set) |
| `approved_tags` | Tags granted by admin or auto-approval |

Agent-level `proposed_tags` is the union of all per-skill/per-reasoner proposed tags (computed by `CollectAllProposedTags()`). Agent-level `approved_tags` is the union of all per-skill/per-reasoner approved tags.

**Tag Lifecycle:**

| Stage | Field | Description |
|-------|-------|-------------|
| Registration | `proposed_tags` | Tags the developer wants (per skill/reasoner and agent level) |
| Evaluation | Tag approval rules | Control plane checks each tag against rules |
| Approved | `approved_tags` | Tags the admin (or auto-approval) grants |
| Runtime | `CanonicalAgentTags()` | Prefers `approved_tags`, falls back to `tags`; normalized, deduplicated |

**Tag Normalization:**
- All tags lowercased and trimmed
- Empty strings filtered out
- Duplicates removed
- Case-insensitive comparison for rule matching
- Deployment metadata tags excluded from canonical tags

**Tag Approval Modes:**

| Mode | Behavior |
|------|----------|
| `auto` (default) | Tags auto-approved, agent proceeds immediately |
| `manual` | Agent enters `pending_approval` state, waits for admin review |
| `forbidden` | Registration rejected outright (HTTP 403) |

Tags serve as:
- **Identity declaration** - "I am a finance agent"
- **Capability advertisement** - "I handle PCI-compliant operations"
- **Authorization scope** - Determines which access policies apply
- **Discovery metadata** - Other agents can find me by tags
- **Admin context** - Helps admin decide whether to approve

### 2. Agent Lifecycle States

```go
type AgentLifecycleStatus string

const (
    AgentStatusStarting        = "starting"          // Initializing
    AgentStatusReady           = "ready"             // Fully operational
    AgentStatusDegraded        = "degraded"          // Partial functionality
    AgentStatusOffline         = "offline"           // Not responding
    AgentStatusPendingApproval = "pending_approval"  // Awaiting tag approval
)
```

**State Transitions:**

| Current Status | Event | New Status |
|----------------|-------|------------|
| (new) | Registration, all tags auto-approved | `starting` |
| (new) | Registration, some tags need manual review | `pending_approval` |
| (new) | Registration, any forbidden tag | Registration rejected (403) |
| `pending_approval` | Admin approves tags | `starting` |
| `pending_approval` | Admin rejects tags | `offline` |
| `starting` | Health check passes | `ready` |
| `ready` | Health check fails | `offline` |

Agents in `pending_approval` state cannot be called — the permission middleware returns HTTP 503 Unavailable.

### 3. Access Policies (Policy Engine)

Access policies define tag-based authorization rules for cross-agent calls. They support function-level allow/deny lists and parameter constraints.

```yaml
# agentfield.yaml
features:
  did:
    authorization:
      access_policies:
        - name: finance_to_billing
          caller_tags: ["finance"]
          target_tags: ["billing"]
          allow_functions: ["charge_*", "refund_*", "get_*"]
          deny_functions: ["delete_*", "admin_*"]
          constraints:
            amount:
              operator: "<="
              value: 10000
          action: allow
          priority: 10

        - name: support_readonly
          caller_tags: ["support"]
          target_tags: ["customer-data"]
          allow_functions: ["get_*", "query_*"]
          action: allow
          priority: 5
```

**Policy Fields:**

| Field | Description |
|-------|-------------|
| `name` | Unique policy name |
| `caller_tags` | Tags the calling agent must have (empty = any) |
| `target_tags` | Tags the target agent must have (empty = any) |
| `allow_functions` | Whitelist of callable functions (supports wildcards) |
| `deny_functions` | Blacklist of functions (checked first, overrides allow) |
| `constraints` | Parameter-level constraints (e.g., `amount <= 10000`) |
| `action` | `"allow"` or `"deny"` — the decision when all conditions match |
| `priority` | Higher priority policies evaluated first |
| `enabled` | Toggle without deletion |

**Evaluation Algorithm (first-match-wins):**

1. Policies sorted by `priority DESC, id ASC` (deterministic ordering)
2. For each enabled policy:
   - Check caller tags match (empty policy tags = wildcard)
   - Check target tags match
   - Check deny functions — if matched, **immediately deny**
   - Check allow functions — if list exists but function not in it, skip policy
   - Evaluate constraints — missing parameters or violations cause deny (fail-closed)
   - All checks pass → return `Allowed = (action == "allow")`
3. No matching policy → return `Matched: false` (request allowed for backward compatibility)

**Constraint Operators:** `<=`, `>=`, `<`, `>`, `==`, `!=`
- Numeric comparison for numeric values
- String comparison (`==`/`!=`) as fallback
- Missing parameters in input → deny (fail-closed)

**Policy Evaluation Result:**

```go
type PolicyEvaluationResult struct {
    Allowed    bool   // Whether access is granted
    Matched    bool   // Whether any policy matched
    PolicyName string // Which policy matched
    PolicyID   int64
    Reason     string // Human-readable explanation
}
```

### 4. Verifiable Credentials (VCs)

#### AgentTagVC (Per-Agent)

Issued when admin approves an agent's tags. Certifies which tags an agent is authorized to hold. Signed with Ed25519 by the control plane issuer DID.

```json
{
  "@context": ["https://www.w3.org/2018/credentials/v1"],
  "type": ["VerifiableCredential", "AgentTagCredential"],
  "id": "urn:agentfield:agent-tag-vc:550e8400-e29b-41d4-a716-446655440000",
  "issuer": "did:web:localhost%3A8080:agents:control-plane",
  "issuanceDate": "2026-02-08T10:30:00Z",
  "credentialSubject": {
    "id": "did:web:localhost%3A8080:agents:finance-bot",
    "agent_id": "finance-bot",
    "permissions": {
      "tags": ["finance", "payment"],
      "allowed_callees": ["*"]
    },
    "approved_by": "admin",
    "approved_at": "2026-02-08T10:30:00Z"
  },
  "proof": {
    "type": "Ed25519Signature2020",
    "created": "2026-02-08T10:30:00Z",
    "verificationMethod": "did:web:localhost%3A8080:agents:control-plane#key-1",
    "proofPurpose": "assertionMethod",
    "proofValue": "s6mNf...XkMg=="
  }
}
```

**Key fields:**
- `credentialSubject.id` — Agent's DID (cryptographic identity)
- `credentialSubject.permissions.tags` — Approved tags (what the agent is authorized to claim)
- `credentialSubject.permissions.allowed_callees` — `["*"]` means policy engine decides
- `proof` — Ed25519 signature from control plane issuer; falls back to `UnsignedAuditRecord` if issuer DID unavailable

**Verification at call time (`TagVCVerifier`):**
1. Load VC record from storage
2. Check revocation (`revoked_at` timestamp)
3. Check expiration (`expires_at` timestamp)
4. Parse VC document from JSON
5. Verify Ed25519 signature against issuer public key
6. Validate subject binding (VC agent ID matches requested agent)

If VC verification succeeds, the permission middleware uses **VC-verified tags** (cryptographic proof). If VC exists but verification fails (revoked/expired/invalid), the middleware uses **empty tags** (fail-closed security — no fallback to unverified tags).

> **Note:** The legacy `PermissionVC` (per caller-target pair) has been superseded by `AgentTagVC`. The tag-based model scales as O(n) agents + policy rules rather than O(n²) pair-wise approvals.

### 5. DID Methods

#### did:web (Primary)
- DID resolves to URL: `did:web:agentfield.example.com:agents:agent-a`
- Control plane hosts DID document at that URL
- **Real-time revocation** — return 404 or revoked status
- Verifiers fetch fresh public key on each verification
- Domain configurable via `features.did.authorization.domain`

#### did:key (Supported)
- DID derived from public key: `did:key:z6MkpTHR8VNs...`
- Self-contained, no external resolution
- **Cannot be revoked** — only time-based expiry

### 6. Decentralized Verification

Agents can verify authorization locally without hitting the control plane on every call. All three SDKs (Python, Go, TypeScript) implement a `LocalVerifier` that caches:

- **Access policies** — fetched from `GET /api/v1/policies`
- **Revocation list** — fetched from `GET /api/v1/revocations`
- **Admin public key** — fetched from `GET /api/v1/admin/public-key`

**Cache refresh:** Every 5 minutes (configurable via `refresh_interval`).

**Local verification steps:**
1. Check caller DID not in revocation list
2. Validate timestamp within window (default 300 seconds)
3. Verify Ed25519 signature using cached admin public key
4. Evaluate policies locally using cached rules
5. Check parameter constraints

**Opt-out per function:** Functions marked with `require_realtime_validation=True` bypass local verification and forward to the control plane.

```python
# Python SDK
@app.reasoner(require_realtime_validation=True)
async def high_security_operation(input_data):
    ...
```

```typescript
// TypeScript SDK
agent.registerReasoner("sensitive_op", handler, {
    requireRealtimeValidation: true,
});
```

**Fail-closed behavior:**
- No policies cached → deny access
- Tag VC signature verification fails → use empty tags (deny unless explicit allow-all policy)
- Control plane unreachable → use stale cache (controlled degradation)

---

## Trust Model

### What We Trust

| Entity | Trust Level | Rationale |
|--------|-------------|-----------|
| Control Plane | Full | Central authority, hosts DIDs, issues VCs, enforces policies |
| Admin | Full | Approves/rejects tags, defines access policies, manages revocations |
| Agent's Private Key | Cryptographic | Proves DID ownership via Ed25519 signatures |

### What We Don't Trust

| Entity | Protection Mechanism |
|--------|---------------------|
| Developers proposing tags | Tags are *proposed*, not active until admin approves |
| Agents spoofing DIDs | DID ownership proven via Ed25519 cryptographic signature |
| Forged AgentTagVCs | VC signature verified against control plane issuer's public key |
| Modified VCs | Any modification breaks Ed25519 signature → rejected |
| Expired approvals | Expiration checked on each call (both VC and approval) |
| Replay attacks | Timestamp window + in-memory signature cache with TTL |

### Two-Step Authorization

**Step 1: Tag Assignment (Admin Approval)**
- **Question:** Does this agent deserve these tags?
- **Who decides:** Admin (or auto-approval rules)
- **When:** At registration time
- **Result:** AgentTagVC with approved tags, signed by control plane

**Step 2: Function Call (Policy Evaluation)**
- **Question:** Can caller's tags call target's function with these parameters?
- **Who decides:** Policy engine (automated, based on admin-defined policies)
- **When:** Every function call
- **Result:** Allow or deny based on tag-matching policies + parameter constraints

Both steps must pass for access to work.

### Security Boundaries

```
┌─────────────────────────────────────────────────────────────────┐
│  SECURITY BOUNDARY: Control Plane                               │
│                                                                 │
│  - Issues DIDs (did:web)                                       │
│  - Hosts DID documents (enables revocation)                    │
│  - Evaluates tag approval rules                                │
│  - Issues signed AgentTagVCs (Ed25519)                         │
│  - Evaluates access policies at call time                      │
│  - Stores approval records (source of truth)                   │
│  - Publishes policies, revocations, public keys for caching    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Admin controls
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  TRUST BOUNDARY: Admin                                          │
│                                                                 │
│  - Configures tag approval rules (auto/manual/forbidden)       │
│  - Reviews and approves/rejects proposed tags                  │
│  - Defines access policies (caller_tags → target_tags)         │
│  - Can revoke tags and VCs at any time                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Credentials issued
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  AGENT BOUNDARY: Proposed Identity                              │
│                                                                 │
│  - Agents propose tags (not active until approved)             │
│  - Tags can be per-skill/per-reasoner                          │
│  - Agents cannot grant themselves access                       │
│  - Must request and wait for admin approval                    │
│  - Can cache policies for local verification                   │
│  - Can opt functions into realtime validation                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## API Contracts

### Agent Registration

```http
POST /api/v1/nodes/register
Content-Type: application/json

{
  "id": "finance-bot",
  "team_id": "default",
  "base_url": "http://localhost:9001",
  "reasoners": [
    {
      "id": "process_payment",
      "tags": ["finance", "pci-compliant"],
      "proposed_tags": ["finance", "pci-compliant"],
      "input_schema": {}
    }
  ],
  "skills": [
    {
      "id": "get_balance",
      "tags": ["finance"],
      "proposed_tags": ["finance"],
      "input_schema": {}
    }
  ]
}
```

**Response (all tags auto-approved):**
```json
{
  "success": true,
  "message": "Node registered",
  "node_id": "finance-bot"
}
```

**Response (some tags need manual review):**
```json
{
  "success": true,
  "message": "Node registered but awaiting tag approval",
  "node_id": "finance-bot",
  "status": "pending_approval",
  "proposed_tags": ["finance", "pci-compliant"],
  "pending_tags": ["finance"],
  "auto_approved_tags": ["pci-compliant"]
}
```

**Response (forbidden tag):**
```json
HTTP 403
{
  "error": "forbidden_tags",
  "message": "Registration rejected: tags [root] are forbidden",
  "forbidden_tags": ["root"]
}
```

**Registration flow:**
1. Parse registration, normalize tags (`proposed_tags` ↔ `tags` bidirectional sync)
2. Evaluate tag approval rules if service enabled
3. If any forbidden tags → reject with HTTP 403
4. If any manual tags → set lifecycle status to `pending_approval`
5. If all auto → set lifecycle status to `starting`, set approved_tags immediately
6. Create did:web document for the agent
7. Store agent in database

### Admin Endpoints — Tag Approval

#### List Pending Agents
```http
GET /api/v1/admin/agents/pending

Response 200:
{
  "agents": [
    {
      "agent_id": "finance-bot",
      "proposed_tags": ["finance", "reporting", "admin"],
      "approved_tags": [],
      "status": "pending_approval",
      "registered_at": "2026-02-08T12:00:00Z"
    }
  ],
  "total": 1
}
```

#### Approve Agent Tags (Agent-Level)
```http
POST /api/v1/admin/agents/:agent_id/approve-tags
Content-Type: application/json

{
  "approved_tags": ["finance", "reporting"],
  "reason": "Approved standard finance tags"
}

Response 200:
{
  "success": true,
  "message": "Agent tags approved",
  "agent_id": "finance-bot",
  "approved_tags": ["finance", "reporting"]
}
```

#### Approve Tags Per Skill/Reasoner
```http
POST /api/v1/admin/agents/:agent_id/approve-tags
Content-Type: application/json

{
  "approved_tags": ["finance"],
  "skill_tags": {
    "get_balance": ["finance"],
    "charge_customer": ["finance", "pci-compliant"]
  },
  "reasoner_tags": {
    "process_payment": ["finance", "pci-compliant"]
  },
  "reason": "Per-skill approval with different tag scopes"
}
```

When `skill_tags` or `reasoner_tags` are provided, approval is per-skill/per-reasoner. Each skill/reasoner gets only its specified tags. The agent-level `approved_tags` becomes the union of all per-skill/per-reasoner approved tags.

#### Reject Agent Tags
```http
POST /api/v1/admin/agents/:agent_id/reject-tags
Content-Type: application/json

{
  "reason": "Tags not appropriate for this deployment"
}

Response 200:
{
  "success": true,
  "message": "Agent tags rejected",
  "agent_id": "finance-bot"
}
```

Rejection moves the agent to `offline` status.

### Admin Endpoints — Access Policies

#### List Policies
```http
GET /api/v1/admin/policies

Response 200:
{
  "policies": [
    {
      "id": 1,
      "name": "finance_to_billing",
      "caller_tags": ["finance"],
      "target_tags": ["billing"],
      "allow_functions": ["charge_*", "refund_*"],
      "deny_functions": ["delete_*"],
      "constraints": {
        "amount": {"operator": "<=", "value": 10000}
      },
      "action": "allow",
      "priority": 10,
      "enabled": true,
      "description": "Finance agents can call billing functions"
    }
  ],
  "total": 1
}
```

#### Create Policy
```http
POST /api/v1/admin/policies
Content-Type: application/json

{
  "name": "finance_to_billing",
  "caller_tags": ["finance"],
  "target_tags": ["billing"],
  "allow_functions": ["charge_*", "refund_*"],
  "deny_functions": ["delete_*"],
  "constraints": {
    "amount": {"operator": "<=", "value": 10000}
  },
  "action": "allow",
  "priority": 10,
  "description": "Finance agents can call billing functions"
}

Response 201: (created AccessPolicy object)
```

#### Get / Update / Delete Policy
```http
GET    /api/v1/admin/policies/:id
PUT    /api/v1/admin/policies/:id
DELETE /api/v1/admin/policies/:id
```

### Agent-Facing Endpoints (Decentralized Verification)

#### Fetch Policies (for local caching)
```http
GET /api/v1/policies

Response 200:
{
  "policies": [...],
  "total": 5,
  "fetched_at": "2026-02-08T12:00:00Z"
}
```

#### Fetch Revocation List (for local caching)
```http
GET /api/v1/revocations

Response 200:
{
  "revoked_dids": [
    "did:web:example.com:agents:compromised-agent"
  ],
  "total": 1,
  "fetched_at": "2026-02-08T12:00:00Z"
}
```

#### Fetch Admin Public Key (for local VC verification)
```http
GET /api/v1/admin/public-key

Response 200:
{
  "issuer_did": "did:web:localhost%3A8080:agents:control-plane",
  "public_key_jwk": {
    "kty": "OKP",
    "crv": "Ed25519",
    "x": "base64url_encoded_32_byte_key"
  },
  "fetched_at": "2026-02-08T12:00:00Z"
}
```

### DID Resolution (did:web)

```http
GET /agents/:agent_id/did.json

Response 200 (active):
{
  "@context": ["https://www.w3.org/ns/did/v1"],
  "id": "did:web:example.com:agents:agent-a",
  "verificationMethod": [{
    "id": "did:web:example.com:agents:agent-a#key-1",
    "type": "JsonWebKey2020",
    "controller": "did:web:example.com:agents:agent-a",
    "publicKeyJwk": {
      "kty": "OKP",
      "crv": "Ed25519",
      "x": "..."
    }
  }],
  "authentication": ["did:web:example.com:agents:agent-a#key-1"]
}

Response 404 (revoked):
{
  "error": "did_revoked",
  "message": "This DID has been revoked"
}
```

---

## Configuration

### What Goes Where

| Data | Source | Purpose |
|------|--------|---------|
| **Tag approval rules** | Config file | Controls which tags need manual approval |
| **Access policies** | Config file + Database (Admin API) | Tag-based authorization rules |
| **Agent Tag VCs** | Database | Stores signed VCs certifying agent tags |
| **DID documents** | Database | Stores did:web documents for resolution |
| **Agent proposed_tags** | Agent registration | Tags the developer proposes (per skill/reasoner) |
| **Agent approved_tags** | Admin approval / auto | Tags granted after evaluation |

### Full Configuration Example

```yaml
# agentfield.yaml
features:
  did:
    authorization:
      # Enable/disable the authorization system
      enabled: true

      # Enable DID-based authentication on API routes
      did_auth_enabled: true

      # Domain for did:web identifiers
      domain: "agentfield.example.com"

      # Allowed time drift for DID signature timestamps (seconds)
      timestamp_window_seconds: 300

      # Default duration for permission approvals (hours)
      default_approval_duration_hours: 720  # 30 days

      # Separate token for admin operations (tag approval, policy management)
      admin_token: "admin-secret-token"

      # Token sent to agents during request forwarding
      # (agents with RequireOriginAuth validate this)
      internal_token: "internal-secret-token"

      # Tag approval rules
      tag_approval_rules:
        default_mode: auto  # "auto" | "manual" | "forbidden"
        rules:
          - tags: ["admin", "superuser"]
            approval: manual
            reason: "Admin-level tags require review"
          - tags: ["dangerous", "root"]
            approval: forbidden
            reason: "These tags are not allowed"
          - tags: ["internal", "beta"]
            approval: auto
            reason: "Safe tags, no special privileges"

      # Access policies (seeded at startup, also manageable via Admin API)
      access_policies:
        - name: finance_to_billing
          caller_tags: ["finance"]
          target_tags: ["billing"]
          allow_functions: ["charge_*", "refund_*", "get_*"]
          deny_functions: ["delete_*", "admin_*"]
          constraints:
            amount:
              operator: "<="
              value: 10000
          action: allow
          priority: 10
        - name: support_readonly
          caller_tags: ["support"]
          target_tags: ["customer-data"]
          allow_functions: ["get_*", "query_*"]
          action: allow
          priority: 5
```

### Environment Variables

```bash
# Enable authorization
AGENTFIELD_AUTHORIZATION_ENABLED=true

# Enable DID-based authentication on API routes
AGENTFIELD_AUTHORIZATION_DID_AUTH_ENABLED=true

# Domain for did:web identifiers
AGENTFIELD_AUTHORIZATION_DOMAIN=agentfield.example.com

# Separate token for admin operations
AGENTFIELD_AUTHORIZATION_ADMIN_TOKEN=admin-secret-token

# Token sent to agents during request forwarding
AGENTFIELD_AUTHORIZATION_INTERNAL_TOKEN=internal-secret-token
```

Environment variables take precedence over YAML config values.

---

## Database Schema

### did_documents
Stores DID documents for did:web resolution.

```sql
CREATE TABLE did_documents (
    did             TEXT PRIMARY KEY,
    agent_id        TEXT NOT NULL,
    did_document    JSONB NOT NULL,
    public_key_jwk  TEXT NOT NULL,
    revoked_at      TIMESTAMP WITH TIME ZONE,
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);
```

### access_policies
Stores tag-based authorization policies.

```sql
CREATE TABLE access_policies (
    id              BIGSERIAL PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    caller_tags     TEXT NOT NULL,          -- JSON array
    target_tags     TEXT NOT NULL,          -- JSON array
    allow_functions TEXT,                   -- JSON array
    deny_functions  TEXT,                   -- JSON array
    constraints     TEXT,                   -- JSON object {param: {operator, value}}
    action          TEXT NOT NULL DEFAULT 'allow',
    priority        INTEGER NOT NULL DEFAULT 0,
    enabled         BOOLEAN NOT NULL DEFAULT TRUE,
    description     TEXT,
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);
```

### agent_tag_vcs
Stores signed Agent Tag VCs issued upon tag approval.

```sql
CREATE TABLE agent_tag_vcs (
    id              BIGSERIAL PRIMARY KEY,
    agent_id        TEXT NOT NULL UNIQUE,
    agent_did       TEXT NOT NULL,
    vc_id           TEXT NOT NULL UNIQUE,
    vc_document     TEXT NOT NULL,          -- Full W3C VC JSON
    signature       TEXT,                   -- Ed25519 signature
    issued_at       TIMESTAMP WITH TIME ZONE NOT NULL,
    expires_at      TIMESTAMP WITH TIME ZONE,
    revoked_at      TIMESTAMP WITH TIME ZONE,
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);
```

---

## Middleware Architecture

### Request Processing Pipeline

```
Incoming Request
    │
    ▼
┌──────────────────────────┐
│  DID Auth Middleware       │
│  - Extract X-Caller-DID   │
│  - Verify Ed25519 sig     │
│  - Check timestamp window  │
│  - Replay protection       │
│  - Store verified DID      │
│    in request context      │
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│  Permission Middleware     │
│                            │
│  1. Target in pending_     │
│     approval? → 503        │
│                            │
│  2. LAYER 1: Tag Policy    │
│     - Load caller's        │
│       AgentTagVC            │
│     - Verify VC signature  │
│     - Get tags from VC     │
│     - EvaluateAccess()     │
│     - Policy matched?      │
│       YES → allow/deny     │
│                            │
│  3. No policy matched:     │
│     Allow (backward compat)│
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│  Execute Handler           │
│  - Forward to target agent │
│  - Include X-Caller-DID    │
│    and X-Target-DID headers│
└──────────────────────────┘
```

### DID Auth Headers

| Header | Description |
|--------|-------------|
| `X-Caller-DID` | Agent's DID (e.g., `did:web:example.com:agents:agent-a`) |
| `X-DID-Signature` | Base64-encoded Ed25519 signature |
| `X-DID-Timestamp` | ISO 8601 timestamp of signing |

**Signature payload:** `{timestamp}:{SHA256(request_body)}`

**Replay protection:** In-memory signature cache with TTL matching the timestamp window. SHA256 hash of decoded signature is tracked.

---

## Backward Compatibility

### Default Mode is Zero-Disruption

- `tag_approval_rules.default_mode: auto` — all tags auto-approved when no rules configured
- If no access policies are defined, the policy engine returns `Matched: false` and the request is allowed
- If authorization is disabled (`enabled: false`), all middleware is skipped
- `proposed_tags` ↔ `tags` bidirectional sync ensures old SDKs work seamlessly

### Policy Evaluation Behavior

The permission middleware evaluates tag-based access policies:
1. If a policy matches, it decides (allow or deny)
2. If no policy matches, the request is allowed (backward compatible for untagged agents)

### Migration Path

1. **Phase 1:** Deploy with `authorization.enabled: false` (default)
2. **Phase 2:** Enable authorization, configure tag approval rules
3. **Phase 3:** Define access policies for tag-based authorization
4. **Phase 4:** Monitor policy evaluation results, tune policies
5. **Phase 5:** Enable decentralized verification in SDKs for performance

### Authorization Rules

| Scenario | Behavior |
|----------|----------|
| Agent in pending_approval | 503 Unavailable |
| Policy match → allow | Access granted |
| Policy match → deny | 403 Forbidden |
| No policy match | Allowed (backward compatible) |
| Authorization disabled | All middleware skipped |

---

## Appendix: Glossary

| Term | Definition |
|------|------------|
| **DID** | Decentralized Identifier — globally unique identifier for agents |
| **did:key** | DID method where identifier is derived from public key |
| **did:web** | DID method where identifier resolves to a web URL |
| **VC** | Verifiable Credential — signed, tamper-evident credential |
| **AgentTagVC** | VC certifying an agent's approved tags (per-agent, issued on tag approval) |
| **Access Policy** | Tag-based rule controlling which agents can call which functions |
| **Tag Approval Rule** | Configuration controlling which tags require admin review |
| **Approval** | Admin decision granting tags to an agent |
| **Revocation** | Invalidating a DID, VC, or permission before expiration |
| **CanonicalAgentTags** | Normalized tag set: prefers approved_tags, excludes metadata |
| **LocalVerifier** | SDK component that caches policies/revocations for offline verification |

---

*End of Architecture Document*
