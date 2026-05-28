# Verifiable Credentials Example

This example demonstrates how to use **DID (Decentralized Identifiers)** and **Verifiable Credentials (VCs)** in AgentField to create cryptographically verifiable audit trails for agent executions.

## What are Verifiable Credentials?

Verifiable Credentials provide:
- **Tamper Detection**: Input/output hashes detect any modifications
- **Accountability**: DIDs link executions to agents and callers
- **Auditability**: Independent verification for compliance
- **Non-repudiation**: Cryptographic signatures prove execution occurred

## Prerequisites

1. **Control Plane Running** with DID enabled:
   ```yaml
   # agentfield.yaml
   features:
     did:
       enabled: true
       vc_requirements:
         persist_execution_vc: true
   ```

2. **Keystore Directory** exists on control plane:
   ```bash
   mkdir -p ./data/keys
   ```

## Quick Start

```bash
# From the ts-node-examples directory
cd agentfield/examples/ts-node-examples

# Install dependencies
pnpm install

# Start the VC demo agent
pnpm dev:vc
```

## Available Reasoners

### 1. `vc_process` - Basic Processing with VC

Simple text processing that demonstrates the fundamental VC flow.

```bash
curl -X POST http://localhost:8080/api/v1/execute/vc-demo.vc_process \
  -H "Content-Type: application/json" \
  -d '{"input": {"text": "Hello, Verifiable World!"}}'
```

**Response:**
```json
{
  "processed": "HELLO, VERIFIABLE WORLD!",
  "wordCount": 3,
  "timestamp": "2025-01-15T10:30:00.000Z",
  "vcGenerated": true,
  "vcId": "vc-abc123..."
}
```

### 2. `vc_analyze` - AI Analysis with VC Audit Trail

AI-powered text analysis with VC accountability for AI decisions.

```bash
curl -X POST http://localhost:8080/api/v1/execute/vc-demo.vc_analyze \
  -H "Content-Type: application/json" \
  -d '{"input": {"text": "I absolutely love this new product! Best purchase ever.", "analyzeTopics": true}}'
```

**Response:**
```json
{
  "sentiment": "positive",
  "confidence": 0.95,
  "topics": ["product", "purchase", "satisfaction"],
  "summary": "Highly positive review expressing strong satisfaction with a purchase.",
  "vcGenerated": true,
  "vcId": "vc-def456..."
}
```

### 3. `vc_transform` - Data Transformation with Integrity Proof

Data transformation with cryptographic integrity verification.

```bash
curl -X POST http://localhost:8080/api/v1/execute/vc-demo.vc_transform \
  -H "Content-Type: application/json" \
  -d '{"input": {"data": {"name": "  John Doe  ", "items": ["banana", "apple", "cherry"]}, "operations": ["trim", "sort"]}}'
```

**Response:**
```json
{
  "original": {"name": "  John Doe  ", "items": ["banana", "apple", "cherry"]},
  "transformed": {"name": "John Doe", "items": ["apple", "banana", "cherry"]},
  "operationsApplied": ["trim", "sort"],
  "vcGenerated": true,
  "vcId": "vc-ghi789...",
  "integrityProof": {
    "inputHash": "sha256:abc...",
    "outputHash": "sha256:def...",
    "verifiable": true
  }
}
```

### 4. `vc_chain` - Multi-Step Workflow with Chained VCs

Complex workflow demonstrating VC chaining across multiple steps.

```bash
curl -X POST http://localhost:8080/api/v1/execute/vc-demo.vc_chain \
  -H "Content-Type: application/json" \
  -d '{"input": {"text": "Process this data", "steps": ["validate", "process", "enrich", "finalize"]}}'
```

**Response:**
```json
{
  "input": "Process this data",
  "stepsExecuted": [
    {"step": "validate", "success": true, "vcId": "vc-1..."},
    {"step": "process", "success": true, "vcId": "vc-2..."},
    {"step": "enrich", "success": true, "vcId": "vc-3..."},
    {"step": "finalize", "success": true, "vcId": "vc-4..."}
  ],
  "vcChain": ["vc-1...", "vc-2...", "vc-3...", "vc-4...", "vc-workflow..."],
  "workflowVcGenerated": true
}
```

## Verifying Credentials in the UI

1. Open AgentField UI: `http://localhost:8080`
2. Navigate to **Workflows**
3. Find your workflow (filter by "vc-demo" agent)
4. Look for the **VC badge**:
   - ✓ Green checkmark = VCs generated successfully
   - ✗ Red X = No VCs (check configuration)

5. Click **Verify** to run comprehensive verification:
   - Signature validation
   - Hash integrity checks
   - DID authenticity
   - Compliance verification

## Exporting VCs for External Audit

### Export Workflow Compliance Report

```bash
# Get workflow ID from UI or API
WORKFLOW_ID=<your-workflow-id>

# Export as JSON (includes full VC chain + DID resolution bundle)
curl "http://localhost:8080/api/ui/v1/workflows/$WORKFLOW_ID/vc-chain" | jq > audit-report.json
```

### Export All VCs

```bash
curl "http://localhost:8080/api/ui/v1/did/export/vcs" | jq
```

## How VCs Work in This Example

```
┌─────────────────┐
│   Agent SDK     │
│                 │
│  ctx.did.       │
│  generateCred() │
└────────┬────────┘
         │ POST /api/v1/execution/vc
         ▼
┌─────────────────┐
│ Control Plane   │
│                 │
│  VCService.     │
│  GenerateVC()   │
│                 │
│  • Hash I/O     │
│  • Sign w/ DID  │
│  • Store in DB  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Database      │
│                 │
│  execution_vcs  │
│  workflow_vcs   │
│  did_registry   │
└─────────────────┘
```

## VC Document Structure

Each generated VC follows W3C Verifiable Credentials format:

```json
{
  "@context": [
    "https://www.w3.org/2018/credentials/v1",
    "https://agentfield.ai/credentials/v1"
  ],
  "type": ["VerifiableCredential", "AgentExecutionCredential"],
  "id": "urn:uuid:vc-abc123",
  "issuer": "did:key:z6Mk...",
  "issuanceDate": "2025-01-15T10:30:00Z",
  "credentialSubject": {
    "execution": {
      "id": "exec-123",
      "workflowId": "wf-456",
      "timestamp": "2025-01-15T10:30:00Z",
      "status": "succeeded",
      "durationMs": 150
    },
    "caller": {
      "did": "did:key:z6Mj...",
      "type": "agent"
    },
    "target": {
      "did": "did:key:z6Mk...",
      "functionName": "vc_process"
    },
    "data": {
      "inputHash": "sha256:abc...",
      "outputHash": "sha256:def..."
    }
  },
  "proof": {
    "type": "Ed25519Signature2020",
    "created": "2025-01-15T10:30:00Z",
    "verificationMethod": "did:key:z6Mk...#key-1",
    "proofPurpose": "assertionMethod",
    "proofValue": "z3FXQqSN..."
  }
}
```

## Troubleshooting

### VCs Not Generated (X Button in UI)

1. **Check control plane logs:**
   ```bash
   docker-compose logs af-server | grep -i "did\|vc"
   ```

2. **Verify DID is enabled:**
   ```bash
   curl http://localhost:8080/api/ui/v1/config | jq '.features.did'
   ```

3. **Check keystore:**
   ```bash
   ls -la ./data/keys
   ```

### VC Generated But Verification Fails

1. **Check DID resolution:**
   ```bash
   curl "http://localhost:8080/api/ui/v1/did/did:key:z6Mk.../resolution-bundle"
   ```

2. **Verify signature algorithm matches:**
   Both control plane and SDK should use Ed25519 (default).

## Next Steps

- Review the [Plan.md](../../../Plan.md) for comprehensive testing guide
- Check [SDK DID documentation](../../sdk/typescript/src/did/)
- Explore [control plane VC service](../../control-plane/internal/services/vc_service.go)
