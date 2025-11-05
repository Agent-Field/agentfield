# DID Database Schema Migrations

This directory contains SQL migration files for the DID (Decentralized Identity) implementation in the Haxen platform. These migrations create the necessary database tables to support DID-based authentication, verifiable credentials, and workflow execution tracking.

## Migration Files

### 000_migration_runner.sql
Complete migration runner script that includes all tables, indexes, triggers, and views. This can be executed as a single script to set up the entire DID schema.

### 001_create_did_registry.sql
Creates the `did_registry` table for organization-level DID management:
- **Purpose**: Master DID registry for organizations
- **Key Fields**: organization_id, master_seed_encrypted, root_did, agent_nodes
- **Features**: Encrypted master seed storage, agent node mapping

### 002_create_agent_dids.sql
Creates the `agent_dids` table for agent node DID information:
- **Purpose**: Individual agent node DID records
- **Key Fields**: did, agent_node_id, public_key_jwk, status
- **Features**: JWK key storage, reasoner/skill mapping, status tracking

### 003_create_component_dids.sql
Creates the `component_dids` table for reasoner and skill DIDs:
- **Purpose**: Component-level DID records (reasoners and skills)
- **Key Fields**: did, component_type, function_name, capabilities/tags
- **Features**: Component type differentiation, exposure level control

### 004_create_execution_vcs.sql
Creates the `execution_vcs` table for execution verifiable credentials:
- **Purpose**: Individual execution VC records with cryptographic proofs
- **Key Fields**: vc_id, execution_id, vc_document, signature, input/output hashes
- **Features**: VC chaining, cryptographic verification, audit trail

### 005_create_workflow_vcs.sql
Creates the `workflow_vcs` table for workflow-level VC chains:
- **Purpose**: Workflow-level VC aggregation and tracking
- **Key Fields**: workflow_vc_id, component_vc_ids, total/completed steps
- **Features**: Workflow progress tracking, VC chain management

### 006_add_storage_uri_to_execution_vcs.sql
Adds storage metadata columns to `execution_vcs`:
- **Purpose**: Prepare execution VCs for external blob storage
- **Key Fields**: storage_uri, document_size_bytes
- **Features**: URI indexing for fast lookup, document size tracking

### 007_add_storage_uri_to_workflow_vcs.sql
Adds storage metadata columns to `workflow_vcs`:
- **Purpose**: Prepare workflow VCs for external blob storage
- **Key Fields**: storage_uri, document_size_bytes
- **Features**: URI indexing for fast lookup, document size tracking

## Database Schema Overview

```
did_registry (1) ──→ (N) agent_dids
                           │
                           └──→ (N) component_dids
                                     │
                                     ├──→ (N) execution_vcs (as issuer)
                                     ├──→ (N) execution_vcs (as target)
                                     └──→ (N) execution_vcs (as caller)

workflow_vcs ──→ (N) execution_vcs (via component_vc_ids JSON array)
```

## Key Features

### Security
- Encrypted master seed storage
- JWK (JSON Web Key) format for public keys
- Cryptographic signatures for all VCs
- Input/output hash verification

### Performance
- Comprehensive indexing strategy
- Composite indexes for common query patterns
- Optimized foreign key relationships

### Audit Trail
- Complete VC chain tracking
- Timestamp tracking with automatic updates
- Status progression monitoring
- Hierarchical DID relationships

### Data Integrity
- Foreign key constraints
- Check constraints for status values
- Automatic timestamp updates via triggers
- JSON validation for structured fields

## Views

### did_hierarchy_view
Provides a hierarchical view of the complete DID structure from organization to components.

### vc_audit_trail
Comprehensive audit trail view combining execution VCs with workflow context.

### workflow_vc_chain_view
Analysis view for workflow VC chains with completion metrics.

## Usage

### Running Migrations
Execute the migration files in order, or use the complete migration runner:

```sql
-- Option 1: Run complete migration
.read haxen/migrations/000_migration_runner.sql

-- Option 2: Run individual migrations
.read haxen/migrations/001_create_did_registry.sql
.read haxen/migrations/002_create_agent_dids.sql
.read haxen/migrations/003_create_component_dids.sql
.read haxen/migrations/004_create_execution_vcs.sql
.read haxen/migrations/005_create_workflow_vcs.sql
```

### Integration with Go Services
The existing DID services in `haxen/internal/services/` and types in `haxen/pkg/types/did_types.go` are designed to work with these database tables. The migration from file-based storage to database storage should be seamless.

## Migration from File Storage

The current DID implementation uses file-based storage in:
- `data/did_registries/`
- `data/keys/`
- `data/vcs/executions/`
- `data/vcs/workflows/`

After running these migrations, the DID services can be updated to use database storage instead of file storage, providing better performance, consistency, and querying capabilities.
