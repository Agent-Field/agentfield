# AgentField Test Infrastructure

This directory contains shared infrastructure for running functional tests across the AgentField platform. It provides a Docker Compose environment with the control plane and PostgreSQL database for testing SDKs in a realistic deployment scenario.

## Overview

The test infrastructure consists of:

- **PostgreSQL Database** (with pgvector extension) on port 5433
- **AgentField Control Plane** in PostgreSQL mode on port 8080
- **Helper Scripts** for managing the test environment
- **Environment Configuration** for test settings

## Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenRouter API key (for AI integration tests)

### Setup

1. **Create environment file:**
   ```bash
   cp .env.test.example .env.test
   ```

2. **Edit `.env.test` and add your OpenRouter API key:**
   ```env
   OPENROUTER_API_KEY=your-actual-api-key-here
   ```

3. **Start the test environment:**
   ```bash
   ./scripts/start-env.sh
   ```

4. **Wait for services to be healthy** (automatic with start-env.sh)

5. **Run tests:**
   ```bash
   # Python SDK tests
   cd ../sdk/python
   pytest -m functional -v

   # Go SDK tests
   cd ../sdk/go
   go test -tags functional -v
   ```

6. **Stop the environment when done:**
   ```bash
   ./scripts/stop-env.sh
   ```

## Scripts

All scripts are located in the `scripts/` directory:

### `start-env.sh`
Starts the Docker Compose test environment.

- Builds control plane container if needed
- Starts PostgreSQL and control plane services
- Waits for health checks to pass
- Loads environment variables from `.env.test` if present

**Usage:**
```bash
./scripts/start-env.sh
```

### `stop-env.sh`
Stops and cleans up the test environment.

- Stops all containers
- Removes containers, networks, and volumes
- Complete cleanup for fresh start

**Usage:**
```bash
./scripts/stop-env.sh
```

### `restart-env.sh`
Convenience script to stop and restart the environment.

**Usage:**
```bash
./scripts/restart-env.sh
```

### `wait-for-health.sh`
Waits for the control plane to become healthy.

- Polls the `/api/v1/health` endpoint
- Timeout of 60 seconds
- Used internally by `start-env.sh`

**Usage:**
```bash
./scripts/wait-for-health.sh
```

### `logs.sh`
View logs from running services.

**Usage:**
```bash
# Tail all logs
./scripts/logs.sh

# Tail specific service
./scripts/logs.sh control-plane

# View last N lines
./scripts/logs.sh --tail 100

# Follow logs for specific service
./scripts/logs.sh -f postgres
```

## Services

### PostgreSQL Database

- **Image:** `pgvector/pgvector:pg16`
- **Port:** 5433 (external) → 5432 (internal)
- **Database:** `agentfield_test`
- **User:** `agentfield_test`
- **Password:** `test_password`
- **Features:** pgvector extension for vector operations

**Connection string:**
```
postgresql://agentfield_test:test_password@localhost:5433/agentfield_test?sslmode=disable
```

### Control Plane

- **Built from:** `control-plane/` directory
- **Port:** 8080
- **Mode:** PostgreSQL storage mode
- **Health endpoint:** http://localhost:8080/api/v1/health
- **API base:** http://localhost:8080/api/v1

**Environment variables:**
- `AGENTFIELD_STORAGE_MODE=postgres`
- `AGENTFIELD_POSTGRES_URL` (points to postgres service)
- `LOG_LEVEL=debug`
- `GIN_MODE=debug`

## Environment Variables

The `.env.test` file (create from `.env.test.example`) supports:

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENROUTER_API_KEY` | OpenRouter API key for LLM tests | Yes (for AI tests) | - |
| `OPENROUTER_BASE_URL` | OpenRouter API base URL | No | `https://openrouter.ai/api/v1` |
| `OPENROUTER_MODEL` | Default model to use | No | `google/gemini-flash-1.5` |
| `AGENTFIELD_SERVER` | Control plane URL | No | `http://localhost:8080` |

## Using from SDK Tests

### Python SDK

The Python SDK's functional tests use a pytest fixture to manage the test environment:

```python
# tests/functional/conftest.py
@pytest.fixture(scope="session")
def test_environment():
    """Start test infrastructure."""
    # Calls test-infra/scripts/start-env.sh
    # Yields control plane URL
    # Cleans up on teardown
```

**Usage in tests:**
```python
@pytest.mark.functional
def test_something(test_environment):
    control_plane_url = test_environment
    # Your test code here
```

### Go SDK

The Go SDK provides test utilities in `internal/testutil/`:

```go
//go:build functional

package mytest_test

import (
    "testing"
    "github.com/Agent-Field/agentfield/sdk/go/internal/testutil"
)

func TestSomething(t *testing.T) {
    testutil.StartTestInfra(t)
    // Your test code here
}
```

The `StartTestInfra` function:
- Calls `test-infra/scripts/start-env.sh`
- Registers cleanup with `t.Cleanup()`
- Automatically stops environment when test completes

## Troubleshooting

### Services won't start

**Check if ports are available:**
```bash
# Port 8080 (control plane)
lsof -i :8080

# Port 5433 (postgres)
lsof -i :5433
```

If ports are in use, stop conflicting services or modify `docker-compose.yml` to use different ports.

### Health checks failing

**View control plane logs:**
```bash
./scripts/logs.sh control-plane
```

**Common issues:**
- PostgreSQL not ready yet (wait longer or check postgres logs)
- Database migrations failed (check control plane logs)
- Build errors (try rebuilding: `docker compose build --no-cache`)

### Tests can't connect to control plane

**Verify services are running:**
```bash
cd test-infra
docker compose ps
```

All services should show status "Up (healthy)".

**Test connectivity:**
```bash
curl http://localhost:8080/api/v1/health
```

Should return a 200 OK response.

### Database issues

**Reset database:**
```bash
./scripts/stop-env.sh  # Removes volumes
./scripts/start-env.sh # Fresh start
```

**Connect to database directly:**
```bash
docker compose exec postgres psql -U agentfield_test -d agentfield_test
```

### Docker build issues

**Clear Docker cache and rebuild:**
```bash
cd test-infra
docker compose down -v
docker compose build --no-cache
docker compose up -d
```

## CI/CD Integration

The test infrastructure is designed to work identically in CI and local development.

**GitHub Actions usage:**
```yaml
- name: Start test infrastructure
  run: ./test-infra/scripts/start-env.sh

- name: Run tests
  env:
    OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
  run: |
    cd sdk/python
    pytest -m functional -v

- name: Stop test infrastructure
  if: always()
  run: ./test-infra/scripts/stop-env.sh
```

See `.github/workflows/functional-tests.yml` for the complete CI workflow.

## Network Architecture

All services run in a shared Docker network: `agentfield_test_network`

```
┌─────────────────────────────────────────┐
│  Host Machine                           │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │  Docker Network                   │ │
│  │  (agentfield_test_network)        │ │
│  │                                   │ │
│  │  ┌─────────────────────────────┐ │ │
│  │  │  PostgreSQL                 │ │ │
│  │  │  Internal: 5432             │ │ │
│  │  │  External: 5433             │ │ │
│  │  └─────────────────────────────┘ │ │
│  │              ▲                    │ │
│  │              │                    │ │
│  │  ┌───────────┴───────────────┐   │ │
│  │  │  Control Plane            │   │ │
│  │  │  Port: 8080               │   │ │
│  │  └───────────────────────────┘   │ │
│  │              ▲                    │ │
│  └──────────────┼────────────────────┘ │
│                 │                       │
│       ┌─────────┴─────────┐            │
│       │  Test Agents      │            │
│       │  (SDK tests)      │            │
│       └───────────────────┘            │
└─────────────────────────────────────────┘
```

**Communication:**
- Test agents → Control plane: `http://localhost:8080`
- Control plane → PostgreSQL: `postgresql://postgres:5432` (internal DNS)
- External clients → PostgreSQL: `localhost:5433`

## Maintenance

### Updating Control Plane

Control plane is built from the local `control-plane/` directory. To update:

```bash
./scripts/restart-env.sh
```

This rebuilds the control plane image and restarts services.

### Updating PostgreSQL Version

Edit `docker-compose.yml` and change the postgres image version:

```yaml
postgres:
  image: pgvector/pgvector:pg17  # Updated version
```

Then restart:

```bash
./scripts/restart-env.sh
```

### Database Migrations

Control plane automatically runs migrations on startup. To manually run migrations:

```bash
# In control-plane directory
export AGENTFIELD_POSTGRES_URL="postgresql://agentfield_test:test_password@localhost:5433/agentfield_test?sslmode=disable"
goose -dir ./migrations postgres "$AGENTFIELD_POSTGRES_URL" up
```

## Cost Considerations

When running tests with OpenRouter:

- **Model:** `google/gemini-flash-1.5` (~$0.075 per 1M input tokens)
- **Typical test run:** ~5 LLM calls, ~5,000 tokens total
- **Cost per run:** ~$0.0004 (less than a penny)
- **Monthly CI cost (100 runs):** ~$0.04

To minimize costs:
- Use cheap models (Gemini Flash is recommended)
- Set reasonable timeouts
- Cache responses when possible
- Skip AI tests locally if developing non-AI features: `pytest -m "functional and not ai"`

## Related Documentation

- **Main testing guide:** `../docs/TESTING.md`
- **Python SDK tests:** `../sdk/python/tests/functional/README.md`
- **Go SDK tests:** `../sdk/go/README.md`
- **CI workflow:** `../.github/workflows/functional-tests.yml`

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. View logs: `./scripts/logs.sh`
3. Check GitHub issues: https://github.com/Agent-Field/agentfield/issues
4. Review the main testing documentation: `../docs/TESTING.md`
