# Testing Guide

This document provides comprehensive guidance on testing AgentField components, including unit tests, integration tests, and functional tests.

## Table of Contents

- [Overview](#overview)
- [Test Types](#test-types)
- [Running Tests](#running-tests)
- [Functional Tests](#functional-tests)
- [Writing Tests](#writing-tests)
- [CI/CD Integration](#cicd-integration)
- [Troubleshooting](#troubleshooting)

## Overview

AgentField uses a comprehensive testing strategy across three main components:

- **Control Plane** (Go): Unit and integration tests
- **Python SDK**: Unit, integration, and functional tests
- **Go SDK**: Unit and functional tests

### Test Philosophy

1. **Fast feedback**: Unit tests run quickly without external dependencies
2. **Confidence**: Functional tests validate end-to-end behavior in realistic environments
3. **Maintainability**: Tests are clear, focused, and easy to update
4. **CI/CD ready**: All tests run automatically in CI pipelines

## Test Types

### Unit Tests

**Purpose**: Test individual components in isolation

**Characteristics**:
- No external dependencies (databases, networks, LLMs)
- Fast execution (milliseconds)
- Use mocks and fakes
- Run by default

**Location**:
- Control Plane: `control-plane/internal/*/`
- Python SDK: `sdk/python/tests/` (marked with `@pytest.mark.unit`)
- Go SDK: `sdk/go/*/`

### Integration Tests

**Purpose**: Test component interactions within the same process

**Characteristics**:
- May use in-memory databases or local file systems
- Medium execution time (seconds)
- Limited external dependencies

**Location**:
- Python SDK: `sdk/python/tests/integration/`
- Marked with `@pytest.mark.integration`

### Functional Tests

**Purpose**: Test end-to-end behavior against a real control plane

**Characteristics**:
- Requires Docker environment (PostgreSQL + control plane)
- Realistic deployment scenario
- May make real LLM calls (OpenRouter)
- Slower execution (minutes)

**Location**:
- Shared Infrastructure: `test-infra/`
- Python SDK: `sdk/python/tests/functional/`
- Go SDK: `sdk/go/functional_test.go`

## Running Tests

### Control Plane Tests

```bash
cd control-plane

# Run all tests
go test ./...

# Run with coverage
go test -cover ./...

# Run specific package
go test ./internal/handlers

# Run with verbose output
go test -v ./...
```

### Python SDK Tests

```bash
cd sdk/python

# Install dev dependencies
pip install -e .[dev]

# Run all tests (excludes functional and MCP by default)
pytest

# Run only unit tests
pytest -m unit

# Run with coverage
pytest --cov=agentfield --cov-report=term-missing

# Run specific test file
pytest tests/test_client.py

# Run specific test function
pytest tests/test_client.py::test_register_node -v
```

### Go SDK Tests

```bash
cd sdk/go

# Run all tests
go test ./...

# Run with verbose output
go test -v ./...

# Run specific package
go test ./agent

# Run with coverage
go test -cover ./...
```

## Functional Tests

Functional tests validate end-to-end behavior against a real control plane and PostgreSQL database running in Docker.

### Prerequisites

1. **Docker and Docker Compose** installed
2. **OpenRouter API key** (for AI integration tests):
   ```bash
   # Create environment file
   cp test-infra/.env.test.example test-infra/.env.test

   # Edit and add your API key
   vim test-infra/.env.test
   ```

### Running Functional Tests

#### Option 1: Run All Functional Tests

```bash
# From repository root
./scripts/run-functional-tests.sh
```

This script:
1. Starts test infrastructure
2. Runs Python functional tests
3. Runs Go functional tests
4. Stops infrastructure
5. Reports results

#### Option 2: Run Python Functional Tests

```bash
# Start infrastructure
./test-infra/scripts/start-env.sh

# Run tests
cd sdk/python
export OPENROUTER_API_KEY="your-key"
pytest -m functional -v

# Stop infrastructure
./test-infra/scripts/stop-env.sh
```

#### Option 3: Run Go Functional Tests

```bash
# Start infrastructure
./test-infra/scripts/start-env.sh

# Run tests
cd sdk/go
export OPENROUTER_API_KEY="your-key"
go test -tags functional -v

# Stop infrastructure
./test-infra/scripts/stop-env.sh
```

### Test Infrastructure

The test infrastructure (`test-infra/`) provides a Docker Compose environment with:

- **PostgreSQL** (port 5433): Database with pgvector extension
- **Control Plane** (port 8080): AgentField server in PostgreSQL mode

**Helper scripts**:
- `start-env.sh`: Start test environment
- `stop-env.sh`: Stop and clean up
- `restart-env.sh`: Restart environment
- `wait-for-health.sh`: Wait for services to be ready
- `logs.sh`: View Docker logs

See [test-infra/README.md](../test-infra/README.md) for detailed documentation.

### Functional Test Structure

#### Python Tests

Located in `sdk/python/tests/functional/`:

- `test_agent_registration.py`: Agent registration and health checks
- `test_agent_execution.py`: Reasoner execution through control plane
- `test_agent_to_agent_call.py`: Agent-to-agent communication
- `test_ai_integration.py`: LLM calls via OpenRouter
- `test_memory_operations.py`: Memory operations (global, agent, session scopes)

**Fixtures** (in `conftest.py`):
- `test_environment`: Starts/stops Docker infrastructure
- `control_plane_url`: Control plane URL
- `control_plane_client`: HTTP client for control plane
- `openrouter_config`: OpenRouter configuration from environment
- `agent_port_allocator`: Allocates unique ports for test agents

#### Go Tests

Located in `sdk/go/functional_test.go`:

- `TestFunctionalRegistration`: Agent registration
- `TestFunctionalExecution`: Skill execution
- `TestFunctionalAgentCall`: Agent-to-agent calls
- `TestFunctionalAIIntegration`: LLM integration
- `TestFunctionalMemory`: Memory operations

**Test utilities** (`internal/testutil/testenv.go`):
- `StartTestInfra`: Start Docker environment
- `WaitForHealth`: Wait for control plane
- `GetOpenRouterConfig`: Get OpenRouter config from env
- `AllocatePort`: Allocate unique ports

### Skipping Tests

#### Python

```bash
# Skip functional tests (default)
pytest

# Skip AI tests when developing non-AI features
pytest -m "functional and not ai"

# Run only registration tests
pytest tests/functional/test_agent_registration.py
```

#### Go

```bash
# Skip functional tests (they require -tags functional)
go test ./...

# Run only functional tests
go test -tags functional -v
```

## Writing Tests

### Writing Unit Tests

#### Python Example

```python
import pytest
from agentfield import Agent

@pytest.mark.unit
def test_agent_creation():
    """Test that agents can be created with valid config."""
    agent = Agent(node_id="test-agent")
    assert agent.node_id == "test-agent"
```

#### Go Example

```go
func TestAgentCreation(t *testing.T) {
    agent, err := agentfield.New(agentfield.Config{
        NodeID: "test-agent",
    })
    assert.NoError(t, err)
    assert.Equal(t, "test-agent", agent.NodeID)
}
```

### Writing Functional Tests

#### Python Example

```python
import pytest

@pytest.mark.functional
def test_my_feature(control_plane_url, agent_port_allocator):
    """Test my feature against real control plane."""
    agent_port = agent_port_allocator()

    # Create agent
    agent = Agent(
        node_id="test-my-feature",
        agentfield_config=AgentFieldConfig(server_url=control_plane_url),
    )

    # Register reasoner
    @agent.reasoner()
    async def my_reasoner():
        return {"result": "success"}

    # Start agent server
    # ... test implementation ...
```

#### Go Example

```go
//go:build functional

func TestMyFeature(t *testing.T) {
    controlPlaneURL := testutil.StartTestInfra(t)

    agent, err := agentfield.New(agentfield.Config{
        NodeID: "test-my-feature",
        AgentFieldURL: controlPlaneURL,
    })
    require.NoError(t, err)

    // Register skill and test
    // ... test implementation ...
}
```

### Test Naming Conventions

**Python**:
- Files: `test_<module>.py`
- Classes: `Test<Feature>`
- Functions: `test_<specific_behavior>`

**Go**:
- Files: `<module>_test.go`
- Functions: `Test<Feature>`
- Subtests: `t.Run("specific behavior", func(t *testing.T) {...})`

### Best Practices

1. **One assertion per test**: Tests should verify a single behavior
2. **Clear test names**: Test names should describe what they verify
3. **Arrange-Act-Assert**: Structure tests in three phases
4. **Clean up resources**: Use fixtures/cleanup to prevent test pollution
5. **Avoid test interdependence**: Tests should run independently
6. **Fast unit tests**: Unit tests should run in < 100ms
7. **Realistic functional tests**: Functional tests should simulate real usage

## CI/CD Integration

### GitHub Actions Workflows

AgentField uses separate workflows for different test types:

#### Unit Tests (Fast, Always Run)

- **Control Plane**: `.github/workflows/control-plane.yml`
- **Python SDK**: `.github/workflows/sdk-python.yml`
- **Go SDK**: `.github/workflows/sdk-go.yml`

These run on every PR and push to main/develop.

#### Functional Tests (Slower, Selective)

- **Workflow**: `.github/workflows/functional-tests.yml`
- **Triggers**: Changes to `sdk/`, `control-plane/`, or `test-infra/`
- **Duration**: ~15-20 minutes
- **Environment**: Full Docker Compose setup

**Required secrets**:
- `OPENROUTER_API_KEY`: OpenRouter API key for AI tests

### Running Tests Locally Like CI

```bash
# Python unit tests (like CI)
cd sdk/python
pip install -e .[dev]
pytest --strict-markers --strict-config

# Go unit tests (like CI)
cd sdk/go
go test ./...

# Control plane tests (like CI)
cd control-plane
go test ./...
golangci-lint run

# Functional tests (like CI)
export OPENROUTER_API_KEY="your-key"
./scripts/run-functional-tests.sh
```

## Troubleshooting

### Common Issues

#### 1. Functional tests fail to start

**Symptom**: "Failed to start test infrastructure"

**Solutions**:
```bash
# Check if ports are available
lsof -i :8080  # Control plane
lsof -i :5433  # PostgreSQL

# Check Docker is running
docker ps

# Rebuild containers
cd test-infra
docker compose down -v
docker compose build --no-cache
```

#### 2. Tests hang or timeout

**Symptom**: Tests don't complete within timeout

**Solutions**:
```bash
# Increase pytest timeout
pytest -m functional --timeout=300

# Increase Go test timeout
go test -tags functional -timeout 20m

# Check Docker logs
./test-infra/scripts/logs.sh
```

#### 3. Port conflicts

**Symptom**: "Address already in use"

**Solutions**:
```bash
# Find process using port
lsof -i :8080

# Kill process
kill -9 <PID>

# Or change port in docker-compose.yml
```

#### 4. OpenRouter API errors

**Symptom**: AI tests fail with authentication errors

**Solutions**:
```bash
# Verify API key is set
echo $OPENROUTER_API_KEY

# Check .env.test file
cat test-infra/.env.test

# Test API key manually
curl -H "Authorization: Bearer $OPENROUTER_API_KEY" \
     https://openrouter.ai/api/v1/models
```

#### 5. Memory leaks in functional tests

**Symptom**: Tests fail after many runs

**Solutions**:
```bash
# Clean up Docker resources
docker system prune -af --volumes

# Restart Docker
# (OS-specific)

# Restart test environment
./test-infra/scripts/restart-env.sh
```

### Getting Help

1. **Check logs**:
   ```bash
   ./test-infra/scripts/logs.sh
   ```

2. **View test output with verbose mode**:
   ```bash
   pytest -m functional -vv --tb=long
   go test -tags functional -v
   ```

3. **Check GitHub Issues**: [agentfield/issues](https://github.com/Agent-Field/agentfield/issues)

4. **Review test infrastructure docs**: [test-infra/README.md](../test-infra/README.md)

## Coverage Reports

### Python Coverage

```bash
cd sdk/python

# Generate coverage report
pytest --cov=agentfield --cov-report=html

# Open report in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Go Coverage

```bash
cd sdk/go

# Generate coverage profile
go test -coverprofile=coverage.out ./...

# View coverage report
go tool cover -html=coverage.out
```

### Control Plane Coverage

```bash
cd control-plane

# Generate coverage
go test -coverprofile=coverage.out ./...

# View HTML report
go tool cover -html=coverage.out
```

## Performance Testing

While not part of the regular test suite, you can benchmark critical paths:

### Python

```python
import pytest

@pytest.mark.benchmark
def test_agent_call_performance(benchmark):
    result = benchmark(agent.call, "target.reasoner")
    assert result is not None
```

Run with: `pytest --benchmark-only`

### Go

```go
func BenchmarkAgentCall(b *testing.B) {
    agent := createTestAgent()
    b.ResetTimer()

    for i := 0; i < b.N; i++ {
        agent.Call(ctx, "target.skill", input)
    }
}
```

Run with: `go test -bench=. -benchmem`

## Continuous Improvement

### Test Metrics to Track

- Test execution time
- Test flakiness rate
- Code coverage percentage
- Number of tests per component

### Adding New Tests

When adding features:

1. Write unit tests first (TDD)
2. Add integration tests for component interactions
3. Add functional tests for user-facing features
4. Update this guide if introducing new patterns
5. Ensure tests pass in CI before merging

### Reviewing Tests

When reviewing PRs:

- Are there tests for new functionality?
- Do tests follow naming conventions?
- Are tests independent and deterministic?
- Do tests run quickly (unit tests)?
- Is test coverage adequate?

## Related Documentation

- [Test Infrastructure README](../test-infra/README.md)
- [CLAUDE.md](../CLAUDE.md) - Development guide
- [README.md](../README.md) - Project overview
