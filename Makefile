SHELL := /usr/bin/env bash

.PHONY: all install build test lint fmt tidy clean control-plane sdk-go sdk-python
.PHONY: test-functional test-functional-local test-functional-postgres test-functional-cleanup test-functional-ci log-demo-up log-demo-down log-demo-native-up log-demo-native-down

# Local UI stack: control plane + Python/Go/TS agents emitting NDJSON process logs (see tests/functional/docker/docker-compose.log-demo.yml).
log-demo-up:
	docker compose -f tests/functional/docker/docker-compose.log-demo.yml up --build -d

log-demo-down:
	docker compose -f tests/functional/docker/docker-compose.log-demo.yml down

# Same stack on the host when Docker Desktop is not running (writable paths under /tmp/agentfield-log-demo).
log-demo-native-up:
	./scripts/run-log-demo-native.sh

log-demo-native-down:
	./scripts/stop-log-demo-native.sh

all: build

install:
	./scripts/install-dev-deps.sh

build: control-plane sdk-go sdk-python

control-plane:
	( cd control-plane && go build ./... )

sdk-go:
	( cd sdk/go && go build ./... )

sdk-python:
	( cd sdk/python && pip install -e . >/dev/null )

test:
	./scripts/test-all.sh

lint:
	( cd control-plane && golangci-lint run || true )
	( cd sdk/go && golangci-lint run || true )
	( cd sdk/python && ruff check || true )

fmt:
	( cd control-plane && gofmt -w $$(go list -f '{{.Dir}}' ./...) )
	( cd sdk/go && gofmt -w $$(go list -f '{{.Dir}}' ./...) )
	( cd sdk/python && ruff format . )

tidy:
	( cd control-plane && go mod tidy )
	( cd sdk/go && go mod tidy )

clean:
	rm -rf control-plane/bin control-plane/dist
	find . -type d -name "__pycache__" -exec rm -rf {} +

# ============================================================================
# Functional Testing with Docker
# ============================================================================

test-functional: test-functional-local test-functional-postgres
	@echo "✅ All functional tests completed"

test-functional-local:
	@echo "🧪 Running functional tests with SQLite storage..."
	@if [ -z "$$OPENROUTER_API_KEY" ]; then \
		echo "❌ Error: OPENROUTER_API_KEY environment variable is not set"; \
		echo "   Please set it with: export OPENROUTER_API_KEY=your-key"; \
		echo "   Or use: make test-functional-local OPENROUTER_API_KEY=your-key"; \
		exit 1; \
	fi
	mkdir -p test-reports tests/functional/logs
	chmod -R 777 tests/functional/logs || true
	cd tests/functional && \
		docker compose -f docker/docker-compose.local.yml up --build --abort-on-container-exit --exit-code-from test-runner
	@if [ -f tests/functional/logs/functional-tests.log ]; then \
		cp tests/functional/logs/functional-tests.log test-reports/functional-tests-local.log; \
	fi
	@docker cp agentfield-test-runner-local:/reports/junit-local.xml test-reports/ 2>/dev/null || echo "⚠️  No JUnit report found in container"
	$(MAKE) test-functional-cleanup-local

test-functional-postgres:
	@echo "🧪 Running functional tests with PostgreSQL storage..."
	@if [ -z "$$OPENROUTER_API_KEY" ]; then \
		echo "❌ Error: OPENROUTER_API_KEY environment variable is not set"; \
		echo "   Please set it with: export OPENROUTER_API_KEY=your-key"; \
		echo "   Or use: make test-functional-postgres OPENROUTER_API_KEY=your-key"; \
		exit 1; \
	fi
	mkdir -p test-reports tests/functional/logs
	chmod -R 777 tests/functional/logs || true
	cd tests/functional && \
		docker compose -f docker/docker-compose.postgres.yml up --build --abort-on-container-exit --exit-code-from test-runner
	@if [ -f tests/functional/logs/functional-tests.log ]; then \
		cp tests/functional/logs/functional-tests.log test-reports/functional-tests-postgres.log; \
	fi
	@docker cp agentfield-test-runner-postgres:/reports/junit-postgres.xml test-reports/ 2>/dev/null || echo "⚠️  No JUnit report found in container"
	$(MAKE) test-functional-cleanup-postgres

test-functional-cleanup: test-functional-cleanup-local test-functional-cleanup-postgres

test-functional-cleanup-local:
	@echo "🧹 Cleaning up local test environment..."
	cd tests/functional && docker compose -f docker/docker-compose.local.yml down -v 2>/dev/null || true

test-functional-cleanup-postgres:
	@echo "🧹 Cleaning up postgres test environment..."
	cd tests/functional && docker compose -f docker/docker-compose.postgres.yml down -v 2>/dev/null || true

test-functional-ci:
	@echo "🧪 Running functional tests in CI mode..."
	@if [ -z "$$OPENROUTER_API_KEY" ]; then \
		echo "❌ Error: OPENROUTER_API_KEY environment variable is not set"; \
		exit 1; \
	fi
	@echo "Running tests with both storage modes..."
	$(MAKE) test-functional-local || ($(MAKE) test-functional-cleanup-local && exit 1)
	$(MAKE) test-functional-postgres || ($(MAKE) test-functional-cleanup-postgres && exit 1)
	@echo "✅ CI functional tests completed successfully"
