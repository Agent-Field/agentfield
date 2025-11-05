SHELL := /usr/bin/env bash

.PHONY: all install build test lint fmt tidy clean control-plane sdk-go sdk-python

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

