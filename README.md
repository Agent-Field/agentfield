# Haxen Monorepo

Haxen is an open-source platform for building, deploying, and operating production-grade AI agents. This repository brings together the control plane, language-specific SDKs, deployment assets, and documentation required to run Haxen in your own environment.

## Repository Layout

| Path | Description |
| --- | --- |
| `control-plane/` | Go control plane providing orchestration, workflows, permissions, and REST/gRPC APIs. |
| `sdk/go/` | Go SDK for building agents and integrating with the control plane. |
| `sdk/python/` | Python SDK for building agents and integrating with the control plane. |
| `deployments/docker/` | Container images and Compose definitions for local development and PoC deployments. |
| `docs/` | Architecture, contribution, development, and security documentation. |
| `scripts/` | Helper scripts for installing dependencies, building, and testing the entire monorepo. |
| `.github/` | GitHub Actions workflows and community health files. |

## Quick Start

```bash
git clone https://github.com/your-org/haxen.git
cd haxen
./scripts/install.sh      # install Go, Python, and JS dependencies
./scripts/build-all.sh    # build control plane and SDKs
./scripts/test-all.sh     # run repository-wide test suites
```

## Development Workflow

- The control plane is a Go application with a web UI located in `control-plane/web`.
- The Go SDK is maintained as a standalone module under `sdk/go`.
- The Python SDK ships as a modern `pyproject.toml` package under `sdk/python`.
- Dockerfiles under `deployments/docker` provide reproducible local environments.

Refer to `docs/DEVELOPMENT.md` for language-specific tips, project conventions, and troubleshooting steps.

## Testing

| Component | Command |
| --- | --- |
| Control Plane | `cd control-plane && go test ./...` |
| Go SDK | `cd sdk/go && go test ./...` |
| Python SDK | `cd sdk/python && pytest` |

These commands are orchestrated by `scripts/test-all.sh`.

## Releases

GitHub Actions workflows (`.github/workflows/release.yml`) drive automated releases. The release process builds the control plane binary, packages SDKs, and publishes artifacts. See `docs/CONTRIBUTING.md` for guidance on versioning and release PR expectations.

## Contributing

We welcome issues and pull requests! Please review:

- `CODE_OF_CONDUCT.md` for community expectations.
- `docs/CONTRIBUTING.md` for contribution guidelines.
- `docs/DEVELOPMENT.md` for environment setup and tooling information.

## License

Haxen is licensed under the Apache 2.0 License. See `LICENSE` for details.

