# AgentField Control Plane

The AgentField control plane orchestrates agent workflows, manages verifiable credentials, serves the admin UI, and exposes REST/gRPC APIs consumed by the SDKs.

## Requirements

- Go 1.23+
- Node.js 20+ (for the web UI under `web/client`)
- PostgreSQL 15+

## Quick Start

```bash
# From the repository root
cd control-plane
go mod download
cd web/client
npm install
npm run build
cd ../..

# Run database migrations (requires AGENTFIELD_DATABASE_URL)
goose -dir ./migrations postgres "$AGENTFIELD_DATABASE_URL" up

# Start the control plane
AGENTFIELD_DATABASE_URL=postgres://agentfield:agentfield@localhost:5432/agentfield?sslmode=disable \
go run ./cmd/server
```

Visit `http://localhost:8080/ui/` to access the embedded admin UI.

## Local Docker Development

For development with hot-reload, use the `dev.sh` script. This automatically rebuilds and restarts the server when Go files change.

```bash
cd control-plane
./dev.sh            # SQLite mode (default, no dependencies)
./dev.sh postgres   # PostgreSQL mode
./dev.sh down       # Stop containers
./dev.sh clean      # Stop and remove volumes
```

The server runs at `http://localhost:8080` and will automatically reload when you modify `.go`, `.yaml`, or `.yml` files.

**Notes:**
- Uses [Air](https://github.com/air-verse/air) for hot-reload
- Go build cache is persisted in Docker volumes for faster rebuilds
- Web UI is not included in dev mode; run `npm run dev` separately in `web/client/` if needed

## Configuration

Environment variables override `config/agentfield.yaml`. Common options:

- `AGENTFIELD_DATABASE_URL` – PostgreSQL DSN
- `AGENTFIELD_HTTP_ADDR` – HTTP listen address (`0.0.0.0:8080` by default)
- `AGENTFIELD_LOG_LEVEL` – log verbosity (`info`, `debug`, etc.)

Sample config files live in `config/`.

## Web UI Development

```bash
cd control-plane/web/client
npm install
npm run dev
# Build production assets embedded in Go binaries
cd ../..
npm run build
```

Run the Go server alongside the UI so API calls resolve locally. During production builds the UI is embedded via Go's `embed` package.

## Database Migrations

Migrations use [Goose](https://github.com/pressly/goose):

```bash
AGENTFIELD_DATABASE_URL=postgres://agentfield:agentfield@localhost:5432/agentfield?sslmode=disable \
goose -dir ./migrations postgres "$AGENTFIELD_DATABASE_URL" status
```

## Testing

```bash
go test ./...
```

## Linting

```bash
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
golangci-lint run
```

## Releases

The `build-single-binary.sh` script creates platform-specific binaries and README artifacts. CI-driven releases are defined in `.github/workflows/release.yml`.
