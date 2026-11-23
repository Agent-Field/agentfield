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

## Local Development with Hot-Reload

For development with hot-reload, use the `dev.sh` script. This automatically rebuilds and restarts the server when Go files change.

```bash
cd control-plane
./dev.sh            # SQLite mode (default, no dependencies)
./dev.sh postgres   # PostgreSQL mode (set AGENTFIELD_DATABASE_URL first)
```

The server runs at `http://localhost:8080` and will automatically reload when you modify `.go`, `.yaml`, or `.yml` files.

**Notes:**
- Uses [Air](https://github.com/air-verse/air) for hot-reload (auto-installed if missing)
- Web UI is not included in dev mode; run `npm run dev` separately in `web/client/` if needed

## Configuration

Environment variables override `config/agentfield.yaml`. Common options:

- `AGENTFIELD_DATABASE_URL` – PostgreSQL DSN
- `AGENTFIELD_HOST` – bind host for HTTP and admin gRPC (empty by default: all interfaces)
- `AGENTFIELD_PORT` – HTTP port (`8080` by default)
- `AGENTFIELD_LOG_LEVEL` – log verbosity (`info`, `debug`, etc.)

Sample config files live in `config/`.

Set `agentfield.host: 127.0.0.1` in YAML to restrict both listeners to loopback,
or select a host with `af server --host 127.0.0.1 --open=false`. The standalone
`agentfield-server` accepts the same flag. Precedence is `--host` over
`AGENTFIELD_HOST` over YAML, including an explicit empty value to bind all
interfaces. Use an unbracketed IPv6 address such as `::1`. HTTP and admin gRPC
keep their existing port settings (admin defaults to HTTP port + 100).

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
