# AgentField triggers-demo-go agent.
#
# Build context expectation: the repo root, so the in-tree Go SDK can be
# copied in and resolved through the go.mod replace directive. The demo
# deliberately builds against the in-tree SDK so changes to the triggers DX
# flow through immediately, without waiting on a module release.
#
# Build:
#   docker build -f examples/triggers-demo-go/Dockerfile -t agentfield/triggers-demo-go-agent .
# Or via docker-compose from this directory.

FROM golang:1.22-alpine AS build

WORKDIR /src

# Copy the SDK first so the replace directive resolves, and so edits to the
# demo source do not invalidate the dependency-download layer.
COPY sdk/go ./sdk/go
COPY examples/triggers-demo-go/go.mod examples/triggers-demo-go/go.sum ./examples/triggers-demo-go/

WORKDIR /src/examples/triggers-demo-go
RUN go mod download

COPY examples/triggers-demo-go/main.go ./

# CGO is off so the binary runs on a bare alpine image without libc surprises.
RUN CGO_ENABLED=0 GOOS=linux go build -o /out/agent .

FROM alpine:3.20

# Certificates for outbound HTTPS to the control plane.
RUN apk add --no-cache ca-certificates

COPY --from=build /out/agent /usr/local/bin/agent

EXPOSE 8001
CMD ["/usr/local/bin/agent"]
