FROM golang:1.23-alpine AS build
WORKDIR /src
COPY sdk/go /src/sdk/go
COPY integrations/sentry/node /src/integrations/sentry/node
WORKDIR /src/integrations/sentry/node
RUN go build -o /out/sentry-node ./cmd/sentry-node

FROM alpine:3.21
RUN adduser -D -u 10001 agentfield
USER agentfield
COPY --from=build /out/sentry-node /sentry-node
EXPOSE 8014
ENTRYPOINT ["/sentry-node"]
