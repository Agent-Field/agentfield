FROM golang:1.23-alpine AS build
WORKDIR /src
COPY sdk/go /src/sdk/go
COPY integrations/linear/node /src/integrations/linear/node
WORKDIR /src/integrations/linear/node
RUN go build -o /out/linear-node ./cmd/linear-node

FROM alpine:3.21
RUN adduser -D -u 10001 agentfield
USER agentfield
COPY --from=build /out/linear-node /linear-node
EXPOSE 8013
ENTRYPOINT ["/linear-node"]
