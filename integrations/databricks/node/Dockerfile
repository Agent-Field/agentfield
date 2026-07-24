FROM golang:1.25-alpine AS build

WORKDIR /src
COPY sdk/go ./sdk/go
COPY integrations/databricks ./integrations/databricks

WORKDIR /src/integrations/databricks/node
RUN go mod download
RUN CGO_ENABLED=0 GOOS=linux go build -o /out/databricks-node ./cmd/databricks-node

FROM gcr.io/distroless/static-debian12:nonroot

COPY --from=build /out/databricks-node /databricks-node
COPY integrations/databricks/prompts/default-prompts.yaml /etc/agentfield/databricks/prompts.yaml

ENV DATABRICKS_DEFAULT_PROMPTS_FILE=/etc/agentfield/databricks/prompts.yaml
ENV DATABRICKS_NODE_LISTEN=:8013
EXPOSE 8013

ENTRYPOINT ["/databricks-node", "serve"]
