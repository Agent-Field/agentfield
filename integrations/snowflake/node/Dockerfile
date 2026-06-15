FROM golang:1.25-alpine AS build

WORKDIR /src
COPY sdk/go ./sdk/go
COPY integrations/snowflake ./integrations/snowflake

WORKDIR /src/integrations/snowflake/node
RUN go mod download
RUN CGO_ENABLED=0 GOOS=linux go build -o /out/snowflake-node ./cmd/snowflake-node

FROM gcr.io/distroless/static-debian12:nonroot

COPY --from=build /out/snowflake-node /snowflake-node
COPY integrations/snowflake/prompts/default-prompts.yaml /etc/agentfield/snowflake/prompts.yaml

ENV SNOWFLAKE_DEFAULT_PROMPTS_FILE=/etc/agentfield/snowflake/prompts.yaml
ENV SNOWFLAKE_NODE_LISTEN=:8012
EXPOSE 8012

ENTRYPOINT ["/snowflake-node", "serve"]
