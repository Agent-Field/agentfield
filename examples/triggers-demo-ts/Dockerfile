# AgentField triggers-demo-ts agent.
#
# Build context expectation: the repo root (so we can copy the local TS SDK
# and install it in the image — the demo deliberately uses the in-tree SDK
# so changes to triggers DX flow through immediately, without waiting on
# an npm publish).
#
# Build:
#   docker build -f examples/triggers-demo-ts/Dockerfile -t agentfield/triggers-demo-ts-agent .
# Or via docker-compose from this directory.

FROM node:20-slim

ENV NODE_ENV=production

WORKDIR /app

# Install the in-tree SDK first (so npm install resolves the file: reference).
COPY sdk/typescript /tmp/ts-sdk
RUN cd /tmp/ts-sdk && npm ci && npm run build

# Copy demo package manifest and install deps (leverages layer caching)
COPY examples/triggers-demo-ts/package.json examples/triggers-demo-ts/tsconfig.json /app/
RUN npm install --omit=dev

# Copy the agent source
COPY examples/triggers-demo-ts/agent.ts /app/agent.ts

# Install tsx for running TypeScript directly
RUN npm install tsx@^4.19.2

EXPOSE 8001
CMD ["npx", "tsx", "agent.ts"]
