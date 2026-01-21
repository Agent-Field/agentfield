# AgentField Railway Template

Deploy the AgentField control plane with PostgreSQL on Railway.

[![Deploy on Railway](https://railway.com/button.svg)](https://railway.com/new/template?template=https://github.com/Agent-Field/agentfield&envs=AGENTFIELD_STORAGE_MODE,AGENTFIELD_HTTP_ADDR,AGENTFIELD_STORAGE_POSTGRES_ENABLE_AUTO_MIGRATION,AGENTFIELD_UI_ENABLED&AGENTFIELD_STORAGE_MODE=postgres&AGENTFIELD_HTTP_ADDR=0.0.0.0:${{PORT}}&AGENTFIELD_STORAGE_POSTGRES_ENABLE_AUTO_MIGRATION=true&AGENTFIELD_UI_ENABLED=true)

## Quick Setup

1. Click "Deploy on Railway" above
2. Add a **PostgreSQL** plugin to your project
3. Add this environment variable to the control-plane service:
   ```
   AGENTFIELD_POSTGRES_URL=${{Postgres.DATABASE_URL}}
   ```
4. Deploy and access the dashboard at your Railway URL

## What Gets Deployed

- **Control Plane** - AgentField orchestration server with embedded Web UI
- **PostgreSQL** - Add via Railway plugin (required)

## Environment Variables

Configure these in your Railway service:

| Variable | Value | Description |
|----------|-------|-------------|
| `AGENTFIELD_STORAGE_MODE` | `postgres` | Use PostgreSQL backend |
| `AGENTFIELD_POSTGRES_URL` | `${{Postgres.DATABASE_URL}}` | Auto-wired from Railway |
| `AGENTFIELD_HTTP_ADDR` | `0.0.0.0:${{PORT}}` | Railway assigns port |
| `AGENTFIELD_STORAGE_POSTGRES_ENABLE_AUTO_MIGRATION` | `true` | Run migrations on startup |
| `AGENTFIELD_UI_ENABLED` | `true` | Enable web dashboard |

## After Deployment

1. Open the deployed URL to access the AgentField dashboard
2. Create your first agent:

```bash
# Install the CLI
curl -sSf https://agentfield.ai/get | sh

# Initialize a new agent
af init my-agent
cd my-agent

# Point to your Railway deployment
export AGENTFIELD_SERVER=https://your-app.up.railway.app

# Run the agent
af run
```

## Deploy an Example Agent

Once your control plane is running, deploy an agent to connect to it:

```bash
# Clone and navigate to an example
git clone https://github.com/Agent-Field/agentfield.git
cd agentfield/examples/python_agent_nodes/hello_world

# Set the control plane URL
export AGENTFIELD_SERVER=https://your-control-plane.up.railway.app

# Run locally or deploy to Railway as a separate service
python main.py
```

Or use the TypeScript init-example at `examples/ts-node-examples/init-example/`.

## Local Development

```bash
# Clone the repo
git clone https://github.com/Agent-Field/agentfield.git
cd agentfield

# Run locally with Docker Compose
cd deployments/docker
docker compose up
```

## Resources

- [Documentation](https://github.com/Agent-Field/agentfield)
- [Examples](https://github.com/Agent-Field/agentfield/tree/main/examples)
- [Python SDK](https://pypi.org/project/agentfield/)
