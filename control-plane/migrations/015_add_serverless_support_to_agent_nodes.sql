-- Migration: Add serverless deployment support to agent_nodes
-- Description: Adds deployment_type and invocation_url columns to support serverless agent deployments

-- Add deployment_type column (defaults to 'long_running' for existing agents)
ALTER TABLE agent_nodes
ADD COLUMN deployment_type VARCHAR(50) DEFAULT 'long_running' NOT NULL;

-- Add invocation_url column (nullable, only used for serverless agents)
ALTER TABLE agent_nodes
ADD COLUMN invocation_url TEXT;

-- Create index on deployment_type for efficient filtering
CREATE INDEX idx_agent_nodes_deployment_type ON agent_nodes(deployment_type);

-- Add comment to document the column
COMMENT ON COLUMN agent_nodes.deployment_type IS 'Deployment type: "long_running" for traditional agents, "serverless" for Lambda/Cloud Functions';
COMMENT ON COLUMN agent_nodes.invocation_url IS 'Invocation URL for serverless agents (e.g., Lambda function URL, Cloud Function endpoint)';
