const DEFAULT_AGENTFIELD_URL = 'http://localhost:8080';

export function resolveAgentFieldUrl(explicitUrl?: string): string {
  return (
    explicitUrl
    ?? process.env.AGENTFIELD_SERVER
    ?? process.env.AGENTFIELD_SERVER_URL
    ?? DEFAULT_AGENTFIELD_URL
  ).replace(/\/$/, '');
}
