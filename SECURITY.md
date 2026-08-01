# Security Policy

## Reporting a Vulnerability

Please report security vulnerabilities to: contact@agentfield.ai

**Do NOT open public issues for security vulnerabilities.**

We will respond within 48 hours and work with you to understand and address the issue.

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Security Best Practices

When using AgentField:

- Keep your control plane and SDKs updated to the latest version
- Use environment variables for sensitive configuration (API keys, database URLs)
- Set `AGENTFIELD_API_KEY` on any control plane reachable from beyond the machine
  it runs on. The control plane listens on every interface, and installing a
  package runs that package's own build and dependency steps — so an unprotected
  control plane on a shared network lets anyone on it run code as the user
  running the server. Without a key, those endpoints are refused for every
  non-local caller.
- Enable TLS in production deployments
- Review agent permissions and limit scope where possible
- Monitor workflow audit logs for unexpected behavior
