# Deploying agent nodes on Kubernetes

For agent nodes running on Kubernetes:

- Set `terminationGracePeriodSeconds` higher than the agent's drain budget, and leave extra time for the process to exit. Check the SDK release notes for the drain setting your SDK version honours (`AGENTFIELD_SHUTDOWN_TIMEOUT` where supported); older SDK releases cut in-flight Python reasoners at a fixed 30 seconds and Go skills at 5 seconds regardless of the pod's grace period.
- Keep the agent manifest's `version` string stable across ordinary pod rollouts. Changing it represents a new agent version, not a new replica.
- To recover or replay work, call [`POST /api/v1/executions/{execution_id}/restart`](api/EXECUTION_RESTART.md) with `reuse=succeeded-before` instead of submitting the original execute request again.
- Mount `AGENTFIELD_HOME` on a persistent volume claim. Size it for continuing SQLite/BoltDB and execution-payload growth until retention is enabled and configured for your deployment.

The control plane must be able to reach the callback URL registered by each agent; normally this is a Kubernetes Service DNS name.
