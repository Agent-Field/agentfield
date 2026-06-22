# Linear Integration

This pack ships a first-party Linear webhook source plus a deterministic Go capability node.

Use the source when Linear should start an AgentField run from issue, comment, project, team, or cycle events. Use the node when an agent needs bounded Linear API operations such as reading an issue, creating a follow-up issue, or adding a comment.

## Trigger Source

- Source name: `linear`
- Kind: HTTP webhook
- Secret: `LINEAR_WEBHOOK_SECRET`
- Signature: `Linear-Signature` HMAC-SHA256 over the raw body
- Idempotency: `Linear-Delivery`
- Event type: `<type>.<action>`, lowercased, for example `issue.create`

## Capability Node

Required env:

```bash
export LINEAR_API_KEY=lin_api_...
```

Useful capabilities:

- `get_issue`: read an issue by UUID or identifier.
- `list_issues`: list recent issues visible to the token.
- `create_issue`: pass Linear `IssueCreateInput` fields.
- `update_issue`: pass Linear `IssueUpdateInput` fields.
- `comment_issue`: add a comment to an issue.
- `list_teams` and `list_projects`: discover IDs before writes.

These are provider API calls. They do not perform triage, prioritization, or planning.
