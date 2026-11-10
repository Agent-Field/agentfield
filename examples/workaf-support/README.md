# workaf-support

A customer-support agent node on the Go SDK. A ticket arrives on a webhook, and
the node triages it, has a coding agent find the cause in the product's
repository, then has a coding agent write the customer reply and a tested fix.

| Reasoner | What it is |
| --- | --- |
| `handle_ticket` | the webhook's reasoner (`generic_bearer`); calls the four below in order |
| `triage` | plain code: ticket kind and repository, by keyword |
| `investigate` | codeaf harness, read-only, in a clone of the repository; calls `search_docs` first |
| `search_docs` | plain code: greps the README, Markdown and Go doc comments |
| `draft_reply` | codeaf harness in a scratch copy: makes the fix, adds a test, writes the reply; the diff comes back with the result |

Each step is an ordinary reasoner reached with `app.Call`, so each is its own
execution with a parent link. Nothing in the code knows who watches the run.

```sh
export AGENTFIELD_URL=http://localhost:8080 OPENROUTER_API_KEY=… GITHUB_TOKEN=…
go run .
# on the control plane's host, the trigger's bearer:  SUPPORT_INTAKE_BEARER=…
curl -X POST $AGENTFIELD_URL/sources/<trigger id> -H "Authorization: Bearer $SUPPORT_INTAKE_BEARER" \
  -H 'X-Ticket-Id: T-1' -d '{"ticket":"the ledger CLI total is wrong after I add a refund"}'
```

The trigger id is in `GET /api/v1/triggers`. Every setting is in the comment at
the top of `main.go`. `deploy/workaf-proto/` runs it on a separate machine.
