# workaf-research

Answers an engineering question by building and measuring the options. The
question comes from the call: `question`, or a WorkAF job's ticket (`title`
and `body`). Only a call with neither uses the default, which compares three Go
SQLite drivers for a small CLI.

| Reasoner | What it is |
| --- | --- |
| `research` | entry point, `{"question": "…"}` or a WorkAF job's `{"title", "body"}` |
| `plan` | codeaf harness: the two to four options the question names, the criteria, one probe program |
| `investigate` | codeaf harness, called once per option at the same time: builds and measures the probe on that option |
| `synthesize` | codeaf harness: compares the measurements and recommends one |

```sh
export AGENTFIELD_URL=http://localhost:8080 OPENROUTER_API_KEY=…
go run .
curl -X POST $AGENTFIELD_URL/api/v1/execute/async/research.research -d '{"input":{}}'
```
