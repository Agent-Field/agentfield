# workaf-research

Answers an engineering question by building and measuring the options. The
default question compares three Go SQLite drivers for a small CLI.

| Reasoner | What it is |
| --- | --- |
| `research` | entry point, `{"question": "…"}` (empty uses the default) |
| `plan` | codeaf harness: criteria and one probe program every investigation builds |
| `investigate_mattn`, `investigate_modernc`, `investigate_ncruces` | codeaf harness, called at the same time: builds, cross-compiles and times the probe on one driver |
| `synthesize` | codeaf harness: compares the three measurements and recommends one |

```sh
export AGENTFIELD_URL=http://localhost:8080 OPENROUTER_API_KEY=…
go run .
curl -X POST $AGENTFIELD_URL/api/v1/execute/async/research.research -d '{"input":{}}'
```
