# Customizing SDK prompts

The SDKs inject some text into every LLM call: a schema-adherence instruction
when you pass an output schema, an optional tool system prompt, and the
tool-message framing the tool-call loop sends back to the model. All of it lives
in one overridable place per SDK, and the defaults reproduce the SDK's prior
behavior exactly, so an agent that never configures prompts sees no change.

## What you can override

| Field (Python / TypeScript / Go) | What it controls | Default |
|---|---|---|
| `schema_instruction` / (n/a) / (n/a) | The instruction appended to the system message when an output schema is in effect. **Python only.** | the built-in schema instruction |
| `tool_system_prompt` / `toolSystemPrompt` / `SystemPrompt` | An optional system prompt describing how to use tools, appended after your system prompt. | `None` / unset (nothing injected) |
| `tool_call_limit_reached` / `toolCallLimitReached` / `ToolCallLimitReached` | The message returned to the model when the tool-call limit is hit. | `"Tool call limit reached. Please provide a final response."` |
| `tool_error_formatter` / `toolErrorFormatter` / `ToolErrorFormatter` | Formats a failed tool call before it is sent back to the model. | `{ "error": <message>, "tool": <name> }` |
| `tool_result_formatter` / `toolResultFormatter` / `ToolResultFormatter` | Formats a successful tool result before it is sent back to the model. | the raw result, unframed |

A formatter returns a string (sent verbatim) or any JSON-serializable value
(JSON-encoded for you). Returning a string lets you send plain text; returning
an object keeps structured framing.

## Schema instruction: an intentional difference between SDKs

How a schema is enforced differs by SDK, and this is deliberate, not a bug:

- **Python** sends the native `response_format` **and** injects a text
  `schema_instruction` into the system message. Set
  `schema_instruction = None` to drop the text and rely on `response_format`
  alone, which makes Python behave like the other two.
- **TypeScript** uses the Vercel AI SDK's native `generateObject`, so there is
  no injected schema instruction and no `schemaInstruction` field.
- **Go** uses native structured output (`WithSchema` -> `response_format:
  json_schema`) and injects no schema instruction either.

## Python

```python
from agentfield import Agent, PromptTemplates

app = Agent(node_id="my-agent")

# Opt into a tool system prompt (nothing is injected by default):
app.ai_config.prompt_templates.tool_system_prompt = (
    "You have access to tools. Use them when the request needs action."
)

# Drop the schema instruction and rely on the native response_format:
app.ai_config.prompt_templates.schema_instruction = None

# Replace the tool-result framing with your own:
app.ai_config.prompt_templates.tool_result_formatter = (
    lambda tool, result: {"tool": tool, "output": result}
)
```

`schema_instruction` uses `{schema}` as the placeholder for the JSON schema, and
it is filled with `str.replace`, so a custom instruction may safely contain
literal braces (for example an example JSON object).

## TypeScript

```ts
import { Agent } from "@agentfield/sdk";

const app = new Agent({
  nodeId: "my-agent",
  aiConfig: {
    promptTemplates: {
      toolSystemPrompt: "Use tools when the request needs action.",
      toolResultFormatter: (tool, result) => ({ tool, output: result }),
    },
  },
});
```

`promptTemplates` is a partial object merged over the defaults, so you only set
the fields you want to change.

## Go

```go
import "github.com/Agent-Field/agentfield/sdk/go/ai"

cfg := ai.ToolCallConfig{
    SystemPrompt: "Use tools when the request needs action.",
    PromptConfig: &ai.PromptConfig{
        ToolCallLimitReached: "That's enough tool calls.",
        ToolResultFormatter: func(tool string, result map[string]interface{}) interface{} {
            return map[string]interface{}{"tool": tool, "output": result}
        },
    },
}
```

Go configures prompts per call on `ToolCallConfig`.

## Trace message tagging

Each SDK's tool-call trace tags every message it sent with its source, so you
can tell SDK-injected text apart from user and model content. The `source`
strings are shared across SDKs:

- `user` - your prompt and messages
- `assistant` - the model's own messages, including tool-call requests
- `sdk.tool_system_prompt` - the opt-in tool system prompt
- `sdk.tool_result` - a successful tool result
- `sdk.tool_error` - a failed tool call
- `sdk.tool_limit` - the tool-call-limit message
- `sdk.schema_instruction` - the schema instruction (Python only)

The tagged messages reference the content sent on the wire; the wire messages
are not reshaped to make tagging easier.

## Scope note

These templates cover the SDK's own LLM calls. The coding-agent harness
(`app.harness(...)`) has its own prompts and configuration and is not affected
by `prompt_templates`.
