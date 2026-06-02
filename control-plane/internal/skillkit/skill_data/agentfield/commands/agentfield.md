---
description: Design and ship a multi-agent system on AgentField from a plain-English description.
argument-hint: [describe the system — e.g. "a claims processor with risk scoring and human approval"]
---

Use the `agentfield` skill to design, scaffold, wire, and live-smoke-test a multi-agent system on AgentField.

User request: $ARGUMENTS

Follow the skill's hard gate: fetch the live docs from `agentfield.ai/llms.txt` first, run `af doctor --json`, pick the model (asking the user if needed), clarify only along architecture-changing axes, design the topology from the five principles (do not pick a named pattern off a menu), then scaffold and verify with the canonical async smoke test.
