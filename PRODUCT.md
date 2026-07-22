# Product

## Register

product

## Users

Developers who are comfortable with GitHub and the terminal but are not necessarily infrastructure experts. They install AI agent nodes on their own machine, run them through a local control plane, and want everything "just working" by the time Claude Code / Codex / other coding agents query those agents. Context: the AgentField Desktop app is a Docker-Desktop-style companion — glanced at a few times a day, not lived in.

## Product Purpose

AgentField Desktop is the set-it-and-forget-it surface for the local AgentField control plane, which acts as a **sub-harness**: products and coding agents (Claude Code, Codex, …) offload agent API calls to locally installed agent nodes, often running on smaller/cheaper models. The two hero jobs, in order:

1. **Install agent nodes effortlessly** — from a curated catalog and, critically, from any GitHub repo that hosts a node. The ecosystem play is "various GitHub repos publish agent nodes, people install them for free" — so discovery + one-paste install is the front door.
2. **See what's running and what it costs** — live runs, health, and (as data allows) token usage / cost attribution per agent and per run, so users trust and can budget the offloading.

Secondary: start/stop/keys/auto-start management. Success = a user never has to remember to start a server, installs a node from a repo link in under 30 seconds, and can answer "what ran and what did it cost me?" at a glance.

## Brand Personality

Calm, precise, trustworthy. "Quiet infrastructure" — the tool disappears into the task. Modern developer-tool sensibility (Linear, Raycast, OpenAI's Codex app): dark-friendly, dense but never cluttered, instant feedback, no ceremony.

## Anti-references

- Enterprise admin consoles (Jenkins, pgAdmin): chrome-heavy, table-everything.
- Generic Electron-app blandness: default-looking controls, no personality.
- SaaS marketing gloss inside a utility app: gradients, hero metrics, decorative illustration.

## Design Principles

1. **Status you can trust** — state is always visible, current, and honest; ambiguity (e.g. "unknown") is surfaced, not hidden.
2. **One glance, one action** — every view answers its question immediately and puts the single most likely action within reach.
3. **Earned familiarity** — standard patterns from the best developer tools; novelty only where it demonstrably helps.
4. **Keys are sacred** — secret handling always looks and feels deliberate and safe.
5. **The app is furniture** — it should feel native, quiet, and fast; delight lives in micro-feedback, not choreography.

## Accessibility & Inclusion

WCAG AA contrast for all text; full keyboard operability for every action; status never conveyed by color alone; reduced-motion alternatives for all animation.

## Locked desktop decisions

Confirmed with product (2026-07-22). Full UX/visual detail lives in `DESIGN.md`.

1. **Accent:** gold / amber (tray-aligned). Not slate-teal; not Apple blue.
2. **Cold-launch route:** Home when any agents are installed; Install when zero. Do not remember last view.
3. **Usage in v1:** Home shows spend/tokens totals (and by-harness callers); Activity shows per-row tokens/cost when the API provides them.
4. **Update banner:** keep across all views; dismiss is per-version; Settings keeps the durable check.
