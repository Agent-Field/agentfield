---
date: "2026-04-20"
researcher: "codebase-online-researcher"
topic: "GitHub Copilot CLI — Programmatic Invocation, Auth, Extensions, MCP, and Integration Patterns"
tags: [research, external, copilot-cli]
status: complete
local_binary: "/home/eioannidis/.local/bin/copilot"
local_version: "GitHub Copilot CLI 1.0.34"
---

# GitHub Copilot CLI — External Research Report

> **Scope**: Wrapping Copilot CLI as an agent inside an orchestrator (AgentField), shipping Copilot
> "skills" alongside an installer, and reusing Copilot CLI authentication. Research combines live
> `copilot --help` output (v1.0.34 on this machine), local config inspection, and official
> [docs.github.com](https://docs.github.com/en/copilot/how-tos/copilot-cli) documentation.

---

## 1. Programmatic / Non-Interactive Invocation

**Official docs**: [About GitHub Copilot CLI — Programmatic interface](https://docs.github.com/en/copilot/concepts/agents/copilot-cli/about-copilot-cli#programmatic-interface) · [CLI command reference](https://docs.github.com/en/free-pro-team@latest/copilot/reference/copilot-cli-reference/cli-command-reference)

### 1.1 The `-p` / `--prompt` Flag

The primary mechanism for non-interactive one-shot use is the `-p` / `--prompt` flag:

```bash
copilot -p "Fix the bug in main.js" --allow-all-tools
```

> "Execute a prompt in non-interactive mode (exits after completion)"
> — `copilot --help` (v1.0.34), confirmed by [official docs](https://docs.github.com/en/copilot/concepts/agents/copilot-cli/about-copilot-cli#programmatic-interface)

Key companion flags for scripting:

| Flag | Effect |
|------|--------|
| `-p` / `--prompt <text>` | Execute prompt non-interactively and exit |
| `--allow-all-tools` | Pre-approve every tool (required for unattended use); env: `COPILOT_ALLOW_ALL=true` |
| `--allow-all` / `--yolo` | Equivalent to `--allow-all-tools --allow-all-paths --allow-all-urls` |
| `-s` / `--silent` | Output only the agent response (no stats); **use with `-p` for scripting** |
| `--output-format json` | Emit JSONL (one JSON object per line) instead of human-readable text |
| `--no-ask-user` | Disable the `ask_user` tool; agent works fully autonomously |
| `--stream on\|off` | Enable/disable streaming mode |
| `--no-color` | Disable ANSI colour (useful in CI log parsers) |
| `--no-auto-update` | Disable automatic self-update (auto-disabled when `CI` env var is detected) |

### 1.2 Minimal One-Shot Invocations

```bash
# Simplest possible: read-only task
copilot -p "Summarise git log --oneline -10" --allow-tool='shell(git:*)'

# Full agentic: can edit files and run anything
copilot -p "Refactor foo.py to use dataclasses" --allow-all

# JSONL output — parse with jq
copilot -p "List open issues" --allow-all --output-format json | jq .

# Silent for piping to downstream tools
copilot -p "Generate a changelog entry" --allow-all --silent

# Per-session MCP config override (no config file edit required)
copilot -p "Run playwright tests" \
  --additional-mcp-config '{"mcpServers":{"pw":{"type":"stdio","command":"npx","args":["@playwright/mcp@latest"]}}}' \
  --allow-all
```

### 1.3 Piping Options

The `--help` output documents a second invocation pattern — piping shell-script-generated options:

```bash
./script-outputting-options.sh | copilot
```

This allows dynamic construction of flags from a parent process. ([Official docs](https://docs.github.com/en/copilot/concepts/agents/copilot-cli/about-copilot-cli#programmatic-interface))

### 1.4 Autopilot Mode

`--autopilot` (or `--mode autopilot`) starts Copilot in full autopilot mode where it continues working with minimal interruption. Combine with `--max-autopilot-continues <count>` to cap the number of continuation rounds:

```bash
copilot --autopilot --max-autopilot-continues 20 -p "Implement the TODO items in src/" --allow-all
```

### 1.5 Agent Client Protocol (`--acp`)

`--acp` starts Copilot as an **Agent Client Protocol server**, a structured machine-readable interface designed for orchestrator-to-agent communication. This is the native "wrap as an agent" entry point:

```bash
copilot --acp
```

> The ACP flag surfaces from `copilot --help` (v1.0.34). As of this writing the ACP specification
> is not fully documented on docs.github.com; it is an experimental interface. Use `--experimental`
> to ensure experimental features are on.

### 1.6 Session Resume

Copilot stores session state in `~/.copilot/session-state/` (1279 session dirs observed locally) and a SQLite store at `~/.copilot/session-store.db`. Sessions can be resumed:

```bash
copilot --continue                          # resume most recent session
copilot --resume                            # interactive session picker
copilot --resume=0cb916d                    # by 7+ hex-char prefix
copilot --resume=<full-uuid>                # by full UUID
```

### 1.7 CI / Automation Behaviour

Copilot CLI auto-detects CI environments via the `CI`, `BUILD_NUMBER`, `RUN_ID`, or `SYSTEM_COLLECTIONURI` environment variables and disables auto-update. The `--no-auto-update` flag achieves the same effect explicitly. ([`copilot help environment`])

---

## 2. Authentication Model

**Official docs**: [Authenticating GitHub Copilot CLI](https://docs.github.com/en/copilot/how-tos/copilot-cli/set-up-copilot-cli/authenticate-copilot-cli) · [Installing GitHub Copilot CLI](https://docs.github.com/en/copilot/how-tos/copilot-cli/set-up-copilot-cli/install-copilot-cli#authenticating-with-copilot-cli)

### 2.1 Authentication Methods (Priority Order)

Copilot CLI checks for credentials in this **exact order** ([official docs](https://docs.github.com/en/copilot/how-tos/copilot-cli/set-up-copilot-cli/authenticate-copilot-cli#about-authentication)):

1. `COPILOT_GITHUB_TOKEN` environment variable
2. `GH_TOKEN` environment variable
3. `GITHUB_TOKEN` environment variable
4. OAuth token from the OS keychain (service name `copilot-cli`)
5. `gh auth token` — GitHub CLI token fallback (lowest priority)

### 2.2 Supported Token Types

| Token type | Prefix | Supported | Notes |
|-----------|--------|-----------|-------|
| OAuth device-flow token | `gho_` | ✅ | Default via `copilot login` |
| Fine-grained PAT | `github_pat_` | ✅ | Must include "Copilot Requests" permission |
| GitHub App user-to-server | `ghu_` | ✅ | Via environment variable |
| Classic PAT | `ghp_` | ❌ | **Not supported** |

([Official supported token types table](https://docs.github.com/en/copilot/how-tos/copilot-cli/set-up-copilot-cli/authenticate-copilot-cli#supported-token-types))

### 2.3 OS Keychain Storage

| Platform | Keychain |
|---------|---------|
| macOS | Keychain Access |
| Windows | Credential Manager |
| Linux | libsecret (GNOME Keyring / KWallet) |

Fallback when keychain unavailable (headless server): plaintext token stored in **`~/.copilot/config.json`** (confirmed locally — `copilotTokens` field). ([Official docs](https://docs.github.com/en/copilot/how-tos/copilot-cli/set-up-copilot-cli/authenticate-copilot-cli#how-copilot-cli-stores-credentials))

### 2.4 Config File (Local Observation)

`~/.copilot/config.json` was observed locally and contains:

- `loggedInUsers` — array of `{host, login}` objects
- `lastLoggedInUser` — `{host, login}`
- `copilotTokens` — **redacted in this report**
- `trustedFolders` — list of pre-approved paths
- `installedPlugins` — array of installed plugin manifests
- `disabledSkills` — list of skill IDs to suppress
- `model` — last selected model identifier
- Various UI preferences (`theme`, `renderMarkdown`, etc.)

### 2.5 Subprocess Auth Inheritance

A subprocess that inherits `COPILOT_GITHUB_TOKEN`, `GH_TOKEN`, or `GITHUB_TOKEN` will authenticate automatically without any stored credentials. This is the **recommended pattern for AgentField orchestration**:

```bash
# Parent process sets the token; child Copilot CLI inherits it
export COPILOT_GITHUB_TOKEN="github_pat_..."
copilot -p "do task" --allow-all --silent
```

> "An environment variable silently overrides a stored OAuth token." — [official docs](https://docs.github.com/en/copilot/how-tos/copilot-cli/set-up-copilot-cli/authenticate-copilot-cli)

### 2.6 BYOK / No-Auth Mode

If `COPILOT_PROVIDER_BASE_URL` is set, GitHub authentication is **not required**. Copilot CLI routes model calls to the custom provider instead. The following features are **unavailable without GitHub auth**: `/delegate`, GitHub MCP server, GitHub Code Search. ([Official docs](https://docs.github.com/en/copilot/how-tos/copilot-cli/set-up-copilot-cli/authenticate-copilot-cli#unauthenticated-use))

Fully air-gapped mode: `COPILOT_OFFLINE=true` disables all network access including telemetry. ([`copilot help environment`])

### 2.7 GitHub Enterprise Cloud (Data Residency)

```bash
copilot login --host https://example.ghe.com
GH_HOST=mycompany.ghe.com copilot ...
```

---

## 3. Extensions / Skills / Plugins

**Official docs**: [About agent skills](https://docs.github.com/en/copilot/concepts/agents/about-agent-skills) · [Adding agent skills](https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/add-skills) · [About CLI plugins](https://docs.github.com/en/copilot/concepts/agents/copilot-cli/about-cli-plugins) · [Creating a plugin](https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/plugins-creating) · [Plugin reference](https://docs.github.com/en/copilot/reference/cli-plugin-reference)

### 3.1 Skills

Skills are the primary extensibility mechanism — folders containing a `SKILL.md` specification file that Copilot loads when relevant ([about agent skills](https://docs.github.com/en/copilot/concepts/agents/about-agent-skills)). The Agent Skills specification is an [open standard](https://github.com/agentskills/agentskills).

#### Skill Discovery Locations (searched automatically)

| Scope | Paths searched |
|-------|---------------|
| **Personal** (cross-project) | `~/.copilot/skills/`, `~/.claude/skills/`, `~/.agents/skills/` |
| **Project** (repo-specific) | `.github/skills/`, `.claude/skills/`, `.agents/skills/` |

Additional directories via env: `COPILOT_CUSTOM_INSTRUCTIONS_DIRS` (comma-separated).

**Locally observed**: `~/.copilot/skills/` contains 47 skill subdirectories including `playwright-cli`, `pdf`, `docx`, `research-codebase`, `semantic-code-search`, `test-generation`, etc.

#### `SKILL.md` Schema

```yaml
---
name: my-skill              # required; kebab-case; unique identifier
description: "..."          # required; when Copilot should use this skill
license: MIT                # optional
allowed-tools: shell        # optional; pre-approves tools (use with caution)
---

# Skill instructions in Markdown...
```

([Official SKILL.md schema](https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/add-skills#example-skillmd-file))

> ⚠️ Only add `shell` to `allowed-tools` for fully trusted skills — it removes confirmation prompts for arbitrary shell execution.

#### Skill Management Commands

```bash
/skills list                    # list all loaded skills
/skills info <name>             # inspect skill + file location
/skills reload                  # hot-reload after adding/editing skills
/skills                         # toggle skills on/off interactively
```

Via CLI: `copilot --available-tools=...` to restrict which tools (and by extension skills) are exposed.

#### Shipping Skills with an Installer

To ship skills with AgentField's installer, place skill directories in one of these locations:

- **Per-user (recommended)**: `~/.copilot/skills/<skill-name>/SKILL.md`
- **Per-repo**: `.github/skills/<skill-name>/SKILL.md`
- **In a plugin** (see §3.2): `plugin.json` → `"skills": ["skills/"]`

The `--config-dir <directory>` flag overrides `~/.copilot/` entirely, allowing isolated installer-managed configs. The `COPILOT_HOME` env var achieves the same.

### 3.2 Plugins

Plugins are distributable packages that bundle agents, skills, hooks, MCP configs, and LSP configs into a single installable unit. ([About CLI plugins](https://docs.github.com/en/copilot/concepts/agents/copilot-cli/about-cli-plugins))

#### Install Sources

```bash
copilot plugin install spark@copilot-plugins      # from registered marketplace
copilot plugin install owner/repo                  # GitHub repo (root)
copilot plugin install owner/repo:plugins/mypkg   # GitHub repo subdirectory
copilot plugin install https://github.com/o/r.git # git URL
copilot plugin install ./my-plugin                 # local directory
```

([Plugin install docs](https://docs.github.com/en/copilot/reference/cli-plugin-reference#plugin-specification-for-install-command))

The `--plugin-dir <directory>` global flag loads a plugin for the current session only (without persisting to config):

```bash
copilot --plugin-dir ./agentfield-skills-plugin -p "do task" --allow-all
```

#### Plugin Directory Structure

```
my-plugin/
├── plugin.json           # required manifest
├── agents/               # *.agent.md files
│   └── helper.agent.md
├── skills/               # skill subdirectories with SKILL.md
│   └── deploy/
│       └── SKILL.md
├── hooks.json            # event handlers
└── .mcp.json             # MCP server config
```

([Plugin structure docs](https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/plugins-creating#plugin-structure))

#### `plugin.json` Schema

```json
{
  "name": "agentfield-skills",
  "description": "AgentField orchestrator skills for Copilot CLI",
  "version": "1.0.0",
  "author": { "name": "AgentField", "email": "team@example.com" },
  "license": "MIT",
  "agents": "agents/",
  "skills": ["skills/"],
  "hooks": "hooks.json",
  "mcpServers": ".mcp.json"
}
```

([`plugin.json` reference](https://docs.github.com/en/copilot/reference/cli-plugin-reference#pluginjson))

#### Local Plugin Install Path

Observed locally at `~/.copilot/installed-plugins/`:

```
~/.copilot/installed-plugins/
├── _direct/                      # directly-installed from GitHub repos
│   └── FStarLang--proof-copilot/
└── superpowers-marketplace/
    └── superpowers/
```

#### Default Marketplaces

| Marketplace | GitHub Repo |
|------------|------------|
| `copilot-plugins` | [github/copilot-plugins](https://github.com/github/copilot-plugins) |
| `awesome-copilot` | [github/awesome-copilot](https://github.com/github/awesome-copilot) |

([Marketplace docs](https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/plugins-marketplace))

#### Custom Agents

Agents are `*.agent.md` files in `~/.copilot/agents/` or `agents/` in a plugin:

```yaml
---
name: my-agent
description: Helps with specific tasks
tools: ["bash", "edit", "view"]
---

You are a specialized assistant that...
```

Locally observed: `~/.copilot/agents/` contains 18 `.md` files including `orchestrator.md`, `worker.md`, `planner.md`, `reviewer.md`, `agentic-workflows.agent.md`, `DeepTest.agent.md`.

Select an agent at startup: `copilot --agent orchestrator`

---

## 4. Installation

**Official docs**: [Installing GitHub Copilot CLI](https://docs.github.com/en/copilot/how-tos/copilot-cli/set-up-copilot-cli/install-copilot-cli)

### 4.1 Install Methods

| Method | Command | Platform |
|--------|---------|---------|
| **npm** | `npm install -g @github/copilot` (Node.js 22+ required) | All |
| **WinGet** | `winget install GitHub.Copilot` | Windows |
| **Homebrew** | `brew install copilot-cli` | macOS, Linux |
| **Install script** | `curl -fsSL https://gh.io/copilot-install \| bash` | macOS, Linux |
| **Direct download** | [github/copilot-cli/releases](https://github.com/github/copilot-cli/releases/) | All |

Pre-release variant: append `@prerelease` (npm), `.Prerelease` (WinGet), `@prerelease` (Homebrew).

Custom install dir (install script): `PREFIX=$HOME/.local curl -fsSL https://gh.io/copilot-install | bash`

### 4.2 Binary Details (Locally Observed)

| Property | Value |
|---------|-------|
| Binary name | `copilot` |
| Install path (this machine) | `/home/eioannidis/.local/bin/copilot` |
| Binary size | 139 MB (single self-contained ELF x86-64 executable) |
| Format | ELF 64-bit LSB, dynamically linked |
| Version command | `copilot --version` or `copilot version` |
| Current version | `GitHub Copilot CLI 1.0.34` |
| Config dir | `~/.copilot/` (override: `COPILOT_HOME` or `--config-dir`) |
| Log dir | `~/.copilot/logs/` (override: `--log-dir`) |

### 4.3 Config Dir Override (Isolation Pattern)

For AgentField spawning multiple isolated Copilot instances:

```bash
COPILOT_HOME=/var/agentfield/instances/agent-001/.copilot \
  copilot -p "do task" --allow-all --silent
```

Or:
```bash
copilot --config-dir /var/agentfield/instances/agent-001 -p "do task" --allow-all --silent
```

---

## 5. MCP Server Configuration

**Official docs**: [Adding MCP servers for GitHub Copilot CLI](https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/add-mcp-servers)

### 5.1 Config File Locations (Priority / Sources)

MCP configuration is loaded from multiple sources, shown in `copilot mcp --help` ([v1.0.34]):

| Source | Location |
|--------|---------|
| **User** (persistent) | `~/.copilot/mcp-config.json` |
| **Workspace** | `.mcp.json` in current working directory |
| **Plugin** | `.mcp.json` or `.github/mcp.json` in installed plugin |
| **Builtin** | GitHub MCP server (pre-installed, cannot be removed but can be disabled) |

### 5.2 `mcp-config.json` Schema

Locally observed at `~/.copilot/mcp-config.json`:

```json
{
  "mcpServers": {
    "fstar": {
      "type": "stdio",
      "command": "/home/eioannidis/fstar-mcp/target/release/fstar-mcp",
      "tools": ["*"],
      "args": ["--log", "/tmp/fstar-mcp.log"]
    }
  }
}
```

Full schema from [official docs](https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/add-mcp-servers#editing-the-configuration-file):

```json
{
  "mcpServers": {
    "playwright": {
      "type": "local",
      "command": "npx",
      "args": ["@playwright/mcp@latest"],
      "env": {},
      "tools": ["*"]
    },
    "context7": {
      "type": "http",
      "url": "https://mcp.context7.com/mcp",
      "headers": { "CONTEXT7_API_KEY": "YOUR-API-KEY" },
      "tools": ["*"]
    }
  }
}
```

Transport types: `stdio` / `local`, `http` (Streamable HTTP), `sse` (legacy Server-Sent Events).

### 5.3 CLI Commands for MCP Management

```bash
copilot mcp add context7 -- npx -y @upstash/context7-mcp    # add stdio server
copilot mcp add --transport http notion https://mcp.notion.com/mcp  # add HTTP server
copilot mcp list                                              # list all servers (all sources)
copilot mcp list --json                                       # machine-readable list
copilot mcp get <name>                                        # show server details
copilot mcp remove <name>                                     # remove server
```

Interactive slash commands: `/mcp show`, `/mcp add`, `/mcp edit <name>`, `/mcp delete <name>`, `/mcp disable <name>`, `/mcp enable <name>`

### 5.4 Per-Session MCP Override (No Config File Edit)

```bash
copilot --additional-mcp-config '{"mcpServers":{"myserver":{"type":"stdio","command":"my-mcp-server"}}}' \
  -p "do task" --allow-all
```

Or via file reference:
```bash
copilot --additional-mcp-config @/path/to/session-mcp.json -p "do task" --allow-all
```

The `--additional-mcp-config` flag can be repeated and **augments** (does not replace) `~/.copilot/mcp-config.json`. ([v1.0.34 `--help`])

### 5.5 Built-in GitHub MCP Server

The GitHub MCP server ships built-in and is available without configuration. It can be disabled:

```bash
copilot --disable-builtin-mcps -p "offline task" --allow-all
copilot --disable-mcp-server github-mcp-server -p "partial task" --allow-all
```

To expand the default GitHub MCP toolset:
```bash
copilot --add-github-mcp-tool my_tool -p "..."
copilot --add-github-mcp-toolset all -p "..."       # all toolsets
copilot --enable-all-github-mcp-tools -p "..."      # all tools
```

### 5.6 AgentField Control Plane as MCP Server

To expose AgentField's control plane as an MCP server consumed by Copilot CLI:

```json
{
  "mcpServers": {
    "agentfield": {
      "type": "http",
      "url": "http://localhost:8080/mcp",
      "headers": {
        "Authorization": "Bearer ${AGENTFIELD_TOKEN}"
      },
      "tools": ["*"]
    }
  }
}
```

Or via stdio, wrapping the `af` CLI:
```json
{
  "mcpServers": {
    "agentfield": {
      "type": "stdio",
      "command": "af",
      "args": ["mcp", "serve"],
      "tools": ["*"]
    }
  }
}
```

The control plane's `internal/mcp/` package (see `CLAUDE.md`) provides the MCP integration layer. Toolsets AgentField could expose include: workflow execution, agent registry queries, skill dispatch, memory read/write, DID/VC audit queries.

---

## 6. Rate Limits, Session State, and CWD Handling

### 6.1 Rate Limits

**No public documentation found** on specific rate limits for Copilot CLI. Rate limits are governed by the user's Copilot subscription plan. GitHub's general [Copilot API rate limiting](https://docs.github.com/en/copilot/managing-copilot/managing-github-copilot-in-your-organization/managing-github-copilot-features-in-your-organization/managing-policies-for-copilot-in-your-organization) documentation does not specify per-process or per-minute limits for CLI use. For orchestration scenarios spawning many parallel instances, plan limits should be tested empirically.

### 6.2 Session State

Session state is persisted between calls:

| Artifact | Path |
|---------|------|
| Session state dirs | `~/.copilot/session-state/<uuid>/` (1279 observed) |
| Session SQLite store | `~/.copilot/session-store.db` (48 MB observed) |
| Command history | `~/.copilot/command-history-state.json` |
| Embedding cache | `~/.copilot/embedding-cache.db` |
| Logs | `~/.copilot/logs/` |

Isolate sessions by overriding `COPILOT_HOME` or `--config-dir` per subprocess instance.

### 6.3 CWD Handling

Copilot CLI **defaults to the current working directory** for file access. This is critical for AgentField's orchestrator spawning Copilot subprocesses — each subprocess should be launched with `cwd` set to the target repository:

```python
subprocess.run(
    ["copilot", "-p", "Fix the bug", "--allow-all", "--silent"],
    cwd="/path/to/target/repo",
    env={**os.environ, "COPILOT_GITHUB_TOKEN": token},
)
```

Path access rules:
- Default: CWD + subdirectories + system temp dir
- `--add-dir <directory>`: expand allowed paths (repeatable)
- `--allow-all-paths`: disable path verification entirely
- `--disallow-temp-dir`: remove automatic temp dir access

In-session CWD changes: `/cwd /path`, `/cd /path` (does **not** change the spawning process's CWD).

Trust prompt: on first run in a new directory, Copilot prompts to trust the folder. Pre-populate `trustedFolders` in `~/.copilot/config.json` to suppress this in automation:

```json
{
  "trustedFolders": ["/path/to/workspace"]
}
```

### 6.4 Output Formats

| Flag | Output |
|------|--------|
| *(default)* | Human-readable text with ANSI colours, stats |
| `--silent` / `-s` | Agent response only; no stats |
| `--output-format json` | JSONL — one JSON object per line |
| `--no-color` | Plain text without ANSI codes |
| `--plain-diff` | Plain diff (no syntax highlighting) |

The `--output-format json` flag is the preferred interface for structured parsing in an orchestrator. Output is JSONL (newline-delimited JSON), one object per message/event.

### 6.5 OpenTelemetry / Observability

Full OTel support for tracing every LLM call and tool execution ([`copilot help monitoring`]):

```bash
# Export traces to a local Jaeger collector
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
  copilot -p "do task" --allow-all

# Write JSONL traces to file (no collector required)
COPILOT_OTEL_FILE_EXPORTER_PATH=/var/log/copilot-otel.jsonl \
  copilot -p "do task" --allow-all
```

Trace hierarchy: `invoke_agent → chat <model> → execute_tool <tool>`

Metrics exported: `gen_ai.client.operation.duration`, `gen_ai.client.token.usage`, `github.copilot.tool.call.count`, `github.copilot.tool.call.duration`, `github.copilot.agent.turn.count`

---

## 7. Summary Table: Key Flags for Orchestration

| Use Case | Command Pattern |
|---------|----------------|
| One-shot non-interactive | `copilot -p "task" --allow-all --silent` |
| JSON output for parsing | `copilot -p "task" --allow-all --output-format json` |
| Pre-authorized token (CI) | `COPILOT_GITHUB_TOKEN=<token> copilot -p "task" --allow-all` |
| Isolated config per agent | `COPILOT_HOME=~/.copilot-agent-N copilot -p "task" --allow-all` |
| Custom MCP per session | `copilot --additional-mcp-config @session.json -p "task" --allow-all` |
| ACP protocol server | `copilot --acp` |
| Ship skills in installer | Place `SKILL.md` under `~/.copilot/skills/<name>/` |
| Ship skills as plugin | `copilot plugin install ./agentfield-plugin` or `--plugin-dir` |
| BYOK (no GitHub auth) | `COPILOT_PROVIDER_BASE_URL=... COPILOT_MODEL=... copilot -p "task" --allow-all` |
| Full offline mode | `COPILOT_OFFLINE=true COPILOT_PROVIDER_BASE_URL=... copilot -p "task" --allow-all` |
| Resume previous session | `copilot --resume=<session-id> --allow-all -p "continue task"` |

---

## 8. Additional Resources

- [GitHub Copilot CLI home page](https://docs.github.com/en/copilot/how-tos/copilot-cli)
- [About GitHub Copilot CLI](https://docs.github.com/en/copilot/concepts/agents/copilot-cli/about-copilot-cli)
- [CLI command reference](https://docs.github.com/en/free-pro-team@latest/copilot/reference/copilot-cli-reference/cli-command-reference)
- [Authenticating GitHub Copilot CLI](https://docs.github.com/en/copilot/how-tos/copilot-cli/set-up-copilot-cli/authenticate-copilot-cli)
- [Adding agent skills](https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/add-skills)
- [About agent skills (open standard)](https://docs.github.com/en/copilot/concepts/agents/about-agent-skills)
- [Agent Skills open standard (GitHub)](https://github.com/agentskills/agentskills)
- [About CLI plugins](https://docs.github.com/en/copilot/concepts/agents/copilot-cli/about-cli-plugins)
- [Creating a plugin](https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/plugins-creating)
- [Plugin reference](https://docs.github.com/en/copilot/reference/cli-plugin-reference)
- [Adding MCP servers](https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/add-mcp-servers)
- [Installing Copilot CLI](https://docs.github.com/en/copilot/how-tos/copilot-cli/set-up-copilot-cli/install-copilot-cli)
- [Copilot CLI releases (binary downloads)](https://github.com/github/copilot-cli/releases/)
- [Awesome Copilot (community skills/plugins)](https://github.com/github/awesome-copilot)
- [Copilot-plugins marketplace](https://github.com/github/copilot-plugins)

---

## 9. Gaps and Limitations

1. **ACP (`--acp`) protocol specification**: The Agent Client Protocol flag is present in v1.0.34 but the full wire protocol is not publicly documented on docs.github.com as of 2026-04-20. Test empirically.

2. **Rate limits**: No public per-CLI-invocation or per-minute rate limit figures are documented. Use exponential backoff and respect `429` HTTP responses.

3. **JSONL schema**: The exact schema of objects emitted by `--output-format json` is not documented. Must be inferred from empirical testing.

4. **Session isolation in parallel instances**: When running multiple Copilot CLI subprocesses in parallel, use separate `COPILOT_HOME` dirs to prevent config/session-store contention. This is inferred from the file-based session storage model, not explicitly documented.

5. **Token lifetime / expiry**: Token expiry for OAuth tokens from `copilot login` is not documented — depends on organization OAuth policy settings.

6. **Plugin update in non-interactive mode**: `copilot plugin update` may require a TTY. Test in CI before assuming it works headlessly.

---

*Research conducted 2026-04-20 using: local `copilot` binary v1.0.34, config at `~/.copilot/`, and live fetches from docs.github.com. All `--help` output is from the locally installed binary.*
```

---

## Summary of Findings

Here is a digest of the most actionable facts for AgentField:

### (a) Wrapping Copilot CLI as an Agent in AgentField's Orchestrator

- **One-shot invocation**: `copilot -p "<task>" --allow-all --silent --output-format json`
- **No interactive TTY needed**; the binary exits after completing the prompt
- **ACP server mode** (`--acp`) is the native orchestrator interface — starts Copilot as an Agent Client Protocol server; wire protocol not yet publicly documented
- **Auth inheritance**: set `COPILOT_GITHUB_TOKEN` in the subprocess environment; it takes precedence over everything else
- **Isolate state** per agent: `COPILOT_HOME=/path/to/agent-N-home` or `--config-dir`
- **`cwd`** the subprocess to the target repository directory — Copilot restricts file access to CWD by default
- **OTel tracing** (`OTEL_EXPORTER_OTLP_ENDPOINT` or `COPILOT_OTEL_FILE_EXPORTER_PATH`) gives full observability into LLM calls and tool use from the orchestrator

### (b) Shipping Copilot "Skills" with the AgentField Installer

- Drop skill directories under **`~/.copilot/skills/<name>/SKILL.md`** (user-level, cross-project)
- Or bundle as a **plugin** (`plugin.json` + `skills/` + optionally `agents/`, `hooks.json`, `.mcp.json`) and run `copilot plugin install ./agentfield-skills-plugin`
- The `--plugin-dir ./agentfield-skills-plugin` flag loads a plugin for a single session without persisting it — useful for controlled orchestrator invocations
- Skill files follow the [Agent Skills open standard](https://github.com/agentskills/agentskills): YAML frontmatter (`name`, `description`) + Markdown body

### (c) Reusing Copilot CLI Authentication

- **Env var pattern** (recommended for CI/automation): `COPILOT_GITHUB_TOKEN > GH_TOKEN > GITHUB_TOKEN`
- Fine-grained PAT with **"Copilot Requests" permission** is the correct token type
- Classic PATs (`ghp_`) are **not supported**
- If `gh` CLI is authenticated on the machine, Copilot CLI will use its token as a fallback automatically — no extra setup needed
- Tokens stored in OS keychain (service name `copilot-cli`) or `~/.copilot/config.json` fallback
