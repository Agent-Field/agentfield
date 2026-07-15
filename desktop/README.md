# AgentField Desktop

A desktop companion for AgentField, in the spirit of Docker Desktop: one
window that shows the health of the local control plane, the agent nodes on
this machine, live execution activity — and installs curated agents with one
click. Designed Mac-first: no menu bar clutter, seamless titlebar, sidebar
navigation, light/dark from the OS.

## Views

- **Dashboard** — headline tiles (agents running, executing now, runs today,
  success rate, from `GET /api/ui/v1/dashboard/summary`) plus recent activity.
- **Agents** — nodes from `~/.agentfield/installed.yaml` with a status badge
  per agent, cross-checked against the control plane's `GET /api/v1/nodes`:
  - `running` — registry says running and the control plane sees the node
  - `stopped` — registry says stopped and the control plane does not see it
  - `unknown` — registry and control plane disagree (stale registry / conflict)
- **Activity** — in-flight workflow runs (live pulse) and a short tail of
  finished ones, from `GET /api/ui/v2/workflow-runs`.
- **Install** — a curated, hard-coded catalog (`src/shared/catalog.ts`)
  plus an **Install from repository** field for pasting any GitHub repo that
  hosts an installable node (`https://github.com/<owner>/<repo>`, or
  `…/<repo>//<subdir>` to pick one node out of a multi-node repo). Both shell
  out to `af install <source>` and stream progress lines into the row; the af
  CLI stays the single contract for installs. Catalog entries are keyed by the
  node's manifest name so installed state is detected. The hard-coded list is
  the pre-marketplace seam — replace with a remote catalog fetch when registry
  search lands. Security: catalog installs only ever send a curated name; the
  paste-a-repo channel is the one place a raw source reaches the main process,
  and it is validated there to an `https://github.com/…` shape before spawn
  (`parseRepoSource` in `src/main/installer.ts`) — no other host or scheme, and
  never a value that could be read as a CLI flag.
- **Settings** — the "set it and forget it" surface: open at login (hidden,
  tray-only, via an OS login item), start the control plane automatically,
  and pick which agents auto-start. Persisted to `settings.json` in the
  app's user-data dir.

The renderer polls a single snapshot over IPC every 5 seconds.

Agents can be started, stopped, and restarted from their rows — each action
shells out to `af run <name>` / `af stop <name>` (restart is stop-then-run;
the CLI has no restart verb). On every launch the app runs the autostart
sequence (`src/main/autostart.ts`): start the control plane when nothing is
listening (never when the port is taken — by a foreign service or a live
control plane), then bring up the selected agents; agents whose registry
entry went stale (running with no control-plane presence, e.g. after a
reboot) are restarted rather than skipped. The goal: by the time Claude,
Codex, or anything else queries your agents, they are already answering —
no one has to remember to start a server first.

The control-plane probe only trusts `/health` responses that look like
AgentField's payload — an unrelated service on port 8080 renders as
"Port in use", never as a running control plane.

## Agent keys (secrets)

Agents declare the environment they need (API keys, tokens) in their
`agentfield-package.yaml` under `user_environment` — required variables,
`require_one_of` groups (e.g. an Anthropic **or** an OpenRouter key), and
optionals with defaults. `af run` resolves them process env → encrypted
secret store → manifest default, and fails headlessly when a required one
is missing. The app closes that gap in the UI:

- Any agent that declares variables gets a **Keys** button and, while
  something required is unresolved, a **Needs keys** chip. Clicking Start
  in that state opens the editor instead of running into a guaranteed
  "missing required environment variables" failure.
- The editor shows every declared variable with its resolution status
  (from environment / stored / default / missing), a set field (password
  input for `type: secret`), and a **Revoke** button for stored values.
- Reads and writes go through the af CLI — `af secrets ls` / `set` / `rm`
  against the encrypted store (`~/.agentfield/secrets`, AES-256-GCM), the
  exact store `af run` decrypts into the agent's process. Values are piped
  over stdin (never argv), written to the scope the manifest names (global
  by default, so a shared key is entered once), validated against the
  manifest's regex when present, and **never read back** — the renderer
  only ever sees status flags.

All of this lives in `src/main/secrets.ts` (pure parsing/report logic,
unit-tested) with the catalog-style guard that the renderer can only name
variables the manifest declares.

## Tray icon (Windows/Linux) and deep links

On Windows and Linux the app puts a status icon in the tray: the brand dot
turns gold while the control plane is running and gray otherwise, and the
menu offers Open AgentField / Open web UI / Quit. Closing the window hides it
to the tray (Docker-Desktop style) — Quit lives in the tray menu.

On macOS the desktop app adds **no** in-app tray: the menu-bar companion there
is `af-tray` (`control-plane/cmd/af-tray`), and **the app now provisions and
installs it itself** (`src/main/tray-companion.ts`). On launch it stages the
bundled `af-tray` into `~/.agentfield/bin/af-tray` — the same managed location
the curl installer uses, so both installers converge there — and runs `af-tray
install`, which builds `~/Applications/AgentField.app` and the launchd agents.
A desktop-app-only install therefore gets the menu-bar icon without ever running
the curl installer. It only re-stages when the bundled copy is a newer stamped
version (via `af-tray version`, matching how `cli.ts` gates the CLI), and only
re-runs `af-tray install` when the binary changed or the tray's launchd agent
is not loaded — never unconditionally, since install reloads launchd and would
blink the tray on every launch. The **Show the menu bar icon** toggle in
Settings (macOS only) drives this: turning it off runs `af-tray uninstall`. The
bundled `af-tray` is built by `npm run bundle-cli` on macOS only (it carries the
systray/CGO dependency); Windows/Linux keep the in-app tray instead.

The app registers the `agentfield://` URL scheme (single-instance: a second
launch focuses the running app). `agentfield://dashboard|agents|activity|install`
opens the app on that view; a bare or unknown target lands on the dashboard.
This is how the macOS `af-tray` opens the desktop app when it is installed —
and why it can *detect* it: `open agentfield://…` fails fast when nothing has
registered the scheme, and the tray then falls back to the web UI.

## Icons

All icons render from the brand "•af" mark (the exact outlined paths from the
web UI logo). `npm run icons` regenerates `build/icon.{png,icns}`, the runtime
window icon, the tray glyphs, and af-tray's `appicon.icns` — outputs are
committed, so this only needs re-running when the mark changes.

## Prerequisites

- Node.js 20+ (developed on Node 22)
- An AgentField control plane on `http://localhost:8080` (optional — the app
  degrades gracefully when it is not running, and can start one itself)
- Nothing else: the packaged app bundles the `af` CLI (see below). In dev,
  either have `af` on PATH or run `npm run bundle-cli` once.

## Bundled CLI, resolution, and updates

The packaged app carries the af CLI (`resources/bin`, staged from `vendor/`
by `npm run bundle-cli` or the release pipeline) so a desktop-app-only
install works on a machine that never saw AgentField. On every launch the
app resolves which af to drive (`src/main/cli.ts`), in order:

1. **managed** — `~/.agentfield/bin` (where the curl installer puts it, and
   where the app provisions its bundled copy: the shared location both
   installers converge on, so there is never a double install)
2. **PATH** — a developer's own `af`
3. **bundled** — the copy inside the app package

A copy older than the app's minimum version is skipped — the app runs on its
bundled CLI meanwhile and Settings shows an "Update AgentField" button that
installs the bundled copy into the managed location (never over a newer
one; `Version: dev` builds are trusted as-is). On a machine with no CLI at
all, first launch auto-provisions `~/.agentfield/bin` (both `agentfield`
and the `af` alias, Windows user PATH registered) so terminals and coding
agents get a working `af` too.

Unless switched off in Settings, the app also installs both bundled skills
on launch (`af skill install <name> --non-interactive`, sequentially):
**agentfield** (how to build agents) and **agentfield-use** (how to discover
and call the agents installed here) — so detected coding agents (Claude
Code, Codex, Gemini, …) can drive this machine's agents without extra
setup. Idempotent via skillkit's state file, shared with the curl installer.

## App updates

The app updates itself from the public GitHub releases (`src/main/updates.ts`):
packaged builds poll `/releases/latest` shortly after launch and every few
hours (stable releases only — RC prereleases are never offered; an RC install
IS offered the stable build of its own version once that lands). A newer
release surfaces as an "Update available" banner across the top of the window
and under **Settings → App updates**. Installing downloads this platform's
installer asset from the release and hands off to it: Windows quits into the
NSIS one-click installer (which replaces the app in place and relaunches);
macOS opens the downloaded DMG for a drag-install (silent replacement needs
signed builds — a known follow-up). A release without an installer for the
platform falls back to opening the release page. Dismissing the banner hides
it for that version only (persisted in settings); Settings keeps offering the
update, and the next release brings the banner back. Dev builds never
auto-check (their package.json version is static), but the manual check in
Settings works everywhere.

## Development

```bash
cd desktop
npm install
npm run dev        # electron-vite dev server + Electron window (needs a display)
```

## Build, typecheck, test, package

```bash
npm run typecheck  # tsc --noEmit over main, preload, shared, and renderer
npm run build      # typecheck + electron-vite production build into out/
npm test           # vitest unit tests for the data/install modules (headless)
npm run dist       # package installers into release/ (DMG+zip on macOS, NSIS on Windows)
npm run dist:dir   # unpacked app for a quick smoke test
```

Packaging is unsigned for now (no notarization/signing identities configured).

## Architecture

- **All Node-side data access lives in `src/main/agentfield.ts`** (registry
  parsing, control-plane HTTP probes, badge derivation, executions/metrics
  fetch, snapshot composition) and **installs in `src/main/installer.ts`**
  (spawns `af install`, sanitizes spinner/ANSI output, accepts curated catalog
  names from the renderer plus — on the one Install-from-repository channel —
  a raw source that `parseRepoSource` validates to a github.com https URL).
  Neither imports Electron, so both are unit-tested directly with Vitest.
- **Secure Electron layout:** the renderer runs with `contextIsolation: true`,
  `nodeIntegration: false`, `sandbox: true`. The preload
  (`src/preload/index.ts`) exposes a small typed API via `contextBridge`.
- **Shared IPC types** live in `src/shared/types.ts`; the install catalog in
  `src/shared/catalog.ts`; deep-link parsing in `src/shared/deeplink.ts`.
- **Tray presentation logic** (state/labels/glyph selection) is pure in
  `src/main/tray-model.ts` (unit-tested); the Electron glue is `src/main/tray.ts`.
- **Mac-first chrome** in `src/main/index.ts`: `titleBarStyle: hiddenInset` +
  sidebar vibrancy on macOS (minimal app menu since macOS needs one for
  Cmd+Q/copy-paste), hidden titlebar with native control overlay on Windows,
  no menu bar anywhere else. The sidebar and view header are draggable
  regions.

## Current limitations

- Control plane URL is hard-coded to `http://localhost:8080` (not configurable yet).
- No stop control for the control plane itself yet (the app only starts it).
- macOS/Linux shell PATH setup stays with the curl installer — the app only
  provisions `~/.agentfield/bin` there (absolute path always works).
- The macOS `af-tray` companion installs its own `~/Applications/AgentField.app`
  wrapper — a second bundle named "AgentField" distinct from this desktop app in
  `/Applications`. Harmless (the tray bundle is `LSUIElement`, menu-bar only) but
  a naming collision; disambiguating it is a separate, out-of-scope decision.
- The registry is read directly from `~/.agentfield/installed.yaml`; once
  `af list -o json` lands, the app should shell out to the CLI instead (see
  the `TODO(af-cli)` seam in `src/main/agentfield.ts`).
- Packaging has no signing identity. Builds without one get a valid **ad-hoc**
  signature (`scripts/macos-adhoc-sign.mjs`, afterPack) — electron-builder
  would otherwise leave a resource-less linker signature that codesign/spctl
  reject as invalid. Ad-hoc is enough to run locally-built apps normally;
  *downloaded* builds still carry quarantine and need right-click-Open or
  `xattr -dr com.apple.quarantine` until real signing/notarization lands.
