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
- **Install** — a curated, hard-coded catalog (`src/shared/catalog.ts`).
  Installing shells out to `af install <source>` and streams progress lines
  into the row; the af CLI stays the single contract for installs. Entries
  are keyed by the node's manifest name so installed state is detected. The
  hard-coded list is the pre-marketplace seam — replace with a remote catalog
  fetch when registry search lands.

The renderer polls a single snapshot over IPC every 5 seconds.

The control-plane probe only trusts `/health` responses that look like
AgentField's payload — an unrelated service on port 8080 renders as
"Port in use", never as a running control plane.

## Prerequisites

- Node.js 20+ (developed on Node 22)
- An AgentField control plane on `http://localhost:8080` (optional — the app
  degrades gracefully when it is not running)
- The `af` CLI on PATH for the Install view (the app tells you if it's missing)

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
  (spawns `af install`, sanitizes spinner/ANSI output, only accepts catalog
  names — never raw sources — from the renderer). Neither imports Electron,
  so both are unit-tested directly with Vitest.
- **Secure Electron layout:** the renderer runs with `contextIsolation: true`,
  `nodeIntegration: false`, `sandbox: true`. The preload
  (`src/preload/index.ts`) exposes a small typed API via `contextBridge`.
- **Shared IPC types** live in `src/shared/types.ts`; the install catalog in
  `src/shared/catalog.ts`.
- **Mac-first chrome** in `src/main/index.ts`: `titleBarStyle: hiddenInset` +
  sidebar vibrancy on macOS (minimal app menu since macOS needs one for
  Cmd+Q/copy-paste), hidden titlebar with native control overlay on Windows,
  no menu bar anywhere else. The sidebar and view header are draggable
  regions.

## Current limitations

- Control plane URL is hard-coded to `http://localhost:8080` (not configurable yet).
- No start/stop controls yet — installs only (run/stop are next).
- The registry is read directly from `~/.agentfield/installed.yaml`; once
  `af list -o json` lands, the app should shell out to the CLI instead (see
  the `TODO(af-cli)` seam in `src/main/agentfield.ts`).
- macOS chrome (traffic-light inset, vibrancy) is implemented per platform
  guards but has only been exercised on Windows so far — needs one smoke run
  on a real Mac.
- Packaging is unsigned; add signing/notarization before distribution.
