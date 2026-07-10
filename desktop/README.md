# AgentField Desktop

A read-only v1 desktop dashboard for AgentField, in the spirit of Docker Desktop:
one window that shows the health of the local control plane and the agent nodes
installed on this machine.

What it shows:

- **Control plane health** — polls `GET http://localhost:8080/health` and renders
  Running (healthy), Reachable (unhealthy), or Not reachable.
- **Installed agent nodes** — reads `~/.agentfield/installed.yaml` (name, version,
  language, port, PID) and derives a status badge per agent by cross-checking the
  registry against the control plane's `GET /api/v1/nodes` view:
  - `running` — registry says running and the control plane sees the node
  - `stopped` — registry says stopped and the control plane does not see it
  - `unknown` — registry and control plane disagree (stale registry / conflict),
    or the registry status is unrecognized

The renderer polls a single snapshot over IPC every 5 seconds, with a manual
Refresh button.

## Prerequisites

- Node.js 20+ (developed on Node 22)
- An AgentField control plane on `http://localhost:8080` (optional — the app
  degrades gracefully when it is not running)
- Optionally, agents installed via `af install ...` (populates
  `~/.agentfield/installed.yaml`)

## Development

```bash
cd desktop
npm install
npm run dev        # electron-vite dev server + Electron window (needs a display)
```

## Build, typecheck, test

```bash
npm run typecheck  # tsc --noEmit over main, preload, shared, and renderer
npm run build      # typecheck + electron-vite production build into out/
npm test           # vitest unit tests for the data-access module (headless)
```

## Architecture

- **All Node-side data access lives in one module:** `src/main/agentfield.ts`
  (registry parsing, control-plane HTTP probes, badge derivation, snapshot
  composition). It has no Electron imports, so it is unit-tested directly with
  Vitest.
- **Secure Electron layout:** the renderer runs with `contextIsolation: true`,
  `nodeIntegration: false`, `sandbox: true`. The preload
  (`src/preload/index.ts`) exposes exactly one method — `window.agentfield.getSnapshot()`
  — via `contextBridge`/`ipcRenderer.invoke`.
- **Shared IPC types** live in `src/shared/types.ts` and are imported type-only
  by main, preload, and renderer.
- Standard electron-vite project layout: `src/main`, `src/preload`,
  `src/renderer` (with `src/renderer/index.html`), built into `out/`.

## Current limitations

- **Read-only** — no install/start/stop actions; it only observes.
- Control plane URL is hard-coded to `http://localhost:8080` (not configurable yet).
- The registry is read directly from `~/.agentfield/installed.yaml`; once
  `af list -o json` lands, the app should shell out to the CLI instead so the CLI
  stays the single source of truth for registry parsing (see the `TODO(af-cli)`
  seam in `src/main/agentfield.ts`).
- Not packaged — no electron-builder / installer targets yet.
- Developed headless (WSL); the GUI is untested on Windows/macOS so far.
