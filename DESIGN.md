# AgentField Desktop — Design Language & UX Spec

> **Status:** Handoff-ready design direction (2026-07-22)
> **Surface:** `desktop/` Electron app (renderer + chrome)
> **Register:** product (design serves the task)
> **Audience:** Developers comfortable with GitHub; not infra experts
> **North-star products:** Linear · Raycast · OpenAI Codex desktop — quiet, dense, instant, trustworthy
> **Companion docs:** `PRODUCT.md` (strategy), this file (visual + UX + IA)

This document is the single source of truth for implementing the desktop redesign. Smaller implementation agents should implement against it without re-litigating taste.

---

## 0. Executive verdict

**Current score: 23 / 40 (Acceptable — significant UX overhaul needed before users are happy).**

The app is competent engineering wrapped in a generic Mac-system Electron shell. It does not yet serve its two hero jobs:

1. **Install agent nodes effortlessly** (especially paste-any-GitHub-repo)
2. **See what’s running and what it costs** (tokens / $ — already available via control plane `GET /api/ui/v1/usage/stats`)

Until those are front-and-center, polish is rearranging furniture in the wrong room.

**AI-slop verdict:** Mild yes. Safe Apple-blue accent, uppercase micro-labels, pill buttons, hairline panels, hero-metric tiles. Reads as “default Electron admin,” not Codex/Linear.

---

## 1. Product mental model (must ship in the UI)

```
Coding agent / product  →  AgentField control plane (sub-harness)
                              ↓ routes calls
                         Local agent nodes (from GitHub repos)
                              ↓ burn tokens on (often cheaper) models
                         Usage: tokens + $ attributed per agent / run
```

User sentences the UI must answer in ≤3 seconds:

| Moment | Question | Answer lives in |
|--------|----------|-----------------|
| First open, empty | “How do I get an agent?” | Install (default when 0 agents) |
| Daily glance | “Is my sub-harness up?” | Home status strip |
| After work | “What ran and what did it cost?” | Activity & Usage |
| Budget check | “Which client (Claude Code / Codex / my product) spent what?” | Home callers card + Activity by-harness |
| Wiring up | “How do my coding agents / products reach this?” | Home **Connect clients** card |
| Broken start | “Why won’t this agent start?” | Agents → Needs keys (inline) |
| New node from friend | “I have a GitHub URL…” | Install hero paste field |

---

## 2. Information architecture (nav redesign)

### 2.1 Nav v2 (4 items — Install merges into Agents)

> **Revision 2026-07-22 (v2):** “Install” is an *action on the agent library*, not a place. Two sibling nav items (Install / Agents) forced users to understand our internal split. Agents is now one marketplace-style surface: installed agents are the content; adding more is a `+` action on that surface.

| Order | ID | Label | Role |
|------:|----|-------|------|
| 1 | `home` | **Home** | At-a-glance trust (rename from Dashboard) |
| 2 | `agents` | **Agents** | Library + marketplace: installed nodes, `+ Add agent` (paste GitHub URL + featured grid), inline Keys |
| 3 | `activity` | **Activity** | Live runs + history at scale + **usage/cost** (hero job #2) |
| 4 | `settings` | **Settings** | Set-and-forget + App/CLI updates + All secrets |

**Remove top-level Secrets.** Same encrypted store already editable via Agents → Keys. “All secrets” is a Settings section. Deep link `agentfield://secrets` → Settings.

**Deep-link compatibility:** `install` stays a valid `View` (tray, docs, web UI mint these links). The renderer maps `install` → Agents with add-mode open. Do not remove it from `VIEWS`.

### 2.2 Default route logic (LOCKED)

Always start cold launches on **Home** when agents exist — do **not** remember last view. Empty library is the only override:

```
if agents.length === 0 → Agents (add-mode open — the marketplace IS the empty state)
else → Home
  (if !controlPlane.healthy, Home shows primary “Start control plane” CTA)
```

Deep links stay: `agentfield://home|install|agents|activity|settings`
Migrate: `dashboard` → `home`, `secrets` → `settings`, `install` → `agents` (add-mode).

### 2.3 View responsibilities (one job each)

#### Home
Section order (v2 — attention before plumbing):
1. Control-plane status (plain English + Start button when down)
2. 3–4 metrics: Agents running · Executing now · **Spend today** · Success rate (or Tokens today if cost null)
3. **Needs attention** strip when non-empty (agents needing keys, unknown badges, CP port conflict) — directly under metrics, not below the fold
4. Recent activity (3 rows) → “See all” → Activity
5. **Connect clients** card (see §4.10): endpoint + skills + callers line — reference info sits last
- Empty (0 agents): single CTA “Browse agents” → Agents add-mode

#### Agents (library + marketplace — one view, two modes)
**Library mode (default when ≥1 installed):**
- Dense list of installed nodes: status · Start/Stop · Keys (expand) · overflow for Restart / Web UI / Uninstall
- View-header action: **`+ Add agent`** (primary affordance for growth; always visible)

**Add mode (via `+`, via `agentfield://install`, or automatically when 0 installed):**
1. **Paste GitHub URL** hero at top (echo + validation + run-code disclosure, §4.4)
2. **Featured grid** below — browsable marketplace cards (§4.11), not a buried list
3. Streaming install progress on the active card / hero
4. When the library is non-empty, add mode shows above/instead of the list with a quiet “Back to installed (N)” affordance

- No separate Secrets nav

#### Activity (designed for thousands of runs — see §4.12)
- Quiet summary strip: `N running · M failed today · K runs (24h)` — inline text, not tiles
- Dense single-line rows (~34px), time-grouped: **Running** → **Today** → **Yesterday** → dates
- Filter chips: All / Running / Succeeded / Failed · agent filter when >1 agent
- Client-side pagination: render 50, “Show more” increments
- Columns/meta: status glyph · reasoner · agent · time · duration · **tokens · cost** (omit per-row tokens/$ when the execution usage API has nothing; never invent zeros)
- Row click or ↗ opens web UI run detail (acceptable for v1)

#### Settings (v2 organization — fewer, denser sections)
Order and grouping (each a `.subhead` + one panel):
1. **General** — Open at login · Autostart server · Coding-agent skills · Tray (macOS)
2. **Agents on startup** — per-agent toggles
3. **Updates** — one panel, two rows: App update row + CLI row (merge the two former sections)
4. **All keys** — former Secrets panel
- Advanced: (future) control plane URL — out of scope unless easy

---

## 3. Design language

### 3.1 Personality

**Calm · Precise · Trustworthy.** Quiet infrastructure. The tool disappears into the task. Density over decoration. Feedback over flourish.

**Not:** enterprise admin chrome, SaaS marketing gloss, purple-glow AI aesthetics, cream/terracotta AI defaults.

### 3.2 Color strategy — Restrained

One accent ≤10% of surface. Semantic status colors earn their keep. Prefer **dark-first comfort with OS light/dark**, but give dark mode more craft than `#1e1e20` flat gray.

#### Token system (OKLCH — replace hex CSS vars)

```css
:root {
  /* Surfaces — cool neutral, chroma toward accent hue (~250), NOT warm cream */
  --bg:            oklch(0.97 0.005 250);
  --bg-elevated:   oklch(1 0 0);
  --sidebar:       oklch(0.95 0.008 250);
  --panel:         oklch(1 0 0 / 0.72);
  --hairline:      oklch(0.2 0.01 250 / 0.1);

  /* Ink */
  --text:          oklch(0.22 0.02 250);
  --text-secondary:oklch(0.45 0.015 250);
  --text-tertiary: oklch(0.58 0.01 250);

  /* Brand accent — LOCKED: gold/amber (§3.3). Never Apple #0071e3. */
  --accent:        /* see §3.3 */;
  --accent-pressed:/* darker */;
  --accent-soft:   /* accent @ 12% */;

  /* Semantic */
  --ok:            oklch(0.62 0.17 145);
  --warn:          oklch(0.72 0.14 85);
  --danger:        oklch(0.58 0.2 25);
  --neutral:       oklch(0.6 0.01 250);

  /* Interactive */
  --nav-active:    oklch(0.2 0.01 250 / 0.07);
  --pill:          oklch(0.2 0.01 250 / 0.05);
  --focus-ring:    var(--accent);
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg:            oklch(0.18 0.01 250);
    --bg-elevated:   oklch(0.22 0.012 250);
    --sidebar:       oklch(0.16 0.012 250);
    --panel:         oklch(1 0 0 / 0.04);
    --hairline:      oklch(1 0 0 / 0.1);
    --text:          oklch(0.95 0.005 250);
    --text-secondary:oklch(0.72 0.01 250);
    --text-tertiary: oklch(0.55 0.01 250);
    --nav-active:    oklch(1 0 0 / 0.09);
    --pill:          oklch(1 0 0 / 0.07);
  }
}
```

### 3.3 Accent (LOCKED — gold / amber)

**Decided: soft gold / amber**, aligned with the tray active `•af` glyph (“gold = live”). Slate-teal was considered and rejected. Do not ship Apple `#0071e3`.

```css
--accent:         oklch(0.78 0.12 85);   /* light: deepen if contrast fails on white */
--accent-pressed: oklch(0.68 0.13 85);
--accent-soft:    oklch(0.78 0.12 85 / 0.14);
--accent-ink:     oklch(0.28 0.06 85);   /* text on soft gold chips */
```

Dark mode: bump chroma slightly; primary buttons use gold fill + near-black label (or gold outline + gold text) to hit WCAG AA.

### 3.4 Typography

**One family.** System stack is fine for product UI (SF Pro / Segoe UI Variable) — matches native Mac furniture.

```
font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text",
             "Segoe UI Variable", "Segoe UI", system-ui, sans-serif;
```

**Mono** (keys, ports, URLs, reasoner ids):
`ui-monospace, "SF Mono", Menlo, Consolas, monospace`

| Role | Size | Weight | Notes |
|------|------|--------|-------|
| App brand (sidebar) | 13–14 | 650–700 | Track −0.01em; include mark, not text-only |
| View title | 18–20 | 650 | Track −0.02em |
| Section label | 12 | 550 | **Sentence case**, not ALL CAPS tracked eyebrows |
| Body / row title | 13 | 500 | |
| Secondary | 12 | 400 | `--text-secondary` |
| Metric value | 24–28 | 650 | Tabular nums |
| Chip / badge | 11 | 600 | |

**Ban:** Tiny uppercase tracked labels on every section (`AGENTS RUNNING`, `RECENT ACTIVITY`). Use sentence-case section titles: “Recent activity”, “Featured nodes”.

### 3.5 Spacing & radius

| Token | Value |
|-------|-------|
| Space unit | 4px |
| Sidebar width | 220px (was 200) |
| Content max | 880px (was 820) |
| Content pad | 24–28px horizontal, 20–28px bottom |
| Panel radius | **10px** (slightly tighter than 12) |
| Control radius | **8px** for buttons/inputs; **999** only for true pills (status chips) |
| Row height (interactive lists: Agents, Settings) | ~44–48px |
| Row height (dense feeds: Activity) | **32–36px**, single line |
| Gap between major sections | **24px, consistently** — one `--space-6` rhythm, not per-view ad-hoc margins |
| Within-panel padding | 16px horizontal, consistently |
| Section title → panel gap | 8px via one `.subhead` pattern (kill per-view `margin` hacks on `.section-title`) |

#### Layout grid (v2 — every view, no exceptions)

All five views render inside the SAME content frame; a view may not invent its own margins:

```
.view-body  = the frame: max-width 880 · pad 12px top · 28px x · 28px bottom · 24px section gap
  └─ each direct child is a "section unit": lede | subhead+panel | panel | filter bar | strip
```

- **Rule 1:** the first element of every view starts at the same y. No per-view top margins (Activity's filter bar, Home's tiles, Settings' lede all align).
- **Rule 2:** anything visually inside a panel aligns to the panel's 16px padding column — row content, group headers, empty states, progress lines.
- **Rule 3:** free-standing elements between panels (filter bars, summary strips, footnotes) get a 4px left optical inset to align with panel text, never arbitrary values.
- **Rule 4:** vertical rhythm comes ONLY from `.view-body` gap (24px) and `.subhead` (8px). If a view needs a different gap, it's a design change, not a CSS patch.

### 3.6 Elevation & shell differentiation (v2)

No multi-layer shadows. Panels: hairline border + optional `0 1px 0` inset highlight in dark.

**Sidebar ≠ canvas.** The nav rail and the content canvas must read as two surfaces:

- **macOS:** sidebar keeps vibrancy (transparent over system blur — the “glossy” material). The main column gets a **solid `--bg` canvas** so the blur never bleeds under content.
- **All platforms:** deepen the sidebar tint so the seam is visible even without vibrancy:
  - light: `--sidebar: oklch(0.93 0.01 250)` (was 0.95 — too close to bg 0.97)
  - dark: `--sidebar: oklch(0.14 0.012 250)` with `--bg: oklch(0.19 0.01 250)`
- Keep the 1px `--hairline` seam between rail and canvas.
- Do **not** spread glass/blur to panels or content — vibrancy is the sidebar’s material only (glassmorphism-as-default is banned).

### 3.7 Iconography

**Library:** Lucide (or Phosphor Regular) — single stroke set, 16px nav / 14px inline.
**Do not** invent ad-hoc path strings per view; one icon component wrapping a shared set.

| View | Icon |
|------|------|
| Home | `LayoutDashboard` or `House` |
| Install | `Download` or `PackagePlus` |
| Agents | `Bot` or `Boxes` |
| Activity | `Activity` or `ChartNoAxesColumn` |
| Settings | `Settings` |

Status: filled dots **with** text label (never color alone). Live: soft pulse only if `prefers-reduced-motion: no-preference`.

Brand mark: keep outlined `•af` paths from web UI / tray — render 16–18px beside “AgentField” in sidebar.

---

## 4. Component patterns

### 4.1 Buttons (one vocabulary)

| Variant | Use |
|---------|-----|
| `primary` | One primary action per region (Install, Start, Set key) |
| `secondary` | Neutral fill (`--pill`) |
| `ghost` | Tertiary / Cancel |
| `danger` | Uninstall, Revoke confirm |
| `icon` | Overflow `⋯`, open-external |

**Shape:** 8px radius, height 28–32px, padding 0 12px, font 12–13 / 600.
**Not** every button a full pill (`border-radius: 999`). Reserve pills for status chips.

Loading: swap label → “Installing…” / spinner at 12px; disable only that row’s conflicting actions, **not the entire Agents list**.

### 4.2 Status chip

```
[● Running]  [○ Stopped]  [!] Needs keys  [?] Unknown
```

Text + glyph. Color supports, never replaces. Tooltip on Unknown: “Registry says running, but the control plane doesn’t see this node. Try Restart.”

### 4.3 Agent row

```
[status]  Name                         [Needs keys?]
          One-line description · :port
                               [Start | Stop]  [Keys]  [⋯]
```

- Primary action always visible (Start or Stop)
- Keys visible when env declared
- Restart, Open in Web UI, Uninstall → overflow menu
- Expanding Keys reveals EnvEditor (existing logic)

### 4.4 Install hero (new)

Full-width panel at top of Install:

- Label: “Install from GitHub”
- Input: `https://github.com/org/repo` or `…/repo//subdir`
- Helper: “Any public repo with an `agentfield-package.yaml`. Use `//subdir` for multi-node repos.”
- Primary button: Install
- Progress lines stream under the field

**Trust cues (required — this executes third-party code):**

- As soon as a valid URL is pasted, echo the parsed target as a confirmation line: `Installs agent-field/SWE-AF` with a link-out icon to the repo (recognition before commitment; also catches paste typos)
- Invalid shapes get inline validation before spawn (“Only `https://github.com/…` sources are supported”), not a post-hoc CLI error
- One-line disclosure under the field: “Installing downloads and runs this repo’s agent code on your machine.”
- Curated entries show their `source` repo as secondary meta so curated ≠ anonymous

Curated list below titled “Featured nodes” with language chip (Python / Go) and source repo.

### 4.5 Metric tiles (Home)

Keep ≤4 tiles. Prefer:

1. Agents running (`2 of 4`)
2. Executing now
3. **Spend today** (`$1.23` or `—` / “No usage yet”)
4. Success rate *or* Tokens today

Avoid the “hero-metric SaaS” look: no gradient accents, no identical icon cards. Quiet panels, sentence-case labels.

### 4.6 Activity row + usage

```
[live●|✓|✕]  plan_and_ship
             swe-planner · 1.4k tok · $0.02 · 42s     [↗]
```

When usage API returns null cost: show tokens only. When 404 (old CP): hide cost columns gracefully (“Usage needs a newer control plane”).

Data source: `GET /api/ui/v1/usage/stats?window=24h` (already used by `af-tray`). Also surface per-run totals if `GetExecutionUsageTotals` is exposable to UI; otherwise roll up from stats `by_agent` on Home and show per-run when available.

### 4.7 Empty states (teach, don’t shrug)

| State | Copy direction |
|-------|----------------|
| No agents | “Install your first agent node from GitHub. Coding agents on this machine can then call it through AgentField.” + CTA |
| CP down | “Control plane isn’t running.” + **Start now** button (not “run `af server`”) |
| No activity | “Runs appear here when an agent is called — from Claude Code, Codex, or any client.” |
| No secrets | “Keys you set on an agent appear here. Values stay encrypted and are never shown again.” |

### 4.8 Callouts & errors

- Plain language, next action attached
- Destructive confirms stay inline (current pattern is good)
- Never contradict Settings: if autostart exists, offer Start in-app

### 4.9 Update banner (LOCKED — keep on all views)

Show the thin top banner across all views when an update is available. Dismiss hides it for that version only (existing `dismissedUpdateVersion` behavior). Settings always keeps the durable check / install controls. Do not Settings-only the banner.

### 4.10 Connect clients card (new — the sub-harness front door)

The product’s whole point is other harnesses calling this machine. Today that story is invisible (a “skills” toggle buried in Settings). Make it a first-class Home card:

```
Connect clients
  Endpoint   http://localhost:8080          [copy]
  Skills     Installed in Claude Code, Codex     (via `af skill install`)
  Try it     af agent discover — or ask your coding agent to list agents
```

- Endpoint line: monospace, one-click copy
- Skills line: reflects the existing `installSkills` state + detected coding agents (data the main process already has from skillkit); if skills are off, show “Off — enable in Settings” link
- Keep it to 3 quiet lines; no wizard, no modal
- This card is also the natural future seam for API tokens / remote access when those land

### 4.11 Agents library + marketplace (v2 — replaces separate Install view)

One view, two modes (see §2.3). Layout:

```
View header:  Agents                              [+ Add agent]   ← primary, header action
──────────────────────────────────────────────────────────────
Library mode:
  [panel] installed rows (status · name · needs-keys? · desc · :port · Start/Stop · Keys · ⋯)

Add mode (0 agents, + clicked, or agentfield://install):
  [install hero — paste GitHub URL]                (§4.4 unchanged)
  Featured                                          ← sentence-case section title
  [card grid]  repeat(auto-fill, minmax(240px, 1fr)), gap 12px
```

**Featured card anatomy** (marketplace affordance — browsable, not a settings list):

```
┌──────────────────────────────┐
│ name             [Python]    │  ← row-title + language chip
│ Two-line description,        │
│ clamped.                     │
│ org/repo ↗          [Install]│  ← mono source (trust cue) + action
└──────────────────────────────┘
```

- Installed cards: `✓ Installed` state; Update / Uninstall live in a card overflow `⋯` (never a wall of buttons)
- Install progress streams inside the active card (existing progress-line pattern)
- Cards use hairline border + `--panel`; no icons-for-decoration, no identical icon+heading+text slop — the description and source repo ARE the content
- Empty-library state = add mode itself. Never “go open Install from the sidebar.”

### 4.12 Activity at scale (v2 — dense feed, not cards)

Assume **hundreds–thousands of runs**; agents fan out in parallel. The unit is a feed line, not a card.

```
3 running · 12 failed today · 412 runs (24h)          ← summary strip (plain text, tabular nums)
[All] [Running] [Succeeded] [Failed]   [agent ▾]      ← chips + agent select (when >1 agent)

Running
● plan_and_ship        swe-planner      started 2:41 PM            ↗
Today
✓ review_diff          swe-reviewer     2:38 PM · 41s · 1.4k tok · $0.02
✕ plan_and_ship        swe-planner      2:31 PM · 12s
Yesterday
…
[Show more (350 remaining)]
```

- Row height 32–36px, one line; error message truncated with `title` tooltip, full detail via ↗
- Time-group headers (Running / Today / Yesterday / date) are quiet `.section-title`s inside the feed, sticky optional
- Status = glyph + color (✓ ✕ ● ⊘ ⏱), no full badge pills per row at this density; label appears in the glyph’s `title` and group context
- Render cap 50 rows; “Show more” appends 50 (client-side; no virtualization needed at v1 volumes)
- Right-aligned meta column in mono/tabular nums so hundreds of rows scan vertically
- Home “Recent activity” keeps the roomier 46px rows (3 rows max — different job)

### 4.13 Growth & community mechanics (v2 — OSS conversion, done tastefully)

AgentField is open source; stars, docs traffic, and issue reports are the conversion funnel. The register stays **calm/precise/trustworthy** — so every ask follows four principles:

1. **Earn the moment.** Ask only right after the product delivered value (a successful install, a streak of successful runs) — never on first launch, never while something is broken or installing.
2. **One ask at a time.** Star prompt never co-exists with the update banner; update banner wins.
3. **Permanently dismissible.** Every prompt has “Don’t ask again” persisted in desktop settings. “Later” snoozes 7 days.
4. **No modals.** Reuse the existing top-banner material (`.update-banner` pattern with accent-soft background). Nothing blocks the canvas.

#### Star prompt (milestone banner)

```
★ Enjoying AgentField? A star on GitHub helps other developers find it.
                     [Star on GitHub]  [Later]  [Don’t ask again]
```

- **Trigger:** earliest of (a) first successful agent install, (b) ≥5 succeeded runs visible in the snapshot. Suppress when: update banner visible · CP unhealthy · an install is in flight.
- **Persistence** (desktop settings, main process): `starPrompt: 'pending' | 'done'` + `starPromptSnoozedUntil: string | null`. “Star on GitHub” opens the repo and marks `done`; so does “Don’t ask again”.
- Max one impression per app session.

#### Settings → About section (durable, non-nagging home for links)

Last Settings section, one panel, three rows:
- **AgentField** `v{version}` — “Free & open source.” → button: **Star on GitHub**
- **Docs** — “Guides for installing and authoring agent nodes.” → **Open docs**
- **Found a bug?** — → **Report an issue**

Links: repo `https://github.com/Agent-Field/agentfield`, docs `…/tree/main/docs`, issues `…/issues/new`. Plain anchors / openExternal; no iframes, no live star counts (network chatter + broken-when-offline risk).

#### Contextual docs links (help where the question arises)

| Surface | Link text | Target |
|---------|-----------|--------|
| Install hero helper line | “authoring guide” | docs/installing-agent-nodes.md on GitHub |
| Connect clients card | “How clients connect” | docs (skills/integration doc) |
| Activity empty state | “See how agents get called” | same integration doc |

Quiet inline links (`.link-button` style), sentence case, never a second CTA competing with the section’s primary action.

**Banned:** star-gating features, popup-on-launch, recurring nags after “don’t ask again”, fake urgency, notification badges on nav items for marketing purposes.

### 4.15 Loading states — skeletons, not “Loading…” (v2)

Text placeholders (“Loading…”, “Checking…”) make the app feel like a terminal. Replace with **skeleton rows that match the final layout**:

- Agents/Settings/Home lists: 2–3 skeleton rows (dot circle + two text bars at 40%/65% width) inside the real panel.
- Tiles: label bar + value bar.
- Shimmer: 1.4s linear sweep of a slightly brighter tint across the `--pill` base; `prefers-reduced-motion` → static tint, no sweep.
- Skeletons appear only when data is genuinely absent (first load); poll refreshes never flash skeletons over real content.

### 4.16 Empty-state dot fields (v2)

True no-data states use a shared **dot-field system** above a clear headline and one short supporting sentence. The field is a sparse grid of 4px `--accent` circles whose opacity forms a contextual motif. It is the same vocabulary on every page, with a different composition so the views remain distinct:

| Surface | Motif |
|---------|-------|
| Home empty | A broad flowing constellation; full hero scale and one “Browse agents” action |
| Agents library empty | Compact orbit; normally add-mode replaces this state |
| Activity empty | A pulse-line constellation; full hero scale with the integration-doc action |
| Settings startup / keys empty | Compact orbit composition |

- Plain dots only: no gradient mask, glow, blur, illustration, or decorative line art.
- Hero fields sit on the theme-tokenized `--empty-surface`, a single flat surface slightly deeper than the page in both themes; do not nest another card inside the panel.
- Ambient motion is deliberately slow and variant-specific: Home flows horizontally over 11–14s, Activity emits a center-out pulse over 13–16s, and configuration orbits over 15–18s. Brightness peaks stay below 0.7 and scale only shifts from 0.82→1. Treat the whole field as one moving element; reduced-motion → static motif.
- Hierarchy is always: motif → 15–20px semibold headline → smaller secondary copy (44–48ch max) → optional action.
- Dot fields only represent genuine absence; filtered zero-results use a plain inline message.

Home’s empty hero uses “Give your coding agent a specialist.” Its supporting copy begins “Install a sub-harness for the job you need done” and uses two intentional lines. The second line begins with a fixed 96px × 1.6em harness-name viewport (Claude Code → Codex → Cursor) followed by stable copy. Rotating words are absolutely layered and right-aligned inside that viewport, then animate only opacity plus a 4px internal offset. Every name therefore ends at the same point and neither line position nor height changes during the 2.4s rotation. Reduced-motion holds on Claude Code while the accessible label names all three. Below it, show a static, theme-aware monochrome compatibility rail of official Kimi, MiniMax, GLM, Qwen, DeepSeek, and Mistral marks sourced from `@lobehub/icons-static-svg`. A separate quiet line below the real provider marks reads “…or any open or closed model”; it has no invented logo or badge. The rail is width-constrained and wraps into centered rows instead of clipping. Do not auto-marquee the logos—the harness rotation and dot field already consume the region’s motion budget.

### 4.17 Keyboard ergonomics (v2 — promoted from P3)

Developers live on the keyboard:

- `⌘1–⌘4` (Ctrl on Win/Linux) switch views in nav order; matches the 4-item nav.
- `⌘R` manual snapshot refresh (poll continues regardless).
- `Esc` closes open overflow menus / add-mode (returns to library when non-empty).
- Focus order follows visual order; `:focus-visible` rings already specified (§7).
- Show shortcut hints in nav item tooltips (`title="Home ⌘1"`), not painted-on labels.

### 4.18 Windows/Linux parity note

macOS leans on vibrancy + af-tray; Win/Linux get solid surfaces + in-app tray. All components in this spec must render correctly on solid backgrounds — never depend on vibrancy for contrast (e.g. `--panel` translucency needs a solid fallback under `body:not([data-platform='darwin'])`).

---

## 5. Motion & micro-interactions (v2 — a real motion system)

**Energy:** low. Product motion = state, not choreography. The app should feel *physically settled* — Linear/Raycast-grade — not animated for its own sake.

### 5.1 Stack

- **Library:** `motion` (motion.dev — the modern successor to framer-motion). Import via `LazyMotion` + `domAnimation` features and `m.` components to keep the bundle small. Springs for anything that *moves*; the existing CSS `--ease-out` (`cubic-bezier(0.16, 1, 0.3, 1)`) for simple fades/tints.
- **Reduced motion is a first-class input:** use `useReducedMotion()` from motion in components AND keep the global CSS kill-switch. Fallback is always instant state change or plain crossfade.
- **Springs:** `{ type: 'spring', stiffness: 500, damping: 40 }` for UI furniture (fast, no overshoot felt). Never bounce/elastic.

### 5.2 Interaction inventory

| Interaction | Motion | Notes |
|-------------|--------|-------|
| Nav active | **Sliding highlight** — one shared element (`layoutId="nav-active"`) springs between nav items | The signature micro-interaction; icon+label color crossfade 120ms |
| View change | `AnimatePresence` crossfade 160ms + 4px rise | No slide-parade; exit and enter overlap slightly |
| Metric value change | **Count/roll ticker** ~300ms when a tile value changes after first paint | Tabular nums → zero layout shift; skip on initial load |
| New activity row arriving | Fade + −4px settle, **only for run IDs not seen in the previous poll** | Never stagger-animate the initial paint of a dense list — diff-only entrances |
| Live run indicator | Breathing dot (existing 1.6s pulse) + soft outer ring | Reduced-motion → solid dot |
| Keys / expander open | Height spring + opacity via `AnimatePresence` | Replaces the fake CSS max-height transition |
| Row hover | Background `--pill` 120ms + hairline brighten | |
| Featured card hover | `translateY(-1px)` + border-color toward `--accent-soft`, 120ms | Quiet lift, no shadow bloom |
| Button press | `scale(0.985)` 80ms on `:active` | Transform only |
| Switch | Thumb translate 150ms ease-out (existing) | |
| Copy action | Label crossfades to “Copied ✓” 150ms, back after 1.5s | |
| Install success | SVG checkmark **stroke-draw** ~400ms in the hero/card | Failure: single 2×2px horizontal shake, reduced-motion → none |
| Install progress lines | Per-line fade-in (existing) | |
| Banner enter/exit | 180ms opacity + 4px translateY, exit reversed via `AnimatePresence` | |
| Skeleton shimmer | 1.4s linear sweep on `--pill` tint | Reduced-motion → static tint (§4.15) |
| Empty dot field | Variant-specific staggered opacity/scale flow over 11–18s | Decorative but restrained; peak opacity <0.7, no glow/gradient; reduced-motion → static motif (§4.16) |

### 5.3 Rules

- Animate **transform and opacity only** (height allowed inside motion-managed expanders).
- Entrances are for content that *arrives after mount* — initial paints render settled. No orchestrated page-load stagger, ever.
- One moving thing at a time per region; motion never delays data visibility.
- No decorative blur/glass animation; no scroll-linked effects.

```css
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
  }
}
```

---

## 6. Copy & terminology

| Avoid | Prefer |
|-------|--------|
| Control plane (naked) | “AgentField server” or “control plane” once, then “server” |
| `af server` as only fix | “Start AgentField server” button |
| Secrets (nav) | Keys (on agent) / All keys (Settings) |
| Dashboard | Home |
| Executing now (ok) | “Runs in flight” also fine |
| Registry names only | Show friendly title if manifest has one; monospace name as secondary |
| Unknown (no tip) | Unknown + tooltip/explanation |

Tone: short, confident, GitHub-native. Assume user knows repos and tokens; do not assume they know AgentField internals.

---

## 7. Accessibility (non-negotiable)

- WCAG AA contrast on all text (fix tertiary gray on tinted panels)
- `aria-current="page"` on active nav
- Status never color-only
- Focus rings visible (`:focus-visible` with `--accent`)
- Switches named by row title (`aria-labelledby`)
- Allow text selection on copyable content (errors, URLs, keys names) — drop blanket `user-select: none` or scope it to chrome only
- Reduced motion (above)
- Hover-only “↗” → always visible on focus, or show on row focus-within
- Target sizes ≥ 28px height minimum; prefer 32px

---

## 8. Layout chrome (Mac-first)

Keep:

- `titleBarStyle: hiddenInset` + sidebar vibrancy (darwin)
- Drag regions on sidebar + view header
- Traffic-light padding
- Win32 caption overlay inset on header

Evolve:

- Sidebar brand: mark + AgentField + optional tiny version
- Footer status pill → **interactive** when not healthy (click → Start / Settings)
- View header: title left; optional contextual action right (e.g. Install view: nothing; Agents: “Install…” link)

---

## 9. Usage / cost integration (data already exists)

**Endpoint:** `GET /api/ui/v1/usage/stats?window=24h`
**Shape (from af-tray):** totals `{ cost_usd, input_tokens, output_tokens, total_tokens, executions_with_usage }`, groups `by_model`, `by_provider`, `by_agent`, `by_harness`.

### Desktop snapshot additions

Extend `AgentFieldSnapshot` (or sibling fetch):

```ts
usage: {
  window: '24h'
  costUsd: number | null
  totalTokens: number
  byAgent: Array<{ key: string; costUsd: number | null; totalTokens: number }>
  /** Who called: claude-code, codex, my-product… — the sub-harness ledger. */
  byHarness: Array<{ key: string; costUsd: number | null; totalTokens: number }>
} | null  // null = CP down or endpoint absent
```

### UI placement

| Surface | What (v1 LOCKED) |
|---------|------|
| Home tile | Spend today / Tokens today (required) |
| Home callers line | **by_harness** roll-up: “claude-code $0.84 · codex $0.31” (this is the offloading story) |
| Home “by agent” optional 2-line | Top spender |
| Activity rows | **tokens · $ per row when API allows**; omit those fields (not fake `0`) when missing; Home totals remain either way |
| Agents row meta | 24h tokens (secondary, nice-to-have) |

Graceful degradation mirrors af-tray’s tested contract (`cmd/af-tray/usage_test.go`): 404 → older CP, hide usage entirely; 401/403 → “Usage needs the admin key”; null `cost_usd` → tokens only. Never error the whole Home.

---

## 10. Priority implementation backlog

Ordered for handoff to smaller agents. Each item is independently shippable.

### P0 — Structure & hero jobs

1. **IA nav regroup** — Home / Install / Agents / Activity / Settings; remove Secrets top-level; deep-link migration
2. **Install hero** — Paste GitHub URL above curated catalog; parsed-target echo + inline validation + run-code disclosure (§4.4)
3. **Empty-state routing** — 0 agents → Install; CP down → Start CTA on Home
4. **Usage on Home + Activity** — wire `/api/ui/v1/usage/stats`; Spend/Tokens tile; **by-harness callers line**; Activity meta

### P1 — Trust & recovery

5. **In-app Start control plane** — replace `af server` callout
6. **Connect clients card on Home** (§4.10) — endpoint copy + skills status
7. **Unknown badge explanation** + Restart affordance
8. **Agent row overflow menu** — reduce button density
9. **Copy/terminology pass** — Home, Keys, server language

### P2 — Design language

10. **Token CSS refresh** — OKLCH neutrals + gold/amber accent; kill Apple blue; solid-surface fallback off-macOS (§4.18)
11. **Typography** — sentence-case section titles; kill uppercase eyebrows
12. **Button radius / icon set** — Lucide; consistent variants
13. **Motion tokens** + `prefers-reduced-motion`

### V2 — IA consolidation & scale (2026-07-22 revision — do these next)

V2-1. **Merge Install into Agents** (§2.1, §4.11) — 4-item nav; `+ Add agent` header action; add-mode = hero + featured grid; `install` deep link → Agents add-mode; 0 agents → Agents add-mode
V2-2. **Featured marketplace grid** (§4.11) — cards `minmax(240px,1fr)`; installed = ✓ + overflow; progress in-card
V2-3. **Activity dense feed** (§4.12) — summary strip, 32–36px rows, time groups, Succeeded filter, agent filter, Show more
V2-4. **Shell differentiation** (§3.6) — solid canvas on macOS main column; deeper sidebar tints both themes
V2-5. **Spacing rhythm pass** (§3.5) — one 24px section gap; one `.subhead` pattern; kill ad-hoc `.section-title` margin hacks
V2-6. **Settings regroup** (§2.3) — merge App updates + CLI into one Updates section; order General → Agents on startup → Updates → All keys
V2-7. **Home reorder** (§2.3) — attention strip above the fold; Connect clients last; empty CTA → Agents add-mode
V2-8. **Layout grid enforcement** (§3.5) — every view starts at the same y inside the same frame; no per-view margins; free-standing bars get the 4px optical inset
V2-9. **Growth mechanics** (§4.13) — milestone star banner with Later / Don’t-ask-again persistence; Settings About section (Star / Docs / Report issue); contextual docs links on Install hero, Connect clients, Activity empty state
V2-10. **Motion system** (§5) — adopt `motion` (LazyMotion + m.); sliding nav highlight; view crossfade via AnimatePresence; metric ticker; diff-only feed-row entrances; spring expanders; press scale; card hover lift; install success stroke-draw; reduced-motion first-class
V2-11. **Skeleton loading** (§4.15) — replace all “Loading…/Checking…” text with layout-matched shimmer skeletons
V2-12. **Empty-state dot fields** (§4.16) — shared plain-dot system with flow, pulse, and orbit compositions across true no-data states
V2-13. **Keyboard ergonomics** (§4.17) — ⌘1–4 views, ⌘R refresh, Esc closes menus/add-mode, tooltip hints

### P3 — Polish

14. Language chips + source repo on catalog (Python/Go)
15. Keyboard shortcuts: `1–5` views, `⌘R` refresh
16. Selectable copy for errors/URLs
17. Focus-visible + aria-current
18. Settings “All keys” section (moved Secrets)
19. Do not Settings-only the update banner (locked: keep on all views; dismiss per-version)

---

## 11. Explicit non-goals (this redesign)

- Replacing the control-plane web UI for deep run DAG inspection
- Full marketplace search UI (catalog stays curated + paste-repo until registry search lands)
- Configurable CP URL (unless trivial; README already notes hard-code)
- Stopping the control plane from the app (still start-only)
- Heavy charts / org-wide analytics (Enterprise territory per AGENTS.md)

---

## 12. Heuristic scores (baseline to beat)

| # | Heuristic | Score | Key issue |
|---|-----------|------:|-----------|
| 1 | Visibility of system status | 3 | No cost; Unknown opaque |
| 2 | Match system / real world | 2 | Infra jargon |
| 3 | User control and freedom | 3 | Global busy lock on agents |
| 4 | Consistency and standards | 2 | Keys vs Secrets split |
| 5 | Error prevention | 3 | Weak repo URL feedback |
| 6 | Recognition rather than recall | 2 | Internal names, no search |
| 7 | Flexibility and efficiency | 2 | No shortcuts / refresh |
| 8 | Aesthetic and minimalist design | 3 | Button-dense rows |
| 9 | Error recovery | 2 | CP callout vs autostart |
| 10 | Help and documentation | 1 | No onboarding |
| | **Total** | **23/40** | Acceptable → target **32+** after P0–P2 |

---

## 13. Acceptance criteria (definition of done)

A redesign pass is done when:

1. A new user with 0 agents lands on Install and can paste a GitHub URL as the first interactive element; a pasted URL echoes its parsed repo target before install.
2. Home answers “is it up?”, “what did it cost today?”, **and “who called it?”** without opening Settings or the web UI.
3. Home shows how clients connect (endpoint + skills status) in one card.
4. Secrets is not a top-level nav item; keys remain easy from Agents.
5. Accent is gold/amber (locked); no Apple-default blue.
6. No ALL-CAPS eyebrow section labels.
7. Reduced-motion kills pulse/spin.
8. Status has text, not only color.
9. Detector + manual a11y spot-check pass on Home / Install / Agents / Activity.

---

## 14. File map for implementers

| Area | Paths |
|------|-------|
| Nav / shell | `desktop/src/renderer/src/components/Sidebar.tsx`, `App.tsx` |
| Deep links | `desktop/src/shared/deeplink.ts` |
| Styles / tokens | `desktop/src/renderer/src/styles.css` |
| Home | `DashboardView.tsx` → rename conceptually to Home |
| Install | `InstallPanel.tsx` |
| Agents / keys | `AgentsPanel.tsx`, `EnvEditor.tsx` |
| Activity / usage | `ActivityPanel.tsx` + new usage fetch in `src/main/agentfield.ts` |
| Secrets fold-in | `SecretsPanel.tsx` → Settings section |
| Settings | `SettingsPanel.tsx` |
| Types | `desktop/src/shared/types.ts` |
| Usage API reference | `control-plane/cmd/af-tray/*usage*` · `control-plane/internal/handlers/ui/usage*.go` |

---

## 15. Locked product decisions

Confirmed 2026-07-22 — implementers must not re-litigate:

| # | Decision | Locked choice |
|---|----------|---------------|
| 1 | Accent | **Gold / amber** (tray-aligned). Slate-teal rejected. |
| 2 | Default nav when agents exist | **Always Home** on cold launch. Do not remember last view. Zero agents → Install. |
| 3 | Usage depth in v1 | **Home totals required** + **Activity per-row tokens/cost when the API provides them** (omit fields when absent). |
| 4 | Update banner | **Keep banner on all views**; dismiss is per-version; Settings keeps durable check. |
| 5 | Nav shape (v2) | **4 items** — Install merges into Agents as `+ Add agent` / add-mode. `install` stays a valid deep-link View mapped to Agents add-mode. |
| 6 | Activity density (v2) | Dense single-line feed with time groups + Show more; built for hundreds–thousands of runs. Home keeps roomy 3-row recent list. |
| 7 | Sidebar material (v2) | Vibrancy stays sidebar-only (macOS); main column solid canvas; deeper sidebar tint off-macOS. No panel glassmorphism. |
| 8 | Motion library (v2) | `motion` (motion.dev) with LazyMotion/m. — springs for moves, CSS ease-out for fades. Diff-only entrances; no page-load stagger; reduced-motion always honored. |
| 9 | Growth asks (v2) | Milestone-triggered star banner only (install success or 5+ succeeded runs); permanently dismissible; update banner always wins; no modals, no launch popups. |

---

*End of design handoff. Implement P0 first; visual language (P2) can parallelize once IA structure lands.*
