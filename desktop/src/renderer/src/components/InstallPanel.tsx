import { useEffect, useState } from 'react'
import { AnimatePresence, m, useReducedMotion } from 'motion/react'
import type { CatalogEntry, InstalledAgent } from '../../../shared/types'
import { installedSourceLabel } from '../../../shared/catalog'
import { COMMUNITY_LINKS } from './communityLinks'
import { MenuPopover } from './MenuPopover'
import { SkeletonRows } from './Skeleton'
import { updateWithExecutionConfirmation } from './AgentsPanel'

/**
 * Install-success checkmark (DESIGN.md §5.2): the path draws in ~400ms via
 * the CSS `check-draw` stroke-dashoffset animation. `pathLength=1`
 * normalizes the dash math; reduced motion renders it already drawn.
 */
function InstallCheck() {
  return (
    <svg
      className="check-draw"
      viewBox="0 0 16 16"
      width="13"
      height="13"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M3 8.5L6.5 12L13 4.5"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        pathLength="1"
      />
    </svg>
  )
}

// The Agents view's add-mode surface (DESIGN.md §4.11): paste-a-repo hero +
// featured marketplace card grid. Rendered by App inside the Agents view —
// there is no separate Install view anymore.
interface InstallPanelProps {
  installedAgents: InstalledAgent[]
  provisioningNames: string[]
  onInstalled: () => void
  /** Installed agents count — labels the "Back to installed (N)" affordance. */
  libraryCount: number
  /** Return to library mode; absent when the library is empty (add-mode IS the view). */
  onBackToLibrary?: () => void
}

type InstallPhase =
  | { state: 'idle' }
  | { state: 'installing'; name: string; progress: string; lines: string[] }
  | { state: 'done'; name: string; ok: boolean; message: string; lines: string[] }

/** Parsed GitHub install target (mirrors main/installer.ts parseRepoSource). */
export type ParsedRepo = {
  /** Normalized source passed to installFromSource. */
  normalized: string
  /** `owner/repo` without trailing `.git` (for the confirmation echo). */
  ownerRepo: string
  subdir: string | null
  /** Plain HTTPS repo URL suitable for a browser link (no `//subdir`). */
  href: string
}

// Sentinel `name` for the paste-a-repo flow so its progress/result routes to
// the hero and never collides with a catalog entry name.
const REPO_TARGET = '\0install-from-repo'
const MAX_PROGRESS_LINES = 12

const GITHUB_PREFIX = 'https://github.com/'
// owner and repo: alphanumerics, underscore, dot, dash — no leading dash.
const OWNER_REPO = /^[A-Za-z0-9_.][A-Za-z0-9_.-]*\/[A-Za-z0-9_.][A-Za-z0-9_.-]*$/
// //<subdir> selector: slash-separated segments, no leading `-` or `/`.
const SUBDIR = /^[A-Za-z0-9_.][A-Za-z0-9_./-]*$/

const INVALID_SOURCE_MSG = 'Only `https://github.com/…` sources are supported'

/**
 * Validate and normalize a pasted install source. Accepts ONLY
 * `https://github.com/<owner>/<repo>` optionally followed by `//<subdir>`.
 * Strips one trailing slash; keeps a trailing `.git` (af accepts it).
 * Returns null for anything else — client-side gate before spawn.
 */
export function parseRepoSource(input: string): ParsedRepo | null {
  const trimmed = input.trim()
  if (!trimmed) return null
  if (/\s/.test(trimmed)) return null
  if (trimmed.includes('?') || trimmed.includes('#')) return null
  if (!trimmed.startsWith(GITHUB_PREFIX)) return null

  const rest = trimmed.slice(GITHUB_PREFIX.length)
  const sep = rest.indexOf('//')
  const repoRaw = sep === -1 ? rest : rest.slice(0, sep)
  const subdirRaw = sep === -1 ? null : rest.slice(sep + 2)

  const repo = repoRaw.replace(/\/$/, '')
  if (!OWNER_REPO.test(repo)) return null

  const ownerRepo = repo.replace(/\.git$/i, '')
  const href = `${GITHUB_PREFIX}${ownerRepo}`

  if (subdirRaw === null) {
    return {
      normalized: `${GITHUB_PREFIX}${repo}`,
      ownerRepo,
      subdir: null,
      href
    }
  }

  const subdir = subdirRaw.replace(/\/$/, '')
  if (!subdir || subdir.includes('..') || !SUBDIR.test(subdir)) return null

  return {
    normalized: `${GITHUB_PREFIX}${repo}//${subdir}`,
    ownerRepo,
    subdir,
    href
  }
}

export function InstallPanel({
  installedAgents,
  provisioningNames,
  onInstalled,
  libraryCount,
  onBackToLibrary
}: InstallPanelProps) {
  const [catalog, setCatalog] = useState<CatalogEntry[]>([])
  const [phase, setPhase] = useState<InstallPhase>({ state: 'idle' })
  /** Catalog entry with the uninstall confirm step open. */
  const [confirming, setConfirming] = useState<string | null>(null)
  /** Catalog entry whose card overflow menu is open. */
  const [openMenu, setOpenMenu] = useState<string | null>(null)
  /** Paste-from-GitHub hero input. */
  const [repoUrl, setRepoUrl] = useState('')
  /** Show inline invalid-shape error after the field has been touched. */
  const [repoTouched, setRepoTouched] = useState(false)
  /** Drive the contextual //subdir typing hint. */
  const [repoFocused, setRepoFocused] = useState(false)
  const reducedMotion = useReducedMotion()

  useEffect(() => {
    void window.agentfield.getCatalog().then(setCatalog)
  }, [])

  useEffect(() => {
    if (openMenu === null) return
    const close = () => setOpenMenu(null)
    // Defer so the opening click doesn't immediately close the menu.
    const timer = window.setTimeout(() => {
      window.addEventListener('click', close)
    }, 0)
    return () => {
      window.clearTimeout(timer)
      window.removeEventListener('click', close)
    }
  }, [openMenu])

  useEffect(() => {
    return window.agentfield.onInstallProgress((line) => {
      setPhase((prev) => {
        if (prev.state !== 'installing') return prev
        const lines = [...prev.lines, line].slice(-MAX_PROGRESS_LINES)
        return { ...prev, progress: line, lines }
      })
    })
  }, [])

  const install = async (name: string) => {
    setPhase({ state: 'installing', name, progress: 'Starting…', lines: ['Starting…'] })
    const result = await window.agentfield.install(name)
    setPhase({
      state: 'done',
      name,
      ok: result.ok,
      message: result.message,
      lines: []
    })
    if (result.ok) onInstalled()
  }

  const uninstall = async (name: string) => {
    setConfirming(null)
    setOpenMenu(null)
    setPhase({
      state: 'installing',
      name,
      progress: 'Uninstalling…',
      lines: ['Uninstalling…']
    })
    const result = await window.agentfield.uninstall(name)
    setPhase({
      state: 'done',
      name,
      ok: result.ok,
      message: result.message,
      lines: []
    })
    onInstalled()
  }

  const update = async (name: string) => {
    setOpenMenu(null)
    setPhase({ state: 'installing', name, progress: 'Updating…', lines: ['Updating…'] })
    const result = await updateWithExecutionConfirmation(
      name,
      (force) => window.agentfield.update(name, force ? { force: true } : undefined),
      (message) => window.confirm(message)
    )
    setPhase({
      state: 'done',
      name,
      ok: result.ok,
      message: result.message,
      lines: []
    })
    if (result.ok) onInstalled()
  }

  const installing = phase.state === 'installing'
  const installedByName = new Map(installedAgents.map((agent) => [agent.name, agent]))
  const parsed = parseRepoSource(repoUrl)
  const repoInvalid = repoUrl.trim().length > 0 && parsed === null
  const showRepoError = repoInvalid && repoTouched
  const repoBusy = installing && phase.name === REPO_TARGET
  const heroLines =
    (phase.state === 'installing' || phase.state === 'done') && phase.name === REPO_TARGET
      ? phase.lines
      : []
  // Contextual tip while editing: hide once they've typed the //subdir
  // selector after github.com/<owner>/<repo> (not the https:// scheme).
  const trimmedRepo = repoUrl.trim()
  const afterPrefix = trimmedRepo.startsWith(GITHUB_PREFIX)
    ? trimmedRepo.slice(GITHUB_PREFIX.length)
    : ''
  const hasSubdirSelector = afterPrefix.includes('//')
  const showSubdirHint = repoFocused && !hasSubdirSelector

  const installRepo = async () => {
    setRepoTouched(true)
    if (installing) return
    const source = parseRepoSource(repoUrl)
    if (!source) return
    setPhase({
      state: 'installing',
      name: REPO_TARGET,
      progress: 'Starting…',
      lines: ['Starting…']
    })
    const result = await window.agentfield.installFromSource(source.normalized)
    setPhase((prev) => {
      const priorLines = prev.state === 'installing' && prev.name === REPO_TARGET ? prev.lines : []
      const lines = [...priorLines, result.message].slice(-MAX_PROGRESS_LINES)
      return {
        state: 'done',
        name: REPO_TARGET,
        ok: result.ok,
        message: result.message,
        lines
      }
    })
    if (result.ok) {
      setRepoUrl('')
      setRepoTouched(false)
      onInstalled()
    }
  }

  return (
    <div className="agents-add">
      {onBackToLibrary && (
        <div className="agents-add-back">
          <button type="button" className="action-button ghost" onClick={onBackToLibrary}>
            Back to installed ({libraryCount})
          </button>
        </div>
      )}

      <p className="view-lede">
        Paste any public GitHub repo with an <span className="mono">agentfield-package.yaml</span>.
      </p>

      <div className="install-hero">
        <div className="install-hero-label">Install from GitHub</div>
        <div className="install-hero-row">
          <input
            className="env-input repo-input"
            type="text"
            placeholder="https://github.com/org/repo or …/repo//subdir"
            value={repoUrl}
            disabled={installing}
            spellCheck={false}
            autoCapitalize="off"
            autoCorrect="off"
            onChange={(e) => {
              setRepoUrl(e.target.value)
              if (!repoTouched && e.target.value.trim().length > 0) setRepoTouched(true)
            }}
            onFocus={() => setRepoFocused(true)}
            onBlur={() => {
              setRepoFocused(false)
              if (repoUrl.trim().length > 0) setRepoTouched(true)
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter') void installRepo()
            }}
          />
          <button
            className="install-button"
            disabled={installing || repoUrl.trim().length === 0 || parsed === null}
            onClick={() => void installRepo()}
          >
            {repoBusy ? 'Installing…' : 'Install'}
          </button>
        </div>

        {parsed && (
          <div className="install-hero-echo">
            Installs{' '}
            <span className="mono">
              {parsed.ownerRepo}
              {parsed.subdir ? `//${parsed.subdir}` : ''}
            </span>
            <a href={parsed.href} target="_blank" rel="noreferrer" title="Open repository">
              ↗
            </a>
          </div>
        )}

        {/* Typing note (DESIGN.md §5): 160ms opacity + 4px rise; exits when
            they blur or start a //subdir selector. */}
        <AnimatePresence initial={false}>
          {showSubdirHint && (
            <m.p
              key="subdir-hint"
              className="install-hero-hint"
              initial={reducedMotion ? { opacity: 0 } : { opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.16, ease: [0.16, 1, 0.3, 1] }}
            >
              Tip: append <span className="mono">//subdir</span> when a repo hosts more than
              one node.
            </m.p>
          )}
        </AnimatePresence>

        {showRepoError && <p className="install-hero-error">{INVALID_SOURCE_MSG}</p>}

        <p className="install-hero-disclosure">
          Installing downloads and runs this repo’s agent code on your machine.{' '}
          <a
            className="link-button"
            href={COMMUNITY_LINKS.installingGuide}
            target="_blank"
            rel="noreferrer"
          >
            authoring guide
          </a>
        </p>

        {heroLines.length > 0 && (
          <div className="install-progress-lines" aria-live="polite">
            {heroLines.map((line, i) => (
              <div key={`${i}-${line}`} className="install-progress-line">
                {line}
              </div>
            ))}
          </div>
        )}

        {phase.state === 'done' && phase.name === REPO_TARGET && phase.ok && (
          <p className="install-success">
            <InstallCheck />
            Installed
          </p>
        )}

        {/* `shake` runs its one 250ms cycle on mount (§5.2 failure cue). */}
        {phase.state === 'done' && phase.name === REPO_TARGET && !phase.ok && (
          <p className="install-hero-error shake">{phase.message}</p>
        )}
      </div>

      <h3 className="section-title">Featured</h3>
      {catalog.length === 0 ? (
        <div className="panel">
          <SkeletonRows count={3} />
        </div>
      ) : (
        <div className="market-grid">
          {catalog.map((entry) => (
            <FeaturedCard
              key={entry.name}
              entry={entry}
              installedAgent={installedByName.get(entry.name)}
              provisioning={provisioningNames.includes(entry.name)}
              installing={installing}
              phase={phase}
              confirming={confirming === entry.name}
              menuOpen={openMenu === entry.name}
              onInstall={() => void install(entry.name)}
              onUpdate={() => void update(entry.name)}
              onUninstall={() => void uninstall(entry.name)}
              onConfirmUninstall={() => {
                setOpenMenu(null)
                setConfirming(entry.name)
              }}
              onCancelUninstall={() => setConfirming(null)}
              onToggleMenu={() => setOpenMenu(openMenu === entry.name ? null : entry.name)}
            />
          ))}
        </div>
      )}
    </div>
  )
}

/**
 * One marketplace card (DESIGN.md §4.11): name + language chip, two-line
 * description, mono source repo (the trust cue) + Install. Installed cards
 * show ✓ Installed with Update / Uninstall behind a card overflow menu, and
 * install progress lines stream inside the active card.
 */
function FeaturedCard({
  entry,
  installedAgent,
  provisioning,
  installing,
  phase,
  confirming,
  menuOpen,
  onInstall,
  onUpdate,
  onUninstall,
  onConfirmUninstall,
  onCancelUninstall,
  onToggleMenu
}: {
  entry: CatalogEntry
  installedAgent?: InstalledAgent
  provisioning: boolean
  installing: boolean
  phase: InstallPhase
  confirming: boolean
  menuOpen: boolean
  onInstall: () => void
  onUpdate: () => void
  onUninstall: () => void
  onConfirmUninstall: () => void
  onCancelUninstall: () => void
  onToggleMenu: () => void
}) {
  const active = phase.state !== 'idle' && phase.name === entry.name
  const busy = installing && phase.state === 'installing' && phase.name === entry.name
  const cardLines = busy && phase.state === 'installing' ? phase.lines : []
  const installed = installedAgent !== undefined

  // Curated sources are https git URLs (link out) or af://registry refs
  // (plain text). Show the short org/repo form as the trust cue.
  const sourceHref = entry.source.startsWith('https://') ? entry.source : null
  const sourceLabel = entry.source.replace(/^https:\/\/github\.com\//, '')
  // Drift means a different REPOSITORY, not a different string: a catalog
  // install is recorded as the redirect target (`…/SWE-AF//go`), which must
  // not read as "installed from somewhere else".
  const recordedSource = installedAgent?.source?.trim()
  const recordedSourceLabel = installedSourceLabel(recordedSource, entry.source)

  return (
    <div className="market-card">
      <div className="market-card-head">
        <span className="market-card-name">{entry.name}</span>
        {entry.language ? <span className="chip lang">{entry.language}</span> : null}
      </div>
      <p className="market-card-desc" title={entry.description}>
        {entry.description}
      </p>

      {cardLines.length > 0 && (
        <div className="install-progress-lines" aria-live="polite">
          {cardLines.map((line, i) => (
            <div key={`${i}-${line}`} className="install-progress-line">
              {line}
            </div>
          ))}
        </div>
      )}
      {active && phase.state === 'done' && (
        <span
          className={`row-progress${phase.ok ? ' install-success' : ' error-text shake'}`}
        >
          {phase.ok && <InstallCheck />}
          {phase.message}
        </span>
      )}
      {confirming && !busy && (
        <span className="row-progress warn-text">
          Stops the agent and removes its files, registry entry, and agent-scoped secrets. Shared
          keys stay.
        </span>
      )}

      <div className="market-card-foot">
        <div className="market-provenance">
          {sourceHref ? (
            <a
              className="market-source"
              href={sourceHref}
              target="_blank"
              rel="noreferrer"
              title={`Open ${entry.source}`}
            >
              {sourceLabel} ↗
            </a>
          ) : (
            <span className="market-source">{sourceLabel}</span>
          )}
          {recordedSourceLabel ? (
            <span className="market-source market-installed-source" title={recordedSource}>
              installed from {recordedSourceLabel}
            </span>
          ) : null}
        </div>
        {provisioning ? (
          <button className="install-button" disabled>
            Installing…
          </button>
        ) : installed ? (
          confirming ? (
            <div className="row-actions">
              <button className="action-button danger" disabled={installing} onClick={onUninstall}>
                Uninstall
              </button>
              <button
                className="action-button ghost"
                disabled={installing}
                onClick={onCancelUninstall}
              >
                Cancel
              </button>
            </div>
          ) : (
            <div className="row-actions">
              <span className="market-installed">✓ Installed</span>
              <MenuPopover
                open={menuOpen}
                onToggle={onToggleMenu}
                disabled={installing}
                ariaLabel={`More actions for ${entry.name}`}
              >
                <button
                  className="menu-item"
                  role="menuitem"
                  title="Reinstall the latest version; a running agent restarts after the update"
                  onClick={onUpdate}
                >
                  {busy ? 'Updating…' : 'Update'}
                </button>
                <button className="menu-item danger" role="menuitem" onClick={onConfirmUninstall}>
                  Uninstall
                </button>
              </MenuPopover>
            </div>
          )
        ) : (
          <button className="install-button" disabled={installing} onClick={onInstall}>
            {busy ? 'Installing…' : 'Install'}
          </button>
        )}
      </div>
    </div>
  )
}
