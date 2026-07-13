// Icon generator for AgentField Desktop (and the af-tray macOS bundle icon).
//
// Renders the brand "•af" mark — the exact outlined paths from
// control-plane/web/client/src/assets/logos/logo-short-*.svg, so there is no
// font dependency — into every raster the apps need, using an offscreen
// Electron window as the SVG rasterizer (no native image deps required).
//
// Run from desktop/:   npx electron scripts/make-icons.mjs
// (unset ELECTRON_RUN_AS_NODE first if your shell inherits it)
//
// Outputs (all committed; regenerate only when the brand mark changes):
//   build/icon.icns                      macOS app icon (Apple-grid margins)
//   build/icon.png                       Windows/Linux app icon source (1024)
//   resources/icon.png                   runtime window icon (win/linux, 256)
//   resources/tray/tray-<s>-<g>-<n>.png  Windows tray glyphs
//                                        s: active|inactive  g: light|dark
//   ../control-plane/cmd/af-tray/assets/appicon.icns   same icns for af-tray

import { BrowserWindow, app } from 'electron'
import { mkdirSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const desktopDir = dirname(dirname(fileURLToPath(import.meta.url)))

// ---- Brand -----------------------------------------------------------------

const INK_TOP = '#221d15' // warm near-black gradient, top…
const INK_BOTTOM = '#0b0a08' // …to bottom
const CREAM = '#f5f0eb'
const GOLD = '#d4a24a'
const GRAY_LIGHT = '#9c9c9c' // inactive glyph on dark taskbars
const GRAY_DARK = '#6f6f6f' // inactive glyph on light taskbars

// The "af" letterforms and the dot, verbatim from the logo SVG (1000×1000
// viewBox). Mark bounding box: x 180…865.168, y 329…697.
const AF_PATH =
  'M656.324 689H585.824V636.5L581.324 632.5V536C581.324 520.333 576.824 508.333 567.824 500C558.824 491.667 545.824 487.5 528.824 487.5C506.491 487.5 485.158 495.833 464.824 512.5L424.324 464C456.324 438.667 493.824 426 536.824 426C561.158 426 582.158 430.5 599.824 439.5C617.824 448.167 631.658 460.667 641.324 477C651.324 493 656.324 512.167 656.324 534.5V689ZM510.324 697C491.324 697 474.824 693.5 460.824 686.5C446.824 679.5 435.824 669.667 427.824 657C420.158 644.333 416.324 629.5 416.324 612.5C416.324 586.167 426.991 565.833 448.324 551.5C469.658 536.833 499.491 529.5 537.824 529.5H586.324V578H542.324C528.324 578 517.158 580.833 508.824 586.5C500.824 592.167 496.824 599.833 496.824 609.5C496.824 617.833 499.991 624.833 506.324 630.5C512.991 635.833 521.491 638.5 531.824 638.5C541.491 638.5 550.324 636.167 558.324 631.5C566.324 626.833 572.824 620.5 577.824 612.5C583.158 604.167 586.158 594.833 586.824 584.5L605.324 593C605.324 614 601.324 632.333 593.324 648C585.658 663.333 574.658 675.333 560.324 684C546.324 692.667 529.658 697 510.324 697ZM811.168 689H729.668V406C729.668 354.667 754.168 329 803.168 329H865.168V392.5H834.168C825.835 392.5 819.835 394.5 816.168 398.5C812.835 402.167 811.168 408.833 811.168 418.5V689ZM865.168 499.5H693.168V434H865.168V499.5Z'

const MARK = { minX: 180, maxX: 865.168, minY: 329, maxY: 697 }
const MARK_W = MARK.maxX - MARK.minX
const MARK_CX = (MARK.minX + MARK.maxX) / 2
const MARK_CY = (MARK.minY + MARK.maxY) / 2

function mark(dotColor, textColor) {
  return (
    `<circle cx="280" cy="566" r="100" fill="${dotColor}"/>` +
    `<path d="${AF_PATH}" fill="${textColor}"/>`
  )
}

/** Center the mark at (cx, cy) scaled to targetW wide, in the icon's coords. */
function placeMark(inner, cx, cy, targetW) {
  const s = targetW / MARK_W
  const tx = cx - MARK_CX * s
  const ty = cy - MARK_CY * s
  return `<g transform="translate(${tx} ${ty}) scale(${s})">${inner}</g>`
}

const BG_DEFS =
  `<linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">` +
  `<stop offset="0" stop-color="${INK_TOP}"/>` +
  `<stop offset="1" stop-color="${INK_BOTTOM}"/>` +
  `</linearGradient>`

// macOS app icon: Apple's grid — 824×824 squircle centered on a 1024 canvas
// with a baked-in soft shadow, mark at ~65% of the tile width.
function macIconSVG(size) {
  return (
    `<svg width="${size}" height="${size}" viewBox="0 0 1024 1024" xmlns="http://www.w3.org/2000/svg">` +
    `<defs>${BG_DEFS}` +
    `<filter id="sh" x="-20%" y="-20%" width="140%" height="140%">` +
    `<feDropShadow dx="0" dy="10" stdDeviation="12" flood-color="#000000" flood-opacity="0.35"/>` +
    `</filter></defs>` +
    `<rect x="100" y="100" width="824" height="824" rx="185" fill="url(#bg)" filter="url(#sh)"/>` +
    placeMark(mark(GOLD, CREAM), 512, 512, 536) +
    `</svg>`
  )
}

// Windows/Linux app icon: closer to full-bleed (taskbars don't add margins),
// same rounded-square language.
function winIconSVG(size) {
  return (
    `<svg width="${size}" height="${size}" viewBox="0 0 1024 1024" xmlns="http://www.w3.org/2000/svg">` +
    `<defs>${BG_DEFS}</defs>` +
    `<rect x="32" y="32" width="960" height="960" rx="212" fill="url(#bg)"/>` +
    placeMark(mark(GOLD, CREAM), 512, 512, 620) +
    `</svg>`
  )
}

// Tray glyph: transparent background, mark only. The dot doubles as the
// status light — gold when the control plane is running, gray otherwise.
function traySVG(size, active, glyph) {
  const text = glyph === 'light' ? (active ? CREAM : GRAY_LIGHT) : active ? '#0c0b09' : GRAY_DARK
  const dot = active ? GOLD : glyph === 'light' ? GRAY_LIGHT : GRAY_DARK
  return (
    `<svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}" xmlns="http://www.w3.org/2000/svg">` +
    placeMark(mark(dot, text), size / 2, size / 2, size - 2) +
    `</svg>`
  )
}

// ---- ICNS container ----------------------------------------------------------
// Modern icns is just a TOC of PNG blobs: 'icns' + length, then per-entry
// 4-char type + length + PNG bytes.

const ICNS_TYPES = [
  ['ic10', 1024],
  ['ic14', 512],
  ['ic09', 512],
  ['ic13', 256],
  ['ic08', 256],
  ['ic07', 128],
  ['ic12', 64],
  ['ic11', 32]
]

function buildIcns(pngBySize) {
  const chunks = []
  for (const [type, size] of ICNS_TYPES) {
    const png = pngBySize.get(size)
    const header = Buffer.alloc(8)
    header.write(type, 0, 'ascii')
    header.writeUInt32BE(png.length + 8, 4)
    chunks.push(header, png)
  }
  const body = Buffer.concat(chunks)
  const head = Buffer.alloc(8)
  head.write('icns', 0, 'ascii')
  head.writeUInt32BE(body.length + 8, 4)
  return Buffer.concat([head, body])
}

// ---- Rasterizer ---------------------------------------------------------------

app.disableHardwareAcceleration()

async function rasterize(win, svg, size) {
  const html =
    `<!doctype html><meta charset="utf-8">` +
    `<style>html,body{margin:0;background:transparent;overflow:hidden}svg{display:block}</style>` +
    svg
  await win.loadURL('data:text/html;charset=utf-8,' + encodeURIComponent(html))
  await new Promise((r) => setTimeout(r, 120)) // let the offscreen frame settle
  let image = await win.webContents.capturePage({ x: 0, y: 0, width: size, height: size })
  if (image.getSize().width !== size) {
    image = image.resize({ width: size, height: size, quality: 'best' })
  }
  return image.toPNG()
}

app.whenReady().then(async () => {
  const win = new BrowserWindow({
    width: 1024,
    height: 1024,
    show: false,
    frame: false,
    transparent: true,
    useContentSize: true,
    webPreferences: { offscreen: true }
  })

  const out = (...p) => join(desktopDir, ...p)
  mkdirSync(out('build'), { recursive: true })
  mkdirSync(out('resources', 'tray'), { recursive: true })

  // macOS icns members.
  const macPngs = new Map()
  for (const size of [1024, 512, 256, 128, 64, 32]) {
    macPngs.set(size, await rasterize(win, macIconSVG(size), size))
  }
  const icns = buildIcns(macPngs)
  writeFileSync(out('build', 'icon.icns'), icns)
  writeFileSync(
    join(desktopDir, '..', 'control-plane', 'cmd', 'af-tray', 'assets', 'appicon.icns'),
    icns
  )

  // Windows/Linux app + window icons.
  writeFileSync(out('build', 'icon.png'), await rasterize(win, winIconSVG(1024), 1024))
  writeFileSync(out('resources', 'icon.png'), await rasterize(win, winIconSVG(256), 256))

  // Tray glyphs: 16 (1x), 24 (1.5x), 32 (2x) per variant.
  for (const active of [true, false]) {
    for (const glyph of ['light', 'dark']) {
      for (const size of [16, 24, 32]) {
        const name = `tray-${active ? 'active' : 'inactive'}-${glyph}-${size}.png`
        writeFileSync(
          out('resources', 'tray', name),
          await rasterize(win, traySVG(size, active, glyph), size)
        )
      }
    }
  }

  console.log('icons written to build/, resources/, and af-tray assets')
  app.exit(0)
})
