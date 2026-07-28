# AgentField patches to fyne.io/systray v1.12.2

This directory is a vendored copy of [`fyne.io/systray`](https://github.com/fyne-io/systray)
at tag **v1.12.2**, wired in via a `replace fyne.io/systray => ./third_party/systray`
directive in `control-plane/go.mod`. The upstream `LICENSE` (Apache License 2.0)
is preserved verbatim.

The copy is byte-identical to upstream **except** for the two additive
capabilities described below. No upstream behavior is changed or removed.

## Patch #1 — `(*MenuItem).SetImage`

### Why

The AgentField macOS menu-bar tray wants to show wide, full-color images inside
its `NSMenu` (a 24h token timeseries chart, per-model usage bars, Claude quota
bars). The stock library cannot: both `setIcon` and `setMenuItemIcon` in
`systray_darwin.m` hard-code `[image setSize:NSMakeSize(16, 16)]`, clamping every
menu-item image to 16x16 points and marking template images monochrome. Wide,
colored charts are therefore impossible through the public API.

### The capability: `(*MenuItem).SetImage`

Added a single new method that sets a non-template menu-item image at an explicit
point size, with no 16x16 clamp:

```go
func (item *MenuItem) SetImage(pngBytes []byte, widthPt, heightPt int)
```

The caller passes the desired point size; the PNG pixels should be 2x for retina.
The image is **not** marked as a template, so it keeps its own colors.

### Files touched (everything else is byte-identical to upstream)

- `systray.h` — declare `void setMenuItemImage(const char*, int, int, int, int);`.
- `systray_darwin.m` — define `setMenuItemImage(...)`. It mirrors the existing
  `setMenuItemIcon` but sizes the `NSImage` to the caller's point dimensions
  (instead of clamping to 16x16) and sets `image.template = NO`. It reuses the
  existing `setMenuItemIcon:` Objective-C selector (which just assigns
  `menuItem.image`), so no new selector/plumbing is introduced.

### Files added (net-new, not modifications of upstream files)

- `systray_image_darwin.go` (`//go:build !ios`) — the darwin `SetImage` method,
  calling `C.setMenuItemImage`.
- `systray_image_others.go` (`//go:build ios || !darwin`) — a no-op `SetImage`
  stub so the package still builds on Linux, Windows, and every other target.

## Patch #2 — `SetStatusImage`

### Why

The AgentField menu-bar tray wants a *rendered widget* next to its status icon —
the brand "af" badge plus a mini 24h token sparkline with a trend-colored
endpoint dot — rather than the plain `SetTitle("$0.99")` text it used before. The
number itself stays native text via the stock `SetTitle` (matching first-party
menu extras), but the graphic beside it must be a wide, full-color image. The
stock `setIcon` in `systray_darwin.m` hard-codes `[image setSize:NSMakeSize(16,
16)]` and can mark the image as a template, so a wide colored badge+sparkline is
impossible through the public API.

### The capability: `SetStatusImage`

Added a single new package-level function that sets the status-bar button's image
at an explicit point size, non-template, with no 16x16 clamp:

```go
func SetStatusImage(pngBytes []byte, widthPt, heightPt int)
```

The caller passes the desired point size; the PNG pixels should be 2x for retina.
The image is **not** a template, so it keeps its own colors, and it renders beside
the (native) title set with `SetTitle`. Calling the stock `SetIcon` afterward
restores the plain fallback icon (used when there is no data, the toggle is off,
or the server is down), so the existing icon path is unaffected.

### Files touched (everything else is byte-identical to upstream)

- `systray.h` — declare `void setStatusImage(const char*, int, int, int);`.
- `systray_darwin.m` — define `setStatusImage(...)`. It mirrors the existing
  `setIcon` but sizes the `NSImage` to the caller's point dimensions (instead of
  clamping to 16x16) and sets `image.template = NO`. It reuses the existing
  `setIcon:` Objective-C selector (which assigns `statusItem.button.image` and
  runs `updateTitleButtonStyle`), so no new selector/plumbing is introduced.

### Files added to (net-new patch files, shared with patch #1)

- `systray_image_darwin.go` (`//go:build !ios`) — the darwin `SetStatusImage`
  function, calling `C.setStatusImage`.
- `systray_image_others.go` (`//go:build ios || !darwin`) — a no-op
  `SetStatusImage` stub so the package still builds on every non-macOS target.

## Re-syncing with a newer upstream

To bump the fork: re-copy the upstream tag over this directory, then re-apply the
touch-points above — the two `systray.h` declarations (`setMenuItemImage`,
`setStatusImage`), the two `systray_darwin.m` functions, and confirm the two
`systray_image_*.go` files (which hold both patches' Go wrappers) still compile.
Keep this file up to date.
