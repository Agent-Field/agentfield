package main

// This file holds the tray's image renderers: a wide 24h token timeseries chart
// and the rounded proportional bars used for per-model usage and Claude quota
// rows. Everything here is pure (image/png only, no systray/CGO), so it compiles
// on every platform and is unit-tested directly on CI. The darwin tray code
// (tray_darwin.go) turns these PNGs into menu-item images via the vendored
// systray fork's SetImage, which — unlike the stock 16x16-clamped icon API — can
// show them at full width and in color. See third_party/systray/PATCHES.md.
//
// Anti-aliasing strategy: each renderer draws at an integer supersample factor
// and box-downsamples to the target size, so curves and rounded corners come out
// smooth rather than jagged. The target pixel size is already 2x the point size
// (retina), and the supersample is on top of that.

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"math"
)

// chartSupersample is the internal oversampling factor for the timeseries chart;
// 3x over the already-2x retina pixels is plenty for smooth curves without being
// slow on the tray's refresh cadence.
const chartSupersample = 3

// barSupersample oversamples the rounded bars so their pill caps read as smooth,
// round edges rather than stair-stepped ones.
const barSupersample = 4

// ---- Palette ---------------------------------------------------------------
//
// One accent, ranked by intensity. Every model graphic in the Usage submenu is
// the single brand orange at a rank-dependent opacity, so the section reads as
// one coherent system rather than a rainbow. The only other colors are the
// neutral "other" gray, the neutral track, and the SEMANTIC green/amber/red of
// the Claude quota gauge — a distinction that actually carries meaning.

// accentColor is the brand orange every model usage graphic is tinted with.
var accentColor = color.NRGBA{0xe0, 0x8a, 0x3c, 0xff} // #e08a3c

var (
	// barTrackColor is the faint unfilled-track hue behind proportional bars. A
	// neutral gray at ~30% (not white) so it reads on both dark and light menus.
	barTrackColor = color.NRGBA{0x8a, 0x8f, 0x98, 0x4d}
)

// grayOther is the hue for the aggregated "other" bucket (the long tail beyond
// the ranked top models) in the histogram — a neutral midtone that reads on both
// dark and light menus without competing with the accent.
var grayOther = color.NRGBA{0x8a, 0x8f, 0x98, 0xff}

// modelRankAlpha is the accent's opacity (0..255) for a model at the given rank:
// rank 0 is full strength, and each lower rank fades toward the menu background,
// so intensity encodes rank with a single hue. Ranks beyond the third share the
// faintest step rather than ever going colorless.
func modelRankAlpha(rank int) uint8 {
	switch {
	case rank <= 0:
		return 0xff // 100%
	case rank == 1:
		return 0xa6 // ~65%
	case rank == 2:
		return 0x66 // ~40%
	default:
		return 0x40 // ~25%
	}
}

// modelBarColor returns the accent tinted to the given rank's intensity (0 = top,
// full strength). It is never fully transparent, so no row is ever colorless.
func modelBarColor(rank int) color.NRGBA {
	if rank < 0 {
		rank = 0
	}
	c := accentColor
	c.A = modelRankAlpha(rank)
	return c
}

// stackedLayerColor maps a histogram layer to its hue: the aggregated "other"
// bucket is neutral gray, every ranked model takes the accent at its rank
// intensity.
func stackedLayerColor(key string, rank int) color.NRGBA {
	if key == "other" {
		return grayOther
	}
	return modelBarColor(rank)
}

// quotaBarColor tints a rate-limit bar by utilization: green when there's plenty
// of headroom, orange as it fills, red when nearly exhausted.
func quotaBarColor(pct float64) color.NRGBA {
	switch {
	case pct >= 80:
		return color.NRGBA{0xff, 0x45, 0x3a, 0xff} // red    #ff453a
	case pct >= 50:
		return color.NRGBA{0xe0, 0x8a, 0x3c, 0xff} // orange #e08a3c
	default:
		return color.NRGBA{0x30, 0xd1, 0x58, 0xff} // green  #30d158
	}
}

// ---- Uniform leading slot: spacer + compact proportional bar ----------------
//
// Every row in the Usage submenu carries a leading image of exactly the same
// (slot) size so that every native title starts at the same x. Rows with a
// graphic draw a compact bar inside the slot; rows without one get a fully
// transparent spacer of the same size.

// spacerImagePNG returns a fully transparent PNG of the uniform slot size. It is
// the leading image on rows that have no graphic (summary lines, section
// headers, rollups, footer) so their titles line up with the bar/gauge rows.
func spacerImagePNG(wPx, hPx int) []byte {
	if wPx <= 0 || hPx <= 0 {
		return nil
	}
	return encodePNG(image.NewNRGBA(image.Rect(0, 0, wPx, hPx)))
}

// slotBarPNG renders a compact rounded "pill" bar left-aligned and vertically
// centered inside a uniform leading slot of slotWPx×slotHPx (the rest of the slot
// is transparent). The bar itself is barWPx×barHPx and is filled to the given
// fraction (0..1) in the given color over a faint neutral track. A tiny-but-
// nonzero fraction still shows a full rounded cap so it is never invisible. This
// keeps the model/quota bars aligned with the histogram's left edge and every
// title at the same x.
func slotBarPNG(fraction float64, fill color.NRGBA, slotWPx, slotHPx, barWPx, barHPx int) []byte {
	if slotWPx <= 0 || slotHPx <= 0 || barWPx <= 0 || barHPx <= 0 {
		return nil
	}
	if barWPx > slotWPx {
		barWPx = slotWPx
	}
	if barHPx > slotHPx {
		barHPx = slotHPx
	}
	if fraction < 0 {
		fraction = 0
	}
	if fraction > 1 {
		fraction = 1
	}
	ss := barSupersample
	s := float64(ss)
	W, H := slotWPx*ss, slotHPx*ss
	hi := image.NewNRGBA(image.Rect(0, 0, W, H))

	// Bar region: left-aligned, vertically centered, inset half a 1x pixel so the
	// pill's edges aren't clipped.
	inset := 0.5 * s
	x0 := inset
	x1 := float64(barWPx)*s - inset
	yTop := float64(slotHPx-barHPx) / 2 * s
	y0 := yTop + inset
	y1 := yTop + float64(barHPx)*s - inset
	radius := (y1 - y0) / 2

	// Fill extent. Keep at least a full round cap visible for any nonzero share.
	fillRight := x0 + fraction*(x1-x0)
	if fraction > 0 {
		minRight := x0 + 2*radius
		if fillRight < minRight {
			fillRight = minRight
		}
	}

	for y := int(yTop); y <= int(y1)+1 && y < H; y++ {
		if y < 0 {
			continue
		}
		for x := 0; x <= int(x1)+1 && x < W; x++ {
			px, py := float64(x)+0.5, float64(y)+0.5
			if !insideRoundedRect(px, py, x0, y0, x1, y1, radius) {
				continue
			}
			hi.SetNRGBA(x, y, barTrackColor)
			if fraction > 0 && insideRoundedRect(px, py, x0, y0, fillRight, y1, radius) {
				hi.SetNRGBA(x, y, fill)
			}
		}
	}

	return encodePNG(downsample(hi, slotWPx, slotHPx, ss))
}

// ---- Menu-bar status badge ---------------------------------------------------
//
// The tray's menu-bar item shows the brand "af" badge plus a small status glyph
// to its right: a filled green dot while the control plane is running, a gray
// ring when it is stopped, and a rotating orange arc while it starts up. The
// composite is applied via the vendored systray fork's SetStatusImage; it
// carries no text and is unit-testable on Linux.

// Menu-bar badge layout in points: [16pt badge][2pt gap][8pt status glyph].
const (
	statusBadgeWidthPt  = 27
	statusBadgeHeightPt = 18
	statusBadgeIconPt   = 16
	statusBadgeGapPt    = 2
	statusBadgeDotPt    = 8
)

// serverState is the control plane's coarse lifecycle state as seen by the tray.
type serverState int

const (
	serverStopped  serverState = iota // no process, not answering
	serverStarting                    // process present but /health not yet 200
	serverRunning                     // healthy
)

// deriveServerState maps the two observable facts to a lifecycle state.
func deriveServerState(healthy, processRunning bool) serverState {
	switch {
	case healthy:
		return serverRunning
	case processRunning:
		return serverStarting
	default:
		return serverStopped
	}
}

var (
	statusRunningColor  = color.NRGBA{0x30, 0xd1, 0x58, 0xff} // green #30d158
	statusStoppedColor  = color.NRGBA{0x98, 0x98, 0x9d, 0xff} // idle gray
	statusStartingColor = color.NRGBA{0xe0, 0x8a, 0x3c, 0xff} // orange
)

// statusArcSteps is how many rotation steps the starting arc cycles through;
// the caller advances `phase` on a short ticker to animate it.
const statusArcSteps = 8

// statusBadgePNG composites the menu-bar image at 2x (retina): the brand badge
// on the left and the state glyph on the right. phase (any monotonically
// increasing counter) rotates the starting arc and is ignored for the other
// states. badgePNG is the brand icon at any size; nil/empty omits it.
func statusBadgePNG(badgePNG []byte, state serverState, phase int) []byte {
	const scale = 2
	W, H := statusBadgeWidthPt*scale, statusBadgeHeightPt*scale
	out := image.NewNRGBA(image.Rect(0, 0, W, H))

	iconPx := statusBadgeIconPt * scale
	if badge := decodeNRGBA(badgePNG); badge != nil {
		scaled := resizeNRGBA(badge, iconPx, iconPx)
		overComposite(out, scaled, 0, (H-iconPx)/2)
	}

	// The glyph is drawn supersampled then downsampled for smooth edges.
	ss := chartSupersample
	hi := image.NewNRGBA(image.Rect(0, 0, W*ss, H*ss))
	s := float64(ss)
	dotPx := float64(statusBadgeDotPt * scale)
	cx := (float64(iconPx+statusBadgeGapPt*scale) + dotPx/2) * s
	cy := float64(H) / 2 * s
	r := dotPx / 2 * s

	var col color.NRGBA
	inner := 0.0 // >0 carves a hole → ring
	arcSpan := 0.0
	arcStart := 0.0
	switch state {
	case serverRunning:
		col = statusRunningColor
	case serverStopped:
		col = statusStoppedColor
		inner = r * 0.55
	case serverStarting:
		col = statusStartingColor
		inner = r * 0.55
		arcSpan = 1.5 * math.Pi // 270° arc…
		arcStart = float64(phase%statusArcSteps) / statusArcSteps * 2 * math.Pi
	}

	for y := int(cy - r - 2); y <= int(cy+r+2); y++ {
		for x := int(cx - r - 2); x <= int(cx+r+2); x++ {
			if x < 0 || y < 0 || x >= W*ss || y >= H*ss {
				continue
			}
			dx := float64(x) + 0.5 - cx
			dy := float64(y) + 0.5 - cy
			d := math.Sqrt(dx*dx + dy*dy)
			a := r - d
			if inner > 0 && a > 0 {
				if hole := d - inner; hole < a {
					a = hole
				}
			}
			if a <= 0 {
				continue
			}
			if arcSpan > 0 {
				ang := math.Atan2(dy, dx) - arcStart
				for ang < 0 {
					ang += 2 * math.Pi
				}
				if ang > arcSpan {
					continue
				}
			}
			if a > 1 {
				a = 1
			}
			blendPixel(hi, x, y, col, a)
		}
	}

	glyph := downsample(hi, W, H, ss)
	overComposite(out, glyph, 0, 0)
	return encodePNG(out)
}

// ---- image helpers (badge decode / resize / composite) ----------------------

// decodeNRGBA decodes PNG bytes into an origin-zero NRGBA, or nil on failure.
func decodeNRGBA(b []byte) *image.NRGBA {
	if len(b) == 0 {
		return nil
	}
	img, err := png.Decode(bytes.NewReader(b))
	if err != nil {
		return nil
	}
	sb := img.Bounds()
	out := image.NewNRGBA(image.Rect(0, 0, sb.Dx(), sb.Dy()))
	for y := 0; y < sb.Dy(); y++ {
		for x := 0; x < sb.Dx(); x++ {
			out.Set(x, y, img.At(sb.Min.X+x, sb.Min.Y+y))
		}
	}
	return out
}

// resizeNRGBA bilinearly resamples src to dstW×dstH, interpolating in
// premultiplied-alpha space so transparent edges blend cleanly.
func resizeNRGBA(src *image.NRGBA, dstW, dstH int) *image.NRGBA {
	dst := image.NewNRGBA(image.Rect(0, 0, dstW, dstH))
	sw, sh := src.Bounds().Dx(), src.Bounds().Dy()
	if sw == 0 || sh == 0 || dstW == 0 || dstH == 0 {
		return dst
	}
	at := func(x, y int) (r, g, b, a float64) {
		if x < 0 {
			x = 0
		}
		if x >= sw {
			x = sw - 1
		}
		if y < 0 {
			y = 0
		}
		if y >= sh {
			y = sh - 1
		}
		c := src.NRGBAAt(x, y)
		af := float64(c.A) / 255
		return float64(c.R) * af, float64(c.G) * af, float64(c.B) * af, af
	}
	for dy := 0; dy < dstH; dy++ {
		fy := (float64(dy)+0.5)*float64(sh)/float64(dstH) - 0.5
		y0 := int(math.Floor(fy))
		ty := fy - float64(y0)
		for dx := 0; dx < dstW; dx++ {
			fx := (float64(dx)+0.5)*float64(sw)/float64(dstW) - 0.5
			x0 := int(math.Floor(fx))
			tx := fx - float64(x0)
			r00, g00, b00, a00 := at(x0, y0)
			r10, g10, b10, a10 := at(x0+1, y0)
			r01, g01, b01, a01 := at(x0, y0+1)
			r11, g11, b11, a11 := at(x0+1, y0+1)
			lerp := func(a, b, t float64) float64 { return a + (b-a)*t }
			r := lerp(lerp(r00, r10, tx), lerp(r01, r11, tx), ty)
			g := lerp(lerp(g00, g10, tx), lerp(g01, g11, tx), ty)
			bb := lerp(lerp(b00, b10, tx), lerp(b01, b11, tx), ty)
			a := lerp(lerp(a00, a10, tx), lerp(a01, a11, tx), ty)
			if a <= 0 {
				dst.SetNRGBA(dx, dy, color.NRGBA{})
				continue
			}
			dst.SetNRGBA(dx, dy, color.NRGBA{
				R: uint8(math.Round(r / a)),
				G: uint8(math.Round(g / a)),
				B: uint8(math.Round(bb / a)),
				A: uint8(math.Round(a * 255)),
			})
		}
	}
	return dst
}

// overComposite alpha-composites src over dst with its top-left at (ox,oy),
// using the same straight-NRGBA blend as the rest of the renderers.
func overComposite(dst, src *image.NRGBA, ox, oy int) {
	b := src.Bounds()
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			c := src.NRGBAAt(x, y)
			if c.A == 0 {
				continue
			}
			blendPixel(dst, ox+x-b.Min.X, oy+y-b.Min.Y, color.NRGBA{c.R, c.G, c.B, 0xff}, float64(c.A)/255)
		}
	}
}

// ---- Rendering primitives --------------------------------------------------

// insideRoundedRect reports whether point (px,py) lies inside the rounded
// rectangle [x0,x1]x[y0,y1] with the given corner radius.
func insideRoundedRect(px, py, x0, y0, x1, y1, radius float64) bool {
	if px < x0 || px > x1 || py < y0 || py > y1 {
		return false
	}
	if radius <= 0 {
		return true
	}
	// Clamp to the inner rectangle of corner centers; distance to that clamp is
	// the corner test.
	cx := math.Max(x0+radius, math.Min(px, x1-radius))
	cy := math.Max(y0+radius, math.Min(py, y1-radius))
	dx, dy := px-cx, py-cy
	return dx*dx+dy*dy <= radius*radius
}

// blendPixel alpha-composites color c at coverage a (0..1) over the existing
// pixel, using straight (non-premultiplied) NRGBA math on a transparent canvas.
func blendPixel(img *image.NRGBA, x, y int, c color.NRGBA, a float64) {
	if a <= 0 {
		return
	}
	if a > 1 {
		a = 1
	}
	srcA := float64(c.A) / 255 * a
	if srcA <= 0 {
		return
	}
	dst := img.NRGBAAt(x, y)
	dstA := float64(dst.A) / 255
	outA := srcA + dstA*(1-srcA)
	if outA <= 0 {
		img.SetNRGBA(x, y, color.NRGBA{})
		return
	}
	blend := func(sc, dc uint8) uint8 {
		s := float64(sc) / 255
		d := float64(dc) / 255
		v := (s*srcA + d*dstA*(1-srcA)) / outA
		if v < 0 {
			v = 0
		}
		if v > 1 {
			v = 1
		}
		return uint8(math.Round(v * 255))
	}
	img.SetNRGBA(x, y, color.NRGBA{
		R: blend(c.R, dst.R),
		G: blend(c.G, dst.G),
		B: blend(c.B, dst.B),
		A: uint8(math.Round(outA * 255)),
	})
}

// downsample box-averages an ss*ss supersampled NRGBA image down to wPx x hPx,
// averaging in premultiplied-alpha space so transparent edges blend correctly.
func downsample(src *image.NRGBA, wPx, hPx, ss int) *image.NRGBA {
	out := image.NewNRGBA(image.Rect(0, 0, wPx, hPx))
	n := float64(ss * ss)
	for oy := 0; oy < hPx; oy++ {
		for ox := 0; ox < wPx; ox++ {
			var pr, pg, pb, pa float64
			for sy := 0; sy < ss; sy++ {
				for sx := 0; sx < ss; sx++ {
					c := src.NRGBAAt(ox*ss+sx, oy*ss+sy)
					af := float64(c.A) / 255
					pr += float64(c.R) * af
					pg += float64(c.G) * af
					pb += float64(c.B) * af
					pa += af
				}
			}
			outA := pa / n
			if pa <= 0 {
				out.SetNRGBA(ox, oy, color.NRGBA{})
				continue
			}
			out.SetNRGBA(ox, oy, color.NRGBA{
				R: uint8(math.Round(pr / pa)),
				G: uint8(math.Round(pg / pa)),
				B: uint8(math.Round(pb / pa)),
				A: uint8(math.Round(outA * 255)),
			})
		}
	}
	return out
}

func encodePNG(img image.Image) []byte {
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return nil
	}
	return buf.Bytes()
}
