//go:build !ios

package systray

// AgentField patch (see PATCHES.md): SetImage adds the one capability the stock
// library lacks — a non-template menu-item image sized to explicit point
// dimensions rather than the hard-coded 16x16 clamp in setMenuItemIcon. This
// lets the AgentField tray show wide, full-color chart and bar images inside an
// NSMenu. Everything else in this fork is byte-identical to upstream v1.12.2.

/*
#include <stdbool.h>
#include "systray.h"
*/
import "C"

import "unsafe"

// SetImage sets a colored (non-template) image on a menu item at an explicit
// point size. widthPt/heightPt are in points; the PNG in pngBytes should be
// rendered at 2x those dimensions for a crisp result on retina displays. Unlike
// SetIcon/SetTemplateIcon, the image is neither clamped to 16x16 nor treated as
// a template, so it keeps its own colors and full width. macOS only; a no-op on
// other platforms.
func (item *MenuItem) SetImage(pngBytes []byte, widthPt, heightPt int) {
	if len(pngBytes) == 0 {
		return
	}
	cstr := (*C.char)(unsafe.Pointer(&pngBytes[0]))
	C.setMenuItemImage(cstr, C.int(len(pngBytes)), C.int(item.id), C.int(widthPt), C.int(heightPt))
}

// SetStatusImage sets the menu-bar status item's button image at an explicit
// point size (AgentField patch #2, see PATCHES.md). widthPt/heightPt are in
// points; the PNG in pngBytes should be rendered at 2x those dimensions for a
// crisp result on retina displays. Unlike SetIcon it is neither clamped to 16x16
// nor treated as a template, so a wide colored widget (brand badge + sparkline)
// keeps its own colors and full width beside the (native) title set with
// SetTitle. Calling SetIcon afterward restores the plain fallback icon. macOS
// only; a no-op on other platforms.
func SetStatusImage(pngBytes []byte, widthPt, heightPt int) {
	if len(pngBytes) == 0 {
		return
	}
	cstr := (*C.char)(unsafe.Pointer(&pngBytes[0]))
	C.setStatusImage(cstr, C.int(len(pngBytes)), C.int(widthPt), C.int(heightPt))
}
