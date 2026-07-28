//go:build ios || !darwin

package systray

// AgentField patch (see PATCHES.md): no-op stub of SetImage on every non-macOS
// target, so the wide-image capability compiles everywhere while only doing real
// work on darwin. Callers can invoke SetImage unconditionally.

// SetImage is a no-op on this platform. See the darwin build for behavior.
func (item *MenuItem) SetImage(pngBytes []byte, widthPt, heightPt int) {}

// SetStatusImage is a no-op on this platform (AgentField patch #2). See the
// darwin build for behavior.
func SetStatusImage(pngBytes []byte, widthPt, heightPt int) {}
