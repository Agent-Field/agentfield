package main

import _ "embed"

// Menu-bar icons. These are embedded on every platform (they are just bytes),
// but only referenced by the darwin tray implementation. Replace these with the
// final brand assets when available — icon_active is shown when the control
// plane is healthy, icon_inactive when it is stopped/unreachable.
//
//go:embed assets/icon_active.png
var iconActive []byte

//go:embed assets/icon_inactive.png
var iconInactive []byte

// appIconICNS is written into the generated .app bundle's Resources on macOS.
//
//go:embed assets/appicon.icns
var appIconICNS []byte
