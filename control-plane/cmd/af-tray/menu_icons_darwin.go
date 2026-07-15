//go:build darwin

package main

import _ "embed"

// Menu-item icons. The line icons are from Lucide (https://lucide.dev, ISC),
// rendered to 32×32 PNG (16pt @2x) as black-on-transparent so they are applied
// as macOS *template* images and recolor themselves to match the menu in both
// light and dark mode. The dot-* icons are colored status indicators applied as
// regular (non-template) images so their color is preserved. See
// assets/icons/LICENSE.md.

//go:embed assets/icons/bot.png
var iconBot []byte

//go:embed assets/icons/circle-check.png
var iconSuccess []byte

//go:embed assets/icons/gauge.png
var iconGauge []byte

//go:embed assets/icons/cpu.png
var iconCPU []byte

// Traffic-light variants of the metric icons (green good, yellow caution, red
// bad), applied as regular (colored) images based on each stat's threshold.
//
//go:embed assets/icons/circle-check-green.png
var iconSuccessGreen []byte

//go:embed assets/icons/circle-check-yellow.png
var iconSuccessYellow []byte

//go:embed assets/icons/circle-check-red.png
var iconSuccessRed []byte

//go:embed assets/icons/gauge-green.png
var iconGaugeGreen []byte

//go:embed assets/icons/gauge-yellow.png
var iconGaugeYellow []byte

//go:embed assets/icons/gauge-red.png
var iconGaugeRed []byte

//go:embed assets/icons/cpu-green.png
var iconCPUGreen []byte

//go:embed assets/icons/cpu-yellow.png
var iconCPUYellow []byte

//go:embed assets/icons/cpu-red.png
var iconCPURed []byte

//go:embed assets/icons/layout-dashboard.png
var iconDashboard []byte

//go:embed assets/icons/server.png
var iconServer []byte

//go:embed assets/icons/scroll-text.png
var iconLogs []byte

//go:embed assets/icons/key.png
var iconKey []byte

//go:embed assets/icons/power.png
var iconPower []byte

//go:embed assets/icons/dot-green.png
var iconDotGreen []byte

//go:embed assets/icons/dot-red.png
var iconDotRed []byte

//go:embed assets/icons/dot-gray.png
var iconDotGray []byte
