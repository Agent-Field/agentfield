// Package all bundles every first-party Source via blank imports. A single
// import of this package wires the full default catalog into the registry, so
// callers (typically server.go) do not need to know which sources exist.
package all

import (
	_ "github.com/Agent-Field/agentfield/control-plane/internal/sources/cron"
	_ "github.com/Agent-Field/agentfield/control-plane/internal/sources/genericbearer"
	_ "github.com/Agent-Field/agentfield/control-plane/internal/sources/generichmac"
	_ "github.com/Agent-Field/agentfield/control-plane/internal/sources/github"
	_ "github.com/Agent-Field/agentfield/control-plane/internal/sources/slack"
	_ "github.com/Agent-Field/agentfield/control-plane/internal/sources/stripe"
)
