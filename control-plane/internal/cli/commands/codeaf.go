package commands

import "github.com/Agent-Field/agentfield/control-plane/internal/codeaf"

// Synchronous install hooks share CodeAF's in-flight provisioning guard.
var ensureCodeaf = codeaf.EnsureBestEffort
