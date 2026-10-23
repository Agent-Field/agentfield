package commands

import "github.com/Agent-Field/agentfield/control-plane/internal/codeaf"

// All automatic hooks share CodeAF's once-per-process provisioning guard.
var ensureCodeaf = codeaf.EnsureBestEffort
