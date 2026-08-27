package cli

import (
	"strings"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/spf13/cobra"
)

// FlagChanged reports whether the named flag was explicitly set on the command
// line. A nil command, or a command that does not define the flag, counts as
// unset.
func FlagChanged(cmd *cobra.Command, name string) bool {
	if cmd == nil {
		return false
	}
	flag := cmd.Flags().Lookup(name)
	return flag != nil && flag.Changed
}

// ApplyConfiguredLogLevel re-initializes the global logger from the level in a
// loaded configuration, honoring the precedence flag > env > yaml.
//
// The root command installs a logger from --verbose alone, before any config
// file has been read. AGENTFIELD_LOG_LEVEL and logging.level only become
// visible once a server command has loaded its configuration
// (config.ApplyEnvOverrides folds the env var into the same field), which is
// why the level has to be applied a second time from there.
//
// An explicit --verbose keeps the debug logger the root command installed, so
// a debugging session is never silenced by configuration; an empty configured
// level also leaves the logger alone.
//
// A configured level that is not a level at all (a typo such as "warm") falls
// back to info, as it always has, but says so once at warn instead of leaving
// the operator to work out why their setting had no effect. The parser itself
// is unchanged, so no other caller starts erroring on input it used to accept.
//
// Both server entry points (cmd/af and cmd/agentfield-server) call this, so the
// two shipped binaries cannot drift apart on precedence.
func ApplyConfiguredLogLevel(cmd *cobra.Command, configuredLevel string) {
	if FlagChanged(cmd, "verbose") {
		return
	}
	level := strings.TrimSpace(configuredLevel)
	if level == "" {
		return
	}

	_, recognized := logger.ParseLevelStrict(level)
	logger.InitLoggerWithLevel(level)
	if !recognized {
		logger.Logger.Warn().
			Str("configured_level", level).
			Strs("accepted_levels", logger.AcceptedLevels()).
			Msg("unrecognized log level, falling back to info")
	}
}
