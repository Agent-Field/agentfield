// Package logger provides a global zerolog logger for the AgentField CLI.
package logger

import (
	"os"
	"strings"

	"github.com/rs/zerolog"
)

var (
	// Logger is the global zerolog logger instance.
	Logger zerolog.Logger
)

// InitLogger initializes the global logger with the specified log level.
// Kept for backward compatibility with the CLI --verbose flag.
func InitLogger(verbose bool) {
	level := zerolog.InfoLevel
	if verbose {
		level = zerolog.DebugLevel
	}
	Logger = zerolog.New(os.Stderr).With().Timestamp().Logger().Level(level)
}

// InitLoggerWithLevel initializes the global logger from a level string.
// Accepted values: "debug", "info", "warn", "error". Falls back to info.
func InitLoggerWithLevel(levelStr string) {
	level := ParseLevel(levelStr)
	Logger = zerolog.New(os.Stderr).With().Timestamp().Logger().Level(level)
}

// ParseLevel converts a human-friendly level string to a zerolog.Level.
// Anything it does not recognize — including the empty string — becomes info.
// Callers that need to tell "the operator asked for info" apart from "the
// operator asked for something meaningless" should use ParseLevelStrict.
func ParseLevel(levelStr string) zerolog.Level {
	level, _ := ParseLevelStrict(levelStr)
	return level
}

// ParseLevelStrict converts a human-friendly level string to a zerolog.Level
// and reports whether the string was recognized. An unrecognized value still
// yields info, so callers can log the fallback and carry on rather than
// failing to start.
func ParseLevelStrict(levelStr string) (zerolog.Level, bool) {
	switch strings.ToLower(strings.TrimSpace(levelStr)) {
	case "debug", "verbose", "trace":
		return zerolog.DebugLevel, true
	case "info":
		return zerolog.InfoLevel, true
	case "warn", "warning":
		return zerolog.WarnLevel, true
	case "error", "err":
		return zerolog.ErrorLevel, true
	default:
		return zerolog.InfoLevel, false
	}
}

// AcceptedLevels lists the canonical level names, in increasing severity, for
// use in operator-facing messages. The aliases ParseLevelStrict also accepts
// (verbose, trace, warning, err) are deliberately not advertised.
func AcceptedLevels() []string {
	return []string{"debug", "info", "warn", "error"}
}
