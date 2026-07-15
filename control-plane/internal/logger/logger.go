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
func ParseLevel(levelStr string) zerolog.Level {
	switch strings.ToLower(strings.TrimSpace(levelStr)) {
	case "debug", "verbose", "trace":
		return zerolog.DebugLevel
	case "warn", "warning":
		return zerolog.WarnLevel
	case "error", "err":
		return zerolog.ErrorLevel
	default:
		return zerolog.InfoLevel
	}
}
