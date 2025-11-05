// Package logger provides a global zerolog logger for the AgentField CLI.
package logger

import (
	"os"

	"github.com/rs/zerolog"
)

var (
	// Logger is the global zerolog logger instance.
	Logger zerolog.Logger
)

// InitLogger initializes the global logger with the specified log level.
func InitLogger(verbose bool) {
	level := zerolog.InfoLevel
	if verbose {
		level = zerolog.DebugLevel
	}
	Logger = zerolog.New(os.Stderr).With().Timestamp().Logger().Level(level)
}
