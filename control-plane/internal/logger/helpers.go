// Package logger provides printf-style helpers for zerolog.
package logger

import (
	"fmt"
)

// Infof logs an info-level message with formatting.
func Infof(format string, args ...interface{}) {
	Logger.Info().Msg(fmt.Sprintf(format, args...))
}

// Debugf logs a debug-level message with formatting.
func Debugf(format string, args ...interface{}) {
	Logger.Debug().Msg(fmt.Sprintf(format, args...))
}

// Warnf logs a warning-level message with formatting.
func Warnf(format string, args ...interface{}) {
	Logger.Warn().Msg(fmt.Sprintf(format, args...))
}

// Errorf logs an error-level message with formatting.
func Errorf(format string, args ...interface{}) {
	Logger.Error().Msg(fmt.Sprintf(format, args...))
}

// Successf logs a success-level message (info with a checkmark).
func Successf(format string, args ...interface{}) {
	Logger.Info().Msg("✅ " + fmt.Sprintf(format, args...))
}
