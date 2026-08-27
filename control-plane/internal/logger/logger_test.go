package logger

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/rs/zerolog"
	"github.com/stretchr/testify/require"
)

func TestInitLoggerSetsLevel(t *testing.T) {
	InitLogger(false)
	require.Equal(t, zerolog.InfoLevel, Logger.GetLevel())

	InitLogger(true)
	require.Equal(t, zerolog.DebugLevel, Logger.GetLevel())
}

func TestInitLoggerWithLevel(t *testing.T) {
	InitLoggerWithLevel("debug")
	require.Equal(t, zerolog.DebugLevel, Logger.GetLevel())

	InitLoggerWithLevel("info")
	require.Equal(t, zerolog.InfoLevel, Logger.GetLevel())

	InitLoggerWithLevel("warn")
	require.Equal(t, zerolog.WarnLevel, Logger.GetLevel())

	InitLoggerWithLevel("error")
	require.Equal(t, zerolog.ErrorLevel, Logger.GetLevel())

	// Unknown defaults to info
	InitLoggerWithLevel("unknown")
	require.Equal(t, zerolog.InfoLevel, Logger.GetLevel())

	// Empty string defaults to info
	InitLoggerWithLevel("")
	require.Equal(t, zerolog.InfoLevel, Logger.GetLevel())
}

func TestParseLevel(t *testing.T) {
	tests := []struct {
		input    string
		expected zerolog.Level
	}{
		{"debug", zerolog.DebugLevel},
		{"DEBUG", zerolog.DebugLevel},
		{"verbose", zerolog.DebugLevel},
		{"trace", zerolog.DebugLevel},
		{"info", zerolog.InfoLevel},
		{"INFO", zerolog.InfoLevel},
		{"warn", zerolog.WarnLevel},
		{"warning", zerolog.WarnLevel},
		{"error", zerolog.ErrorLevel},
		{"err", zerolog.ErrorLevel},
		{" info ", zerolog.InfoLevel},
		{"", zerolog.InfoLevel},
		{"invalid", zerolog.InfoLevel},
	}
	for _, tt := range tests {
		require.Equal(t, tt.expected, ParseLevel(tt.input), "ParseLevel(%q)", tt.input)
	}
}

func TestFormattedHelpers(t *testing.T) {
	var buf bytes.Buffer
	Logger = zerolog.New(&buf).With().Timestamp().Logger().Level(zerolog.DebugLevel)

	Infof("hello %s", "world")
	Debugf("debug %d", 1)
	Warnf("warn %s", "value")
	Errorf("error %s", "value")
	Successf("done %d", 2)

	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	require.Len(t, lines, 5)

	messages := make([]map[string]interface{}, 0, len(lines))
	for _, line := range lines {
		entry := make(map[string]interface{})
		require.NoError(t, json.Unmarshal([]byte(line), &entry))
		messages = append(messages, entry)
	}

	require.Equal(t, "info", messages[0]["level"])
	require.Equal(t, "hello world", messages[0]["message"])

	require.Equal(t, "debug", messages[1]["level"])
	require.Equal(t, "debug 1", messages[1]["message"])

	require.Equal(t, "warn", messages[2]["level"])
	require.Equal(t, "warn value", messages[2]["message"])

	require.Equal(t, "error", messages[3]["level"])
	require.Equal(t, "error value", messages[3]["message"])

	require.Equal(t, "info", messages[4]["level"])
	require.Equal(t, "✅ done 2", messages[4]["message"])
}

// TestParseLevelStrictReportsUnknownValues covers contract V7: the strict
// parser tells a caller whether the string was a level at all, while still
// yielding info for anything it does not know, and ParseLevel keeps behaving
// exactly as it did for every other caller.
func TestParseLevelStrictReportsUnknownValues(t *testing.T) {
	recognized := map[string]zerolog.Level{
		"debug":   zerolog.DebugLevel,
		"verbose": zerolog.DebugLevel,
		"trace":   zerolog.DebugLevel,
		"info":    zerolog.InfoLevel,
		"INFO":    zerolog.InfoLevel,
		" warn ":  zerolog.WarnLevel,
		"warning": zerolog.WarnLevel,
		"error":   zerolog.ErrorLevel,
		"err":     zerolog.ErrorLevel,
	}
	for in, want := range recognized {
		level, ok := ParseLevelStrict(in)
		require.True(t, ok, "%q must be recognized", in)
		require.Equal(t, want, level, "%q", in)
		require.Equal(t, want, ParseLevel(in), "ParseLevel must agree for %q", in)
	}

	for _, in := range []string{"", "   ", "warm", "infoo", "silent", "5"} {
		level, ok := ParseLevelStrict(in)
		require.False(t, ok, "%q must be reported as unrecognized", in)
		require.Equal(t, zerolog.InfoLevel, level, "%q must still fall back to info", in)
		require.Equal(t, zerolog.InfoLevel, ParseLevel(in), "ParseLevel must be unchanged for %q", in)
	}
}

// TestAcceptedLevelsListsTheCanonicalNames covers the operator-facing half of
// V5: the names offered in the "unrecognized log level" report are exactly the
// canonical ones, in increasing severity, and each one actually parses.
func TestAcceptedLevelsListsTheCanonicalNames(t *testing.T) {
	require.Equal(t, []string{"debug", "info", "warn", "error"}, AcceptedLevels())
	for _, name := range AcceptedLevels() {
		_, ok := ParseLevelStrict(name)
		require.True(t, ok, "advertised level %q must parse", name)
	}
}
