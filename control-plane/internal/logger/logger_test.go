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
