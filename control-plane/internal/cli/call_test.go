package cli

import (
	"bytes"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestResolveCallInputPrecedence(t *testing.T) {
	schema := map[string]interface{}{
		"type":     "object",
		"required": []interface{}{"message"},
		"properties": map[string]interface{}{
			"message": map[string]interface{}{"type": "string"},
		},
	}

	t.Run("flag beats stdin", func(t *testing.T) {
		input, err := resolveCallInput(&callOptions{
			inputSource: `{"message":"flag"}`,
			stdin:       bytes.NewBufferString(`{"message":"stdin"}`),
			stdinTTY:    false,
		}, schema, nil)
		require.NoError(t, err)
		require.Equal(t, "flag", input["message"])
	})

	t.Run("stdin beats prompt", func(t *testing.T) {
		input, err := resolveCallInput(&callOptions{
			stdin:    bytes.NewBufferString(`{"message":"stdin"}`),
			stdinTTY: false,
		}, schema, nil)
		require.NoError(t, err)
		require.Equal(t, "stdin", input["message"])
	})

	t.Run("interactive prompt", func(t *testing.T) {
		input, err := resolveCallInput(&callOptions{
			stdin:    bytes.NewBufferString("prompted\ny\n"),
			stderr:   bytes.NewBuffer(nil),
			stdinTTY: true,
		}, schema, nil)
		require.NoError(t, err)
		require.Equal(t, "prompted", input["message"])
	})

	t.Run("non tty required input exits 2", func(t *testing.T) {
		_, err := resolveCallInput(&callOptions{
			stdin:    bytes.NewBuffer(nil),
			stdinTTY: false,
		}, schema, nil)
		var exitErr cliExitError
		require.ErrorAs(t, err, &exitErr)
		require.Equal(t, 2, exitErr.Code)
	})

	t.Run("empty schema calls with defaults", func(t *testing.T) {
		input, err := resolveCallInput(&callOptions{
			stdin:    bytes.NewBuffer(nil),
			stdinTTY: false,
		}, map[string]interface{}{}, nil)
		require.NoError(t, err)
		require.Empty(t, input)
	})
}

func TestValidateInputAgainstSchema(t *testing.T) {
	schema := map[string]interface{}{
		"required": []interface{}{"message", "count"},
		"properties": map[string]interface{}{
			"message": map[string]interface{}{"type": "string"},
			"count":   map[string]interface{}{"type": "integer"},
		},
	}

	require.NoError(t, validateInputAgainstSchema(map[string]interface{}{
		"message": "ok",
		"count":   float64(2),
	}, schema))

	err := validateInputAgainstSchema(map[string]interface{}{"message": "ok"}, schema)
	require.ErrorContains(t, err, `missing required field "count"`)

	err = validateInputAgainstSchema(map[string]interface{}{
		"message": "ok",
		"count":   "two",
	}, schema)
	require.ErrorContains(t, err, `field "count" must be an integer`)
}

func TestExitCodeMapping(t *testing.T) {
	require.Equal(t, 3, httpExitCode(500))
	require.Equal(t, 3, httpExitCode(401))
	require.Equal(t, 2, httpExitCode(400))
	require.Equal(t, 0, httpExitCode(200))
	require.Equal(t, 2, ExitCode(cliExitError{Code: 2, Err: errors.New("bad input")}))
	require.Equal(t, 1, ExitCode(errors.New("generic")))
}

func TestTriggerHTTPClientTimeouts(t *testing.T) {
	originalTimeout := requestTimeout
	t.Cleanup(func() { requestTimeout = originalTimeout })

	requestTimeout = 7
	require.Equal(t, 7*time.Second, triggerHTTPClient("application/json").Timeout)
	require.Zero(t, triggerHTTPClient("text/event-stream").Timeout)

	requestTimeout = 0
	require.Equal(t, 30*time.Second, triggerHTTPClient("application/json").Timeout)
}
