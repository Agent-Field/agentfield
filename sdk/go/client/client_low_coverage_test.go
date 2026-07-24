package client

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Agent-Field/agentfield/sdk/go/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPostExecutionLogs_LowCoveragePaths(t *testing.T) {
	t.Run("posts logs to execution endpoint", func(t *testing.T) {
		var gotMethod string
		var gotPath string
		var gotBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			gotMethod = r.Method
			gotPath = r.URL.Path
			require.NoError(t, json.NewDecoder(r.Body).Decode(&gotBody))
			w.WriteHeader(http.StatusNoContent)
		}))
		defer server.Close()

		c, err := New(server.URL)
		require.NoError(t, err)

		err = c.PostExecutionLogs(context.Background(), "exec/1", map[string]any{
			"level":   "info",
			"message": "started",
		})
		require.NoError(t, err)

		assert.Equal(t, http.MethodPost, gotMethod)
		assert.Equal(t, "/api/v1/executions/exec%2F1/logs", gotPath)
		assert.Equal(t, "info", gotBody["level"])
		assert.Equal(t, "started", gotBody["message"])
	})

	t.Run("rejects blank execution id", func(t *testing.T) {
		c, err := New("https://example.com")
		require.NoError(t, err)

		err = c.PostExecutionLogs(context.Background(), " \t\n ", map[string]any{"x": 1})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "executionID is required")
	})
}

func TestGetNode_Error(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/api/v1/nodes/node%2Fwith%20space", r.URL.Path)
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"error":"missing node"}`))
	}))
	defer server.Close()

	c, err := New(server.URL)
	require.NoError(t, err)

	resp, err := c.GetNode(context.Background(), "node/with space")
	require.Error(t, err)
	assert.Nil(t, resp)

	apiErr, ok := err.(*APIError)
	require.True(t, ok)
	assert.Equal(t, http.StatusNotFound, apiErr.StatusCode)
}

func TestAcknowledgeAction_Error(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, http.MethodPost, r.Method)
		assert.Equal(t, "/api/v1/nodes/node-1/actions/ack", r.URL.Path)
		w.WriteHeader(http.StatusBadGateway)
		_, _ = w.Write([]byte(`bad gateway`))
	}))
	defer server.Close()

	c, err := New(server.URL)
	require.NoError(t, err)

	resp, err := c.AcknowledgeAction(context.Background(), "node-1", types.ActionAckRequest{
		ActionID: "action-1",
		Status:   "failed",
	})
	require.Error(t, err)
	assert.Nil(t, resp)

	apiErr, ok := err.(*APIError)
	require.True(t, ok)
	assert.Equal(t, http.StatusBadGateway, apiErr.StatusCode)
}

func TestShutdown_Error(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, http.MethodPost, r.Method)
		assert.Equal(t, "/api/v1/nodes/node-1/shutdown", r.URL.Path)
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":"shutdown failed"}`))
	}))
	defer server.Close()

	c, err := New(server.URL)
	require.NoError(t, err)

	resp, err := c.Shutdown(context.Background(), "node-1", types.ShutdownRequest{Reason: "test"})
	require.Error(t, err)
	assert.Nil(t, resp)

	apiErr, ok := err.(*APIError)
	require.True(t, ok)
	assert.Equal(t, http.StatusInternalServerError, apiErr.StatusCode)
}
