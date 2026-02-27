package client

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ---------------------------------------------------------------------------
// RequestApproval
// ---------------------------------------------------------------------------

func TestRequestApproval(t *testing.T) {
	var receivedBody map[string]interface{}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, http.MethodPost, r.Method)
		assert.Contains(t, r.URL.Path, "/request-approval")

		_ = json.NewDecoder(r.Body).Decode(&receivedBody)

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]string{
			"approval_request_id":  "req-abc",
			"approval_request_url": "https://hub.example.com/r/req-abc",
		})
	}))
	defer server.Close()

	c, err := New(server.URL, WithBearerToken("tok"))
	require.NoError(t, err)

	resp, err := c.RequestApproval(context.Background(), "node-1", "exec-1", RequestApprovalRequest{
		Title:        "Plan Review",
		TemplateType: "plan-review-v1",
		ProjectID:    "proj-1",
	})

	require.NoError(t, err)
	assert.Equal(t, "req-abc", resp.ApprovalRequestID)
	assert.Equal(t, "https://hub.example.com/r/req-abc", resp.ApprovalRequestURL)

	// Verify the request body was sent correctly
	assert.Equal(t, "Plan Review", receivedBody["title"])
	assert.Equal(t, "plan-review-v1", receivedBody["template_type"])
	assert.Equal(t, "proj-1", receivedBody["project_id"])
}

func TestRequestApproval_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"error":"not found"}`))
	}))
	defer server.Close()

	c, err := New(server.URL)
	require.NoError(t, err)

	_, err = c.RequestApproval(context.Background(), "node-1", "exec-1", RequestApprovalRequest{
		ProjectID: "p",
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "request approval")
}

// ---------------------------------------------------------------------------
// GetApprovalStatus
// ---------------------------------------------------------------------------

func TestGetApprovalStatus(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, http.MethodGet, r.Method)
		assert.Contains(t, r.URL.Path, "/approval-status")

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"status":       "approved",
			"response":     map[string]string{"decision": "approved", "feedback": "LGTM"},
			"request_url":  "https://hub.example.com/r/req-abc",
			"requested_at": "2026-02-25T10:00:00Z",
			"responded_at": "2026-02-25T11:00:00Z",
		})
	}))
	defer server.Close()

	c, err := New(server.URL)
	require.NoError(t, err)

	resp, err := c.GetApprovalStatus(context.Background(), "node-1", "exec-1")
	require.NoError(t, err)

	assert.Equal(t, "approved", resp.Status)
	assert.Equal(t, "https://hub.example.com/r/req-abc", resp.RequestURL)
	assert.Equal(t, "2026-02-25T10:00:00Z", resp.RequestedAt)
	assert.Equal(t, "2026-02-25T11:00:00Z", resp.RespondedAt)
	assert.NotNil(t, resp.Response)
}

func TestGetApprovalStatus_Pending(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"status":       "pending",
			"request_url":  "https://hub.example.com/r/req-abc",
			"requested_at": "2026-02-25T10:00:00Z",
		})
	}))
	defer server.Close()

	c, err := New(server.URL)
	require.NoError(t, err)

	resp, err := c.GetApprovalStatus(context.Background(), "node-1", "exec-1")
	require.NoError(t, err)
	assert.Equal(t, "pending", resp.Status)
	assert.Empty(t, resp.RespondedAt)
}

func TestGetApprovalStatus_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":"internal"}`))
	}))
	defer server.Close()

	c, err := New(server.URL)
	require.NoError(t, err)

	_, err = c.GetApprovalStatus(context.Background(), "node-1", "exec-1")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "get approval status")
}

// ---------------------------------------------------------------------------
// WaitForApproval
// ---------------------------------------------------------------------------

func TestWaitForApproval_ResolvesOnApproved(t *testing.T) {
	var callCount atomic.Int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := callCount.Add(1)
		w.Header().Set("Content-Type", "application/json")
		if n == 1 {
			_ = json.NewEncoder(w).Encode(map[string]string{"status": "pending"})
		} else {
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"status":   "approved",
				"response": map[string]string{"decision": "approved"},
			})
		}
	}))
	defer server.Close()

	c, err := New(server.URL)
	require.NoError(t, err)

	resp, err := c.WaitForApproval(context.Background(), "node-1", "exec-1", &WaitForApprovalOptions{
		PollInterval: 10 * time.Millisecond,
		MaxInterval:  20 * time.Millisecond,
	})

	require.NoError(t, err)
	assert.Equal(t, "approved", resp.Status)
	assert.GreaterOrEqual(t, callCount.Load(), int32(2))
}

func TestWaitForApproval_ResolvesOnRejected(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]string{"status": "rejected"})
	}))
	defer server.Close()

	c, err := New(server.URL)
	require.NoError(t, err)

	resp, err := c.WaitForApproval(context.Background(), "node-1", "exec-1", &WaitForApprovalOptions{
		PollInterval: 10 * time.Millisecond,
	})

	require.NoError(t, err)
	assert.Equal(t, "rejected", resp.Status)
}

func TestWaitForApproval_ContextCancellation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]string{"status": "pending"})
	}))
	defer server.Close()

	c, err := New(server.URL)
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	_, err = c.WaitForApproval(ctx, "node-1", "exec-1", &WaitForApprovalOptions{
		PollInterval: 10 * time.Millisecond,
		MaxInterval:  10 * time.Millisecond,
	})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "wait for approval")
}

func TestWaitForApproval_RetriesOnTransientError(t *testing.T) {
	var callCount atomic.Int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := callCount.Add(1)
		if n == 1 {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"error":"transient"}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]string{"status": "approved"})
	}))
	defer server.Close()

	c, err := New(server.URL)
	require.NoError(t, err)

	resp, err := c.WaitForApproval(context.Background(), "node-1", "exec-1", &WaitForApprovalOptions{
		PollInterval: 10 * time.Millisecond,
		MaxInterval:  20 * time.Millisecond,
	})

	require.NoError(t, err)
	assert.Equal(t, "approved", resp.Status)
	assert.GreaterOrEqual(t, callCount.Load(), int32(2))
}

func TestWaitForApproval_ResolvesOnExpired(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"status":       "expired",
			"request_url":  "https://hub.example.com/r/req-abc",
			"requested_at": "2026-02-25T10:00:00Z",
			"responded_at": "2026-02-28T10:00:00Z",
		})
	}))
	defer server.Close()

	c, err := New(server.URL)
	require.NoError(t, err)

	resp, err := c.WaitForApproval(context.Background(), "node-1", "exec-1", &WaitForApprovalOptions{
		PollInterval: 10 * time.Millisecond,
	})

	require.NoError(t, err)
	assert.Equal(t, "expired", resp.Status)
}

func TestGetApprovalStatus_Expired(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"status":       "expired",
			"request_url":  "https://hub.example.com/r/req-abc",
			"requested_at": "2026-02-25T10:00:00Z",
			"responded_at": "2026-02-28T10:00:00Z",
		})
	}))
	defer server.Close()

	c, err := New(server.URL)
	require.NoError(t, err)

	resp, err := c.GetApprovalStatus(context.Background(), "node-1", "exec-1")
	require.NoError(t, err)
	assert.Equal(t, "expired", resp.Status)
}

func TestWaitForApproval_DefaultOptions(t *testing.T) {
	opts := WaitForApprovalOptions{}
	opts.defaults()

	assert.Equal(t, 5*time.Second, opts.PollInterval)
	assert.Equal(t, 60*time.Second, opts.MaxInterval)
	assert.Equal(t, 2.0, opts.BackoffFactor)
}
