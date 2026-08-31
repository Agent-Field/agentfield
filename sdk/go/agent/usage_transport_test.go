package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/sdk/go/ai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newUsageTestAgent(t *testing.T) *Agent {
	t.Helper()
	a, err := New(Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: "https://api.example.com",
		Logger:        log.New(io.Discard, "", 0),
	})
	require.NoError(t, err)
	return a
}

func postJSON(t *testing.T, url string, body string) (int, []byte) {
	t.Helper()
	resp, err := http.Post(url, "application/json", bytes.NewReader([]byte(body)))
	require.NoError(t, err)
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	return resp.StatusCode, raw
}

// TestSyncReasonerBodyCarriesUsageEnvelope maps to the contract: when a sync
// reasoner records usage and returns an object result, the 200 body carries
// the serialized usage under the reserved "__agentfield_usage__" sibling key,
// and a user-owned "usage" key in the result is never touched.
func TestSyncReasonerBodyCarriesUsageEnvelope(t *testing.T) {
	a := newUsageTestAgent(t)
	cost := 0.0123
	a.RegisterReasoner("summarize", func(ctx context.Context, _ map[string]any) (any, error) {
		CostTrackerFrom(ctx).Record(CostEntry{
			Model:        "anthropic/claude-opus-4-8",
			InputTokens:  100,
			OutputTokens: 50,
			TotalTokens:  150,
			CostUSD:      &cost,
			CostSource:   "provider",
		})
		return map[string]any{"answer": 42, "usage": map[string]any{"user": "data"}}, nil
	})

	server := httptest.NewServer(a.handler())
	defer server.Close()

	status, body := postJSON(t, server.URL+"/reasoners/summarize", `{}`)
	require.Equal(t, http.StatusOK, status)

	var result map[string]any
	require.NoError(t, json.Unmarshal(body, &result))

	assert.Equal(t, float64(42), result["answer"], "result content must be undisturbed")
	userUsage, ok := result["usage"].(map[string]any)
	require.True(t, ok, "user-owned usage key must be preserved")
	assert.Equal(t, "data", userUsage["user"])

	envelope, ok := result[UsageEnvelopeKey].(map[string]any)
	require.True(t, ok, "usage envelope must be merged as a sibling key")
	assert.Equal(t, float64(0.0123), envelope["total_cost_usd"])
	assert.Equal(t, float64(150), envelope["total_tokens"])
	entries, ok := envelope["entries"].([]any)
	require.True(t, ok)
	require.Len(t, entries, 1)
	entry := entries[0].(map[string]any)
	assert.Equal(t, "anthropic", entry["provider"])
	// Manually recorded entries carry only what the caller set: reasoner is
	// null here, and the control plane backfills it from the execution's
	// reasoner ID (see the control-plane golden contract test).
	assert.Nil(t, entry["reasoner"])
	assert.Equal(t, "provider", entry["cost_source"])
}

// TestSyncReasonerNonObjectResultPassesThrough maps to the contract:
// non-object results (arrays, scalars) cannot carry a sibling key and must
// pass through byte-identical in shape — no envelope, no wrapping.
func TestSyncReasonerNonObjectResultPassesThrough(t *testing.T) {
	tests := []struct {
		name   string
		result any
		verify func(t *testing.T, body []byte)
	}{
		{
			name:   "array result",
			result: []any{"a", "b"},
			verify: func(t *testing.T, body []byte) {
				var arr []any
				require.NoError(t, json.Unmarshal(body, &arr))
				assert.Equal(t, []any{"a", "b"}, arr)
			},
		},
		{
			name:   "scalar result",
			result: "plain text",
			verify: func(t *testing.T, body []byte) {
				var s string
				require.NoError(t, json.Unmarshal(body, &s))
				assert.Equal(t, "plain text", s)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := newUsageTestAgent(t)
			a.RegisterReasoner("r", func(ctx context.Context, _ map[string]any) (any, error) {
				CostTrackerFrom(ctx).Record(CostEntry{Model: "gpt-4o", InputTokens: 1})
				return tt.result, nil
			})
			server := httptest.NewServer(a.handler())
			defer server.Close()

			status, body := postJSON(t, server.URL+"/reasoners/r", `{}`)
			require.Equal(t, http.StatusOK, status)
			assert.NotContains(t, string(body), UsageEnvelopeKey)
			tt.verify(t, body)
		})
	}
}

// TestSyncReasonerNoUsageNoEnvelope maps to the contract: when no usage was
// recorded, the body is unchanged — the envelope key is omitted entirely.
func TestSyncReasonerNoUsageNoEnvelope(t *testing.T) {
	a := newUsageTestAgent(t)
	a.RegisterReasoner("plain", func(_ context.Context, _ map[string]any) (any, error) {
		return map[string]any{"ok": true}, nil
	})
	server := httptest.NewServer(a.handler())
	defer server.Close()

	status, body := postJSON(t, server.URL+"/reasoners/plain", `{}`)
	require.Equal(t, http.StatusOK, status)

	var result map[string]any
	require.NoError(t, json.Unmarshal(body, &result))
	assert.Equal(t, true, result["ok"])
	_, present := result[UsageEnvelopeKey]
	assert.False(t, present, "no-usage results must not carry an envelope key")
}

// TestSyncSkillAndExecuteBodiesCarryUsageEnvelope maps to the contract: the
// /skills/ and /execute endpoints bind their own per-execution trackers and
// attach usage to object results the same way /reasoners/ does.
func TestSyncSkillAndExecuteBodiesCarryUsageEnvelope(t *testing.T) {
	a := newUsageTestAgent(t)
	handler := func(ctx context.Context, _ map[string]any) (any, error) {
		CostTrackerFrom(ctx).Record(CostEntry{Model: "openrouter/qwen/qwen3-coder", InputTokens: 5, OutputTokens: 2})
		return map[string]any{"done": true}, nil
	}
	require.NoError(t, a.RegisterSkill("fix", handler))
	a.RegisterReasoner("think", handler)

	server := httptest.NewServer(a.handler())
	defer server.Close()

	for _, path := range []string{"/skills/fix", "/execute/think"} {
		t.Run(path, func(t *testing.T) {
			status, body := postJSON(t, server.URL+path, `{"input":{}}`)
			require.Equal(t, http.StatusOK, status)

			var result map[string]any
			require.NoError(t, json.Unmarshal(body, &result))
			assert.Equal(t, true, result["done"])
			envelope, ok := result[UsageEnvelopeKey].(map[string]any)
			require.True(t, ok, "envelope missing from %s body", path)
			assert.Equal(t, float64(7), envelope["total_tokens"])
		})
	}
}

// TestConcurrentExecutionsHaveIsolatedTrackers maps to the contract:
// concurrent executions each get a fresh tracker — usage recorded by one
// in-flight execution never leaks into another's response.
func TestConcurrentExecutionsHaveIsolatedTrackers(t *testing.T) {
	a := newUsageTestAgent(t)
	started := make(chan struct{}, 2)
	release := make(chan struct{})
	makeHandler := func(model string) HandlerFunc {
		return func(ctx context.Context, _ map[string]any) (any, error) {
			CostTrackerFrom(ctx).Record(CostEntry{Model: model, InputTokens: 1, OutputTokens: 1})
			started <- struct{}{}
			<-release
			return map[string]any{"model": model}, nil
		}
	}
	a.RegisterReasoner("alpha", makeHandler("model-alpha"))
	a.RegisterReasoner("beta", makeHandler("model-beta"))

	server := httptest.NewServer(a.handler())
	defer server.Close()

	type outcome struct {
		name string
		body []byte
	}
	results := make(chan outcome, 2)
	var wg sync.WaitGroup
	for _, name := range []string{"alpha", "beta"} {
		wg.Add(1)
		go func(name string) {
			defer wg.Done()
			status, body := postJSON(t, server.URL+"/reasoners/"+name, `{}`)
			assert.Equal(t, http.StatusOK, status)
			results <- outcome{name: name, body: body}
		}(name)
	}

	// Both executions are in-flight (and have recorded) before either
	// completes, proving tracker isolation rather than luck of scheduling.
	for i := 0; i < 2; i++ {
		select {
		case <-started:
		case <-time.After(5 * time.Second):
			t.Fatal("timed out waiting for concurrent executions to start")
		}
	}
	close(release)
	wg.Wait()
	close(results)

	for res := range results {
		var body map[string]any
		require.NoError(t, json.Unmarshal(res.body, &body))
		envelope, ok := body[UsageEnvelopeKey].(map[string]any)
		require.True(t, ok)
		entries := envelope["entries"].([]any)
		require.Len(t, entries, 1, "each execution must see exactly its own entry")
		entry := entries[0].(map[string]any)
		assert.Equal(t, "model-"+res.name, entry["model"])
	}
}

// TestAsyncStatusCarriesUsage maps to the contract: the async terminal status
// callback carries the "usage" object on success AND failure, and omits the
// key entirely when nothing was recorded.
func TestAsyncStatusCarriesUsage(t *testing.T) {
	cost := 0.25
	recordingHandler := func(ctx context.Context, _ map[string]any) (any, error) {
		CostTrackerFrom(ctx).Record(CostEntry{
			Model:       "anthropic/claude-opus-4-8",
			InputTokens: 10, OutputTokens: 4,
			CostUSD: &cost, CostSource: "provider",
		})
		return map[string]any{"ok": true}, nil
	}

	tests := []struct {
		name       string
		handler    HandlerFunc
		wantStatus string
		wantUsage  bool
	}{
		{
			name:       "succeeded payload carries usage",
			handler:    recordingHandler,
			wantStatus: "succeeded",
			wantUsage:  true,
		},
		{
			name: "failed payload carries usage",
			handler: func(ctx context.Context, input map[string]any) (any, error) {
				_, _ = recordingHandler(ctx, input)
				return nil, &ReasonerFailed{Message: "went wrong"}
			},
			wantStatus: "failed",
			wantUsage:  true,
		},
		{
			name: "no recorded usage omits the key",
			handler: func(_ context.Context, _ map[string]any) (any, error) {
				return map[string]any{"ok": true}, nil
			},
			wantStatus: "succeeded",
			wantUsage:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			statusCh := make(chan map[string]any, 1)
			a, server := newStatusCapturingAgent(t, statusCh, false, nil)
			defer server.Close()

			a.executeReasonerAsync(
				&Reasoner{Name: "worker", Handler: tt.handler},
				map[string]any{},
				ExecutionContext{ExecutionID: "exec-u1", RunID: "run-u1", WorkflowID: "wf-u1"},
			)

			select {
			case payload := <-statusCh:
				assert.Equal(t, tt.wantStatus, payload["status"])
				usageRaw, present := payload["usage"]
				if !tt.wantUsage {
					assert.False(t, present, "usage key must be omitted when nothing was recorded")
					return
				}
				usage, ok := usageRaw.(map[string]any)
				require.True(t, ok, "usage must be an object")
				assert.Equal(t, float64(0.25), usage["total_cost_usd"])
				assert.Equal(t, float64(14), usage["total_tokens"])
				entries := usage["entries"].([]any)
				require.Len(t, entries, 1)
				entry := entries[0].(map[string]any)
				assert.Equal(t, "anthropic", entry["provider"])
				assert.Equal(t, "provider", entry["cost_source"])
			case <-time.After(3 * time.Second):
				t.Fatal("timed out waiting for status callback")
			}
		})
	}
}

// TestAIRecordsUsageIntoTracker maps to the contract: an Agent.AI call
// records the response's usage — including OpenRouter's native cost and cache
// fields — into the current execution's tracker.
func TestAIRecordsUsageIntoTracker(t *testing.T) {
	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{
			"id": "1", "model": "openrouter/qwen/qwen3-coder",
			"choices": [{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],
			"usage": {
				"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
				"prompt_tokens_details": {"cached_tokens": 7},
				"cost": 0.0042
			}
		}`))
	}))
	defer llmServer.Close()

	a, err := New(Config{
		NodeID:  "node-1",
		Version: "1.0.0",
		Logger:  log.New(io.Discard, "", 0),
		AIConfig: &ai.Config{
			APIKey:  "test-key",
			BaseURL: llmServer.URL,
			Model:   "openrouter/qwen/qwen3-coder",
			Timeout: 5 * time.Second,
		},
	})
	require.NoError(t, err)

	tracker := NewCostTracker()
	ctx := contextWithCostTracker(context.Background(), tracker)
	ctx = contextWithExecution(ctx, ExecutionContext{ReasonerName: "code"})

	_, err = a.AI(ctx, "hello")
	require.NoError(t, err)

	usage := tracker.Serialize()
	assert.Equal(t, 0.0042, usage["total_cost_usd"])
	entries := usage["entries"].([]map[string]any)
	require.Len(t, entries, 1)
	assert.Equal(t, "openrouter/qwen/qwen3-coder", entries[0]["model"])
	assert.Equal(t, "openrouter", entries[0]["provider"])
	assert.Equal(t, "code", entries[0]["reasoner"])
	assert.Equal(t, 100, entries[0]["input_tokens"])
	assert.Equal(t, 50, entries[0]["output_tokens"])
	assert.Equal(t, 7, entries[0]["cache_read_tokens"])
	assert.Equal(t, "provider", entries[0]["cost_source"])
}

// TestAIStreamRecordsFinalUsageChunk maps to the contract: streamed calls
// record the terminal usage chunk (when the provider sends one) after the
// stream completes, without disturbing the streamed content.
func TestAIStreamRecordsFinalUsageChunk(t *testing.T) {
	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(`data: {"id":"1","model":"stream-model","choices":[{"index":0,"delta":{"content":"hel"}}]}` + "\n\n"))
		_, _ = w.Write([]byte(`data: {"id":"1","model":"stream-model","choices":[{"index":0,"delta":{"content":"lo"}}]}` + "\n\n"))
		_, _ = w.Write([]byte(`data: {"id":"1","model":"stream-model","choices":[],"usage":{"prompt_tokens":9,"completion_tokens":2,"total_tokens":11}}` + "\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer llmServer.Close()

	a, err := New(Config{
		NodeID:  "node-1",
		Version: "1.0.0",
		Logger:  log.New(io.Discard, "", 0),
		AIConfig: &ai.Config{
			APIKey:  "test-key",
			BaseURL: llmServer.URL,
			Model:   "gpt-4o",
			Timeout: 5 * time.Second,
		},
	})
	require.NoError(t, err)

	tracker := NewCostTracker()
	ctx := contextWithCostTracker(context.Background(), tracker)

	chunks, errs := a.AIStream(ctx, "hello")
	var text string
	for chunk := range chunks {
		for _, choice := range chunk.Choices {
			text += choice.Delta.Content
		}
	}
	require.NoError(t, <-errs)

	assert.Equal(t, "hello", text, "streamed content must be forwarded unchanged")
	usage := tracker.Serialize()
	entries := usage["entries"].([]map[string]any)
	require.Len(t, entries, 1)
	assert.Equal(t, "stream-model", entries[0]["model"])
	assert.Equal(t, 9, entries[0]["input_tokens"])
	assert.Equal(t, 2, entries[0]["output_tokens"])
	assert.Equal(t, 11, entries[0]["total_tokens"])
	assert.Nil(t, usage["total_cost_usd"], "no provider cost -> null")
}

// TestAIStreamWithoutTrackerPassesThrough maps to the contract: when no
// tracker is bound the stream is returned untouched (no wrapping goroutine).
func TestAIStreamWithoutTrackerPassesThrough(t *testing.T) {
	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(`data: {"id":"1","model":"m","choices":[{"index":0,"delta":{"content":"ok"}}]}` + "\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer llmServer.Close()

	a, err := New(Config{
		NodeID:  "node-1",
		Version: "1.0.0",
		Logger:  log.New(io.Discard, "", 0),
		AIConfig: &ai.Config{
			APIKey:  "test-key",
			BaseURL: llmServer.URL,
			Model:   "gpt-4o",
			Timeout: 5 * time.Second,
		},
	})
	require.NoError(t, err)

	chunks, errs := a.AIStream(context.Background(), "hello")
	var text string
	for chunk := range chunks {
		for _, choice := range chunk.Choices {
			text += choice.Delta.Content
		}
	}
	require.NoError(t, <-errs)
	assert.Equal(t, "ok", text)
}
