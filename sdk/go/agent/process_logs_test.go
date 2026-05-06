package agent

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func testAgentWithProcessLogRing(ring *processLogRing) *Agent {
	a := &Agent{procLogRing: ring}
	a.procLogOnce.Do(func() {})
	return a
}

func decodeProcessLogEntries(t *testing.T, body string) []processLogEntry {
	t.Helper()
	trimmed := strings.TrimSpace(body)
	if trimmed == "" {
		return nil
	}
	lines := strings.Split(trimmed, "\n")
	entries := make([]processLogEntry, 0, len(lines))
	for _, line := range lines {
		if strings.TrimSpace(line) == "" {
			continue
		}
		var entry processLogEntry
		if err := json.Unmarshal([]byte(line), &entry); err != nil {
			t.Fatalf("unmarshal process log entry: %v", err)
		}
		entries = append(entries, entry)
	}
	return entries
}

func TestProcessLogs_Ring(t *testing.T) {
	t.Run("config parsing and auth", func(t *testing.T) {
		t.Setenv("AGENTFIELD_LOG_BUFFER_BYTES", "invalid")
		if got := processLogsMaxBytes(); got != 4<<20 {
			t.Fatalf("processLogsMaxBytes invalid = %d", got)
		}

		t.Setenv("AGENTFIELD_LOG_BUFFER_BYTES", "2048")
		if got := processLogsMaxBytes(); got != 2048 {
			t.Fatalf("processLogsMaxBytes = %d", got)
		}

		t.Setenv("AGENTFIELD_LOG_MAX_LINE_BYTES", "100")
		if got := processLogsMaxLineBytes(); got != 16384 {
			t.Fatalf("processLogsMaxLineBytes invalid = %d", got)
		}

		t.Setenv("AGENTFIELD_LOG_MAX_LINE_BYTES", "512")
		if got := processLogsMaxLineBytes(); got != 512 {
			t.Fatalf("processLogsMaxLineBytes = %d", got)
		}

		t.Setenv("AGENTFIELD_LOG_MAX_TAIL_LINES", "0")
		if got := processLogsMaxTailLines(); got != 50000 {
			t.Fatalf("processLogsMaxTailLines invalid = %d", got)
		}

		t.Setenv("AGENTFIELD_LOG_MAX_TAIL_LINES", "12")
		if got := processLogsMaxTailLines(); got != 12 {
			t.Fatalf("processLogsMaxTailLines = %d", got)
		}

		t.Setenv("AGENTFIELD_LOGS_ENABLED", "off")
		if processLogsEnabled() {
			t.Fatal("expected process logs disabled")
		}

		t.Setenv("AGENTFIELD_LOGS_ENABLED", "true")
		if !processLogsEnabled() {
			t.Fatal("expected process logs enabled")
		}

		t.Setenv("AGENTFIELD_AUTHORIZATION_INTERNAL_TOKEN", "secret")
		if internalBearerOK("Bearer wrong") {
			t.Fatal("expected bearer mismatch to fail")
		}
		if !internalBearerOK("Bearer secret") {
			t.Fatal("expected bearer token to pass")
		}
	})

	t.Run("ring trimming and snapshots", func(t *testing.T) {
		if ring := newProcessLogRing(1); ring.maxBytes != 1024 {
			t.Fatalf("newProcessLogRing min bytes = %d", ring.maxBytes)
		}

		var nilRing *processLogRing
		nilRing.appendLine("stdout", "ignored", false)

		ring := newProcessLogRing(1024)
		longLine := strings.Repeat("x", 400)
		ring.appendLine("stdout", longLine+"-first", false)
		ring.appendLine("stderr", longLine+"-second", true)
		ring.appendLine("custom", longLine+"-third", false)

		if got := ring.tail(0); got != nil {
			t.Fatalf("tail(0) = %#v", got)
		}

		entries := ring.tail(10)
		if len(entries) != 2 {
			t.Fatalf("tail entries = %d", len(entries))
		}
		if entries[0].Level != "error" || !entries[0].Truncated {
			t.Fatalf("unexpected stderr entry: %#v", entries[0])
		}
		if entries[1].Level != "log" {
			t.Fatalf("unexpected custom level: %#v", entries[1])
		}

		since := ring.snapshotAfter(1, 0)
		if len(since) != 2 {
			t.Fatalf("snapshotAfter len = %d", len(since))
		}
		limited := ring.snapshotAfter(1, 1)
		if len(limited) != 1 || !strings.HasSuffix(limited[0].Line, "-third") {
			t.Fatalf("snapshotAfter limit = %#v", limited)
		}
	})
}

func TestProcessLogs_Handler(t *testing.T) {
	t.Run("method not allowed", func(t *testing.T) {
		t.Setenv("AGENTFIELD_LOGS_ENABLED", "true")
		req := httptest.NewRequest(http.MethodPost, "/agentfield/v1/logs", nil)
		resp := httptest.NewRecorder()

		testAgentWithProcessLogRing(newProcessLogRing(1024)).handleAgentfieldLogs(resp, req)

		if resp.Code != http.StatusMethodNotAllowed {
			t.Fatalf("status = %d", resp.Code)
		}
	})

	t.Run("disabled and unauthorized responses", func(t *testing.T) {
		t.Setenv("AGENTFIELD_LOGS_ENABLED", "false")
		req := httptest.NewRequest(http.MethodGet, "/agentfield/v1/logs", nil)
		resp := httptest.NewRecorder()

		testAgentWithProcessLogRing(newProcessLogRing(1024)).handleAgentfieldLogs(resp, req)

		if resp.Code != http.StatusNotFound || !strings.Contains(resp.Body.String(), "logs_disabled") {
			t.Fatalf("unexpected disabled response: %d %s", resp.Code, resp.Body.String())
		}

		t.Setenv("AGENTFIELD_LOGS_ENABLED", "true")
		t.Setenv("AGENTFIELD_AUTHORIZATION_INTERNAL_TOKEN", "secret")
		unauthorized := httptest.NewRecorder()
		testAgentWithProcessLogRing(newProcessLogRing(1024)).handleAgentfieldLogs(unauthorized, req)
		if unauthorized.Code != http.StatusUnauthorized || !strings.Contains(unauthorized.Body.String(), "unauthorized") {
			t.Fatalf("unexpected unauthorized response: %d %s", unauthorized.Code, unauthorized.Body.String())
		}
	})

	t.Run("tailing and follow mode", func(t *testing.T) {
		t.Setenv("AGENTFIELD_LOGS_ENABLED", "true")
		t.Setenv("AGENTFIELD_AUTHORIZATION_INTERNAL_TOKEN", "secret")
		t.Setenv("AGENTFIELD_LOG_MAX_TAIL_LINES", "1")

		ring := newProcessLogRing(1024)
		ring.appendLine("stdout", "first line", false)
		ring.appendLine("stderr", "second line", true)
		a := testAgentWithProcessLogRing(ring)

		tooLargeReq := httptest.NewRequest(http.MethodGet, "/agentfield/v1/logs?tail_lines=2", nil)
		tooLargeReq.Header.Set("Authorization", "Bearer secret")
		tooLargeResp := httptest.NewRecorder()
		a.handleAgentfieldLogs(tooLargeResp, tooLargeReq)
		if tooLargeResp.Code != http.StatusRequestEntityTooLarge || !strings.Contains(tooLargeResp.Body.String(), "tail_too_large") {
			t.Fatalf("unexpected tail too large response: %d %s", tooLargeResp.Code, tooLargeResp.Body.String())
		}

		defaultReq := httptest.NewRequest(http.MethodGet, "/agentfield/v1/logs", nil)
		defaultReq.Header.Set("Authorization", "Bearer secret")
		defaultResp := httptest.NewRecorder()
		a.handleAgentfieldLogs(defaultResp, defaultReq)
		if defaultResp.Code != http.StatusOK {
			t.Fatalf("status = %d", defaultResp.Code)
		}
		if got := defaultResp.Header().Get("Cache-Control"); got != "no-store" {
			t.Fatalf("cache-control = %q", got)
		}
		entries := decodeProcessLogEntries(t, defaultResp.Body.String())
		if len(entries) != 2 || entries[0].Line != "first line" || entries[1].Line != "second line" {
			t.Fatalf("default entries = %#v", entries)
		}

		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		followReq := httptest.NewRequest(http.MethodGet, "/agentfield/v1/logs?since_seq=1&tail_lines=1&follow=true", nil).WithContext(ctx)
		followReq.Header.Set("Authorization", "Bearer secret")
		followResp := httptest.NewRecorder()
		a.handleAgentfieldLogs(followResp, followReq)
		followEntries := decodeProcessLogEntries(t, followResp.Body.String())
		if len(followEntries) != 1 || followEntries[0].Line != "second line" {
			t.Fatalf("follow entries = %#v", followEntries)
		}
	})
	

t.Run("evicts oldest entries when exceeding maxBytes", func(t *testing.T) {
	ring := newProcessLogRing(100) // VERY small buffer

	big := strings.Repeat("x", 500) // VERY large lines

	ring.appendLine("stdout", big+"1", false)
	ring.appendLine("stdout", big+"2", false)
	ring.appendLine("stdout", big+"3", false)

	entries := ring.tail(10)

	// After eviction, only latest entry should remain
	if len(entries) != 1 {
		t.Fatalf("expected 1 entry after eviction, got %d", len(entries))
	}

	if !strings.HasSuffix(entries[0].Line, "3") {
		t.Fatalf("expected latest entry to remain, got %#v", entries)
	}
})
    t.Run("snapshotAfter returns only newer entries", func(t *testing.T) {
	ring := newProcessLogRing(1024)

	ring.appendLine("stdout", "first", false)
	ring.appendLine("stdout", "second", false)
	ring.appendLine("stdout", "third", false)

	entries := ring.snapshotAfter(2, 0)

	if len(entries) != 1 || entries[0].Line != "third" {
		t.Fatalf("unexpected snapshotAfter result: %#v", entries)
	}
   })
     t.Run("returns NDJSON formatted logs", func(t *testing.T) {
	t.Setenv("AGENTFIELD_LOGS_ENABLED", "true")

	ring := newProcessLogRing(1024)
	ring.appendLine("stdout", "hello", false)

	req := httptest.NewRequest(http.MethodGet, "/agentfield/v1/logs", nil)
	req.Header.Set("Authorization", "Bearer")

	resp := httptest.NewRecorder()

	testAgentWithProcessLogRing(ring).handleAgentfieldLogs(resp, req)

	body := resp.Body.String()

	if !strings.Contains(body, "\n") {
		t.Fatalf("expected NDJSON format")
	}
   })
	  
}
