package agent

import (
	"crypto/subtle"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

// processLogEntry matches NDJSON v1 (docs/api/AGENT_NODE_LOGS.md).
type processLogEntry struct {
	V         int    `json:"v"`
	Seq       int    `json:"seq"`
	TS        string `json:"ts"`
	Stream    string `json:"stream"`
	Line      string `json:"line"`
	Level     string `json:"level,omitempty"`
	Source    string `json:"source,omitempty"`
	Truncated bool   `json:"truncated,omitempty"`
}

type processLogRing struct {
	mu          sync.Mutex
	seq         int
	entries     []processLogEntry
	approxBytes int
	maxBytes    int
}

func newProcessLogRing(maxBytes int) *processLogRing {
	if maxBytes < 1024 {
		maxBytes = 1024
	}
	return &processLogRing{maxBytes: maxBytes}
}

func processLogsMaxBytes() int {
	raw := strings.TrimSpace(os.Getenv("AGENTFIELD_LOG_BUFFER_BYTES"))
	if raw == "" {
		return 4 << 20
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n < 1024 {
		return 4 << 20
	}
	return n
}

func processLogsMaxLineBytes() int {
	raw := strings.TrimSpace(os.Getenv("AGENTFIELD_LOG_MAX_LINE_BYTES"))
	if raw == "" {
		return 16384
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n < 256 {
		return 16384
	}
	return n
}

func processLogsMaxTailLines() int {
	raw := strings.TrimSpace(os.Getenv("AGENTFIELD_LOG_MAX_TAIL_LINES"))
	if raw == "" {
		return 50000
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n < 1 {
		return 50000
	}
	return n
}

func processLogsEnabled() bool {
	v := strings.ToLower(strings.TrimSpace(os.Getenv("AGENTFIELD_LOGS_ENABLED")))
	return v != "0" && v != "false" && v != "no" && v != "off"
}

func internalBearerOK(authHeader string) bool {
	want := strings.TrimSpace(os.Getenv("AGENTFIELD_AUTHORIZATION_INTERNAL_TOKEN"))
	if want == "" {
		return true
	}
	if !strings.HasPrefix(strings.ToLower(authHeader), "bearer ") {
		return false
	}
	got := strings.TrimSpace(authHeader[7:])
	return subtle.ConstantTimeCompare([]byte(got), []byte(want)) == 1
}

func (r *processLogRing) appendLine(stream, line string, truncated bool) {
	if r == nil {
		return
	}
	ts := time.Now().UTC().Format("2006-01-02T15:04:05.000Z07:00")
	level := "info"
	switch strings.ToLower(strings.TrimSpace(stream)) {
	case "stderr":
		level = "error"
	case "stdout":
		level = "info"
	default:
		level = "log"
	}
	r.mu.Lock()
	r.seq++
	e := processLogEntry{
		V: 1, Seq: r.seq, TS: ts, Stream: stream, Line: line,
		Level: level, Source: "process", Truncated: truncated,
	}
	r.entries = append(r.entries, e)
	r.approxBytes += len(line) + 64
	for r.approxBytes > r.maxBytes && len(r.entries) > 1 {
		old := r.entries[0]
		r.entries = r.entries[1:]
		r.approxBytes -= len(old.Line) + 64
	}
	r.mu.Unlock()
}

func (r *processLogRing) tail(n int) []processLogEntry {
	r.mu.Lock()
	defer r.mu.Unlock()
	if n <= 0 {
		return nil
	}
	if len(r.entries) <= n {
		out := make([]processLogEntry, len(r.entries))
		copy(out, r.entries)
		return out
	}
	out := make([]processLogEntry, n)
	copy(out, r.entries[len(r.entries)-n:])
	return out
}

func (r *processLogRing) snapshotAfter(sinceSeq int, limit int) []processLogEntry {
	r.mu.Lock()
	defer r.mu.Unlock()
	var buf []processLogEntry
	for _, e := range r.entries {
		if e.Seq > sinceSeq {
			buf = append(buf, e)
		}
	}
	if limit > 0 && len(buf) > limit {
		buf = buf[len(buf)-limit:]
	}
	return buf
}

func (a *Agent) ensureProcessLogRing() {
	if a == nil {
		return
	}
	a.procLogOnce.Do(func() {
		if !processLogsEnabled() {
			return
		}
		a.procLogRing = newProcessLogRing(processLogsMaxBytes())
		installProcessStdioCapture(a.procLogRing)
	})
}

func (a *Agent) handleAgentfieldLogs(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if !processLogsEnabled() {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"error":"logs_disabled","message":"Process logs API is disabled"}`))
		return
	}
	a.ensureProcessLogRing()
	if a.procLogRing == nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"error":"logs_disabled","message":"Process logs API is disabled"}`))
		return
	}
	if !internalBearerOK(r.Header.Get("Authorization")) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":"unauthorized","message":"Valid Authorization Bearer required"}`))
		return
	}
	q := r.URL.Query()
	tailLines, _ := strconv.Atoi(q.Get("tail_lines"))
	sinceSeq, _ := strconv.Atoi(q.Get("since_seq"))
	follow := strings.EqualFold(q.Get("follow"), "1") || strings.EqualFold(q.Get("follow"), "true") || q.Get("follow") == "yes"
	maxTail := processLogsMaxTailLines()
	if tailLines > maxTail {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusRequestEntityTooLarge)
		_, _ = fmt.Fprintf(w, `{"error":"tail_too_large","message":"tail_lines exceeds max %d"}`, maxTail)
		return
	}
	if tailLines <= 0 && sinceSeq <= 0 && !follow {
		tailLines = 200
	}

	var initial []processLogEntry
	if sinceSeq > 0 {
		initial = a.procLogRing.snapshotAfter(sinceSeq, tailLines)
	} else {
		n := tailLines
		if n <= 0 {
			n = 200
		}
		initial = a.procLogRing.tail(n)
	}

	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(http.StatusOK)
	fl, _ := w.(http.Flusher)

	enc := json.NewEncoder(w)
	for _, e := range initial {
		if err := enc.Encode(e); err != nil {
			return
		}
		if fl != nil {
			fl.Flush()
		}
	}
	if !follow {
		return
	}

	lastSeq := sinceSeq
	if len(initial) > 0 {
		lastSeq = initial[len(initial)-1].Seq
	}

	tick := time.NewTicker(400 * time.Millisecond)
	defer tick.Stop()
	ctx := r.Context()

	for {
		select {
		case <-ctx.Done():
			return
		case <-tick.C:
			newer := a.procLogRing.snapshotAfter(lastSeq, 0)
			for _, e := range newer {
				if err := enc.Encode(e); err != nil {
					return
				}
				lastSeq = e.Seq
			}
			if fl != nil {
				fl.Flush()
			}
		}
	}
}
