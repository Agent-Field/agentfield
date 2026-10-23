package commands

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/cli/framework"
	"github.com/Agent-Field/agentfield/control-plane/internal/codeaf"
	"github.com/Agent-Field/agentfield/control-plane/internal/core/domain"
)

// Finding 3: optional provisioning must not delay launching an installed node.
func TestRunDoesNotWaitForCodeafDownload(t *testing.T) {
	t.Setenv("AGENTFIELD_SKIP_CODEAF", "0")
	t.Setenv("AGENTFIELD_SKIP_AFORGE", "0")
	t.Setenv("AGENTFIELD_HOME", t.TempDir())
	// Keep the synchronous hook stalled, and verify run delegates only to the
	// detached hook. The real process lifetime is tested in codeaf's subprocess test.
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { <-release; w.WriteHeader(503) }))
	defer server.Close()
	releaseDownload := sync.OnceFunc(func() { close(release) })
	defer releaseDownload()
	t.Setenv("AGENTFIELD_CODEAF_BASE_URL", server.URL)
	old, oldBackground := ensureCodeaf, ensureCodeafBackground
	t.Cleanup(func() { ensureCodeaf = old; ensureCodeafBackground = oldBackground })
	ensureCodeaf = func(opts codeaf.Options, w io.Writer) error { return codeaf.EnsureBestEffort(opts, w) }
	backgroundCalls := 0
	ensureCodeafBackground = func() error { backgroundCalls++; return nil }
	run := &fakeAgentService{runningAgent: &domain.RunningAgent{PID: 1, Port: 1234}}
	cmd := NewRunCommand(&framework.ServiceContainer{AgentService: run}).BuildCobraCommand()
	cmd.SetArgs([]string{"fixture"})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	cmd.SetContext(ctx)
	done := make(chan error, 1)
	go func() { done <- cmd.Execute() }()
	select {
	case err := <-done:
		if err != nil || run.runCalls != 1 || backgroundCalls != 1 {
			t.Fatalf("run calls=%d err=%v", run.runCalls, err)
		}
	case <-time.After(200 * time.Millisecond):
		releaseDownload()
		<-done
		t.Fatal("agent launch waited for optional CodeAF download")
	}
}
