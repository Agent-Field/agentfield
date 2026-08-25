package updatecheck

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"
)

type runnerResult struct {
	output string
	err    error
}

type recordingRunner struct {
	results []runnerResult
	calls   [][]string
}

func (r *recordingRunner) Run(_ context.Context, args ...string) ([]byte, error) {
	r.calls = append(r.calls, append([]string(nil), args...))
	result := r.results[len(r.calls)-1]
	return []byte(result.output), result.err
}

func TestCheckerClassifiesGitPackagesFromLsRemote(t *testing.T) {
	const installed = "1111111111111111111111111111111111111111"
	const latest = "2222222222222222222222222222222222222222"
	now := time.Date(2026, 8, 24, 19, 40, 0, 0, time.UTC)
	runner := &recordingRunner{results: []runnerResult{
		{output: latest + "\tHEAD\n"},
		{output: installed + "\tHEAD\n"},
		{output: latest + "\trefs/tags/v1.2.3\n"},
		{err: errors.New("remote unavailable")},
	}}
	checker := NewChecker(runner)
	checker.now = func() time.Time { return now }

	results := checker.Check(context.Background(), []Entry{
		{ID: "available", Name: "Available", Source: "https://github.com/acme/available", InstalledCommit: installed},
		{ID: "current", Name: "Current", Source: "https://github.com/acme/current", InstalledCommit: installed},
		{ID: "pinned", Name: "Pinned", Source: "https://github.com/acme/pinned@v1.2.3", Ref: "v1.2.3", InstalledCommit: installed},
		{ID: "error", Name: "Error", Source: "https://github.com/acme/error", InstalledCommit: installed},
	})

	wantStatuses := []Status{StatusAvailable, StatusCurrent, StatusPinned, StatusError}
	for i, result := range results {
		if result.Update.Status != wantStatuses[i] {
			t.Fatalf("result %s status = %q, want %q", result.ID, result.Update.Status, wantStatuses[i])
		}
		if !result.Update.CheckedAt.Equal(now) {
			t.Fatalf("result %s checked_at = %s, want %s", result.ID, result.Update.CheckedAt, now)
		}
	}
	if results[0].Update.LatestCommit != latest || results[1].Update.LatestCommit != installed {
		t.Fatalf("latest commits were not parsed: %+v", results)
	}
	if results[2].Update.LatestCommit != latest {
		t.Fatalf("pinned package must still report latest commit: %+v", results[2])
	}
	if !strings.Contains(results[3].Update.Message, "remote unavailable") {
		t.Fatalf("runner error message missing: %+v", results[3])
	}

	wantCalls := [][]string{
		{"ls-remote", "--quiet", "https://github.com/acme/available", "HEAD"},
		{"ls-remote", "--quiet", "https://github.com/acme/current", "HEAD"},
		{"ls-remote", "--quiet", "https://github.com/acme/pinned", "v1.2.3"},
		{"ls-remote", "--quiet", "https://github.com/acme/error", "HEAD"},
	}
	if !reflect.DeepEqual(runner.calls, wantCalls) {
		t.Fatalf("git calls = %#v, want %#v", runner.calls, wantCalls)
	}
	for _, call := range runner.calls {
		if len(call) > 0 && call[0] == "clone" {
			t.Fatalf("update checks must never clone: %v", call)
		}
	}
}

func TestCheckerTreatsUnknownInstalledCommitAsAvailableOnce(t *testing.T) {
	runner := &recordingRunner{results: []runnerResult{{output: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\tHEAD\n"}}}
	checker := NewChecker(runner)
	result := checker.Check(context.Background(), []Entry{{ID: "legacy", Name: "Legacy", Source: "https://github.com/acme/legacy"}})[0]
	if result.Update.Status != StatusAvailable {
		t.Fatalf("status = %q, want available so a legacy install updates once", result.Update.Status)
	}
	if !strings.Contains(result.Update.Message, "unknown") {
		t.Fatalf("message = %q, want unknown-commit explanation", result.Update.Message)
	}
}

func TestCheckerCachesAndClearsResults(t *testing.T) {
	runner := &recordingRunner{results: []runnerResult{{output: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\tHEAD\n"}}}
	checker := NewChecker(runner)
	checker.Check(context.Background(), []Entry{{ID: "demo", Name: "Demo", Source: "https://github.com/acme/demo"}})
	if got := checker.Cached("demo"); got.Status != StatusAvailable {
		t.Fatalf("cached status = %q", got.Status)
	}
	checker.Clear("demo")
	if got := checker.Cached("demo"); got.Status != StatusUnknown {
		t.Fatalf("cleared status = %q, want unknown", got.Status)
	}
}

func TestE19FailedCommitMemoSurvivesTransientCheckError(t *testing.T) {
	const failedCommit = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	runner := &recordingRunner{results: []runnerResult{
		{err: errors.New("remote temporarily unavailable")},
		{output: failedCommit + "\tHEAD\n"},
	}}
	checker := NewChecker(runner)
	checker.Set("demo", Update{
		Status:       StatusFailed,
		LatestCommit: failedCommit,
		Message:      "update failed for this commit",
	})
	entry := []Entry{{ID: "demo", Source: "https://github.com/acme/demo", InstalledCommit: "old"}}

	first := checker.Check(context.Background(), entry)[0]
	if first.Update.Status != StatusError {
		t.Fatalf("transient result = %+v, want error", first.Update)
	}
	if cached := checker.Cached("demo"); cached.Status != StatusFailed || cached.LatestCommit != failedCommit {
		t.Fatalf("transient error erased failed-commit memo: %+v", cached)
	}

	second := checker.Check(context.Background(), entry)[0]
	if second.Update.Status != StatusFailed || second.Update.LatestCommit != failedCommit {
		t.Fatalf("same remote commit was made eligible again: %+v", second.Update)
	}
	for _, call := range runner.calls {
		if len(call) > 0 && call[0] == "clone" {
			t.Fatalf("memo recheck must remain ls-remote-only: %v", runner.calls)
		}
	}
}

type contextErrorRunner struct{}

func (contextErrorRunner) Run(ctx context.Context, _ ...string) ([]byte, error) {
	return nil, ctx.Err()
}

func TestCanceledCheckDoesNotOverwriteCachedResult(t *testing.T) {
	checker := NewChecker(contextErrorRunner{})
	previous := Update{Status: StatusCurrent, LatestCommit: "known", CheckedAt: time.Now()}
	checker.Set("demo", previous)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	result := checker.Check(ctx, []Entry{{ID: "demo", Source: "https://github.com/acme/demo", InstalledCommit: "known"}})[0]
	if result.Update.Status != StatusError {
		t.Fatalf("request result = %+v", result.Update)
	}
	if cached := checker.Cached("demo"); cached.Status != StatusCurrent || cached.LatestCommit != "known" {
		t.Fatalf("canceled check overwrote cache: %+v", cached)
	}
}

func TestCheckerReportsInvalidAndEmptyRemoteResults(t *testing.T) {
	runner := &recordingRunner{results: []runnerResult{{output: "\n"}}}
	checker := NewChecker(runner)
	results := checker.Check(context.Background(), []Entry{
		{ID: "invalid", Source: ""},
		{ID: "empty", Source: "https://github.com/acme/empty"},
	})
	if results[0].Update.Status != StatusError || !strings.Contains(results[0].Update.Message, "invalid") {
		t.Fatalf("invalid result = %+v", results[0])
	}
	if results[1].Update.Status != StatusError || !strings.Contains(results[1].Update.Message, "no commit") {
		t.Fatalf("empty result = %+v", results[1])
	}
}

func TestCheckUpdatesBudgetContextDeadline(t *testing.T) {
	for _, test := range []struct {
		name   string
		count  int
		budget time.Duration
	}{
		{name: "ten second floor", count: 1, budget: 10 * time.Second},
		{name: "one hundred twenty second cap", count: 13, budget: 120 * time.Second},
	} {
		t.Run(test.name, func(t *testing.T) {
			results := make([]runnerResult, test.count)
			entries := make([]Entry, test.count)
			for i := range entries {
				results[i] = runnerResult{output: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\tHEAD\n"}
				entries[i] = Entry{ID: fmt.Sprintf("demo-%d", i), Source: "https://github.com/acme/demo", InstalledCommit: "old"}
			}
			checker := NewChecker(&recordingRunner{results: results})
			var remaining time.Duration
			checker.observeCheckContext = func(ctx context.Context) {
				deadline, ok := ctx.Deadline()
				if !ok {
					t.Fatal("aggregate check context has no deadline")
				}
				remaining = time.Until(deadline)
			}
			checker.CheckWithTimeout(context.Background(), entries, test.budget)
			if remaining < test.budget-time.Second || remaining > test.budget {
				t.Fatalf("remaining budget = %s, want approximately %s", remaining, test.budget)
			}
		})
	}
}

type serializedBudgetRunner struct {
	started chan struct{}
	release chan struct{}
	calls   int
}

func (r *serializedBudgetRunner) Run(ctx context.Context, _ ...string) ([]byte, error) {
	r.calls++
	if r.calls == 1 {
		close(r.started)
		<-r.release
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return []byte("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\tHEAD\n"), nil
}

func TestCheckWithTimeoutStartsBudgetAfterSerializedLock(t *testing.T) {
	runner := &serializedBudgetRunner{started: make(chan struct{}), release: make(chan struct{})}
	checker := NewChecker(runner)
	entry := []Entry{{ID: "demo", Source: "https://github.com/acme/demo", InstalledCommit: "old"}}
	firstDone := make(chan struct{})
	go func() {
		checker.Check(context.Background(), entry)
		close(firstDone)
	}()
	<-runner.started

	result := make(chan []Result, 1)
	go func() { result <- checker.CheckWithTimeout(context.Background(), entry, 30*time.Millisecond) }()
	time.Sleep(40 * time.Millisecond)
	close(runner.release)
	<-firstDone
	checked := <-result
	if len(checked) != 1 || checked[0].Update.Status != StatusAvailable {
		t.Fatalf("queued check lost its budget before acquiring the lock: %+v", checked)
	}
}
