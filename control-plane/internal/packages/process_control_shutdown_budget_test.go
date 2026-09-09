package packages

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"
)

func TestShutdownBudgetDefaultsForUnavailableProcessEnvironment(t *testing.T) {
	missingPID := 1 << 30
	entry := InstalledPackage{Runtime: RuntimeInfo{PID: &missingPID}}

	if environment := processEnvironment(missingPID); environment != nil {
		t.Fatalf("environment=%q, want nil for missing /proc entry", environment)
	}
	if got := agentShutdownBudget(entry); got != defaultAgentShutdownBudget {
		t.Fatalf("budget=%s, want %s", got, defaultAgentShutdownBudget)
	}
}

func TestShutdownBudgetReadsProcessEnvironmentAndParsesSeconds(t *testing.T) {
	pid := os.Getpid()
	entry := InstalledPackage{Runtime: RuntimeInfo{PID: &pid}}

	if environment := processEnvironment(pid); environment == nil {
		t.Fatal("environment=nil for current process")
	}
	if got := agentShutdownBudgetWith(entry, func(int) []string {
		return []string{"AGENTFIELD_SHUTDOWN_TIMEOUT=7"}
	}); got != 7*time.Second {
		t.Fatalf("budget=%s, want 7s", got)
	}
}

func TestShutdownBudgetErrorAndFallbackBranches(t *testing.T) {
	positivePID := 123
	tests := []struct {
		name  string
		entry InstalledPackage
		env   []string
	}{
		{name: "missing pid"},
		{name: "non-positive pid", entry: InstalledPackage{Runtime: RuntimeInfo{PID: intPointer(0)}}},
		{name: "missing variable", entry: InstalledPackage{Runtime: RuntimeInfo{PID: &positivePID}}, env: []string{"OTHER=value"}},
		{name: "malformed value", entry: InstalledPackage{Runtime: RuntimeInfo{PID: &positivePID}}, env: []string{"AGENTFIELD_SHUTDOWN_TIMEOUT=eventually"}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := agentShutdownBudgetWith(test.entry, func(int) []string { return test.env })
			if got != defaultAgentShutdownBudget {
				t.Fatalf("budget=%s, want %s", got, defaultAgentShutdownBudget)
			}
		})
	}
}

func TestRequestHTTPShutdownUsesShutdownBudgetFallbackValue(t *testing.T) {
	var body struct {
		Graceful       bool    `json:"graceful"`
		TimeoutSeconds float64 `json:"timeout_seconds"`
	}
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Errorf("decode shutdown request: %v", err)
		}
		response.WriteHeader(http.StatusAccepted)
	}))
	defer server.Close()

	pid := 123
	entry := InstalledPackage{Runtime: RuntimeInfo{PID: &pid}}
	budget := agentShutdownBudgetWith(entry, func(int) []string {
		return []string{"AGENTFIELD_SHUTDOWN_TIMEOUT=malformed"}
	})
	port := server.Listener.Addr().(*net.TCPAddr).Port
	accepted, timedOut := requestHTTPShutdown(context.Background(), port, budget)

	if !accepted || timedOut {
		t.Fatalf("accepted=%v timedOut=%v", accepted, timedOut)
	}
	if !body.Graceful || body.TimeoutSeconds != defaultAgentShutdownBudget.Seconds() {
		t.Fatalf("body=%+v, want graceful with timeout_seconds=%v", body, defaultAgentShutdownBudget.Seconds())
	}
}

func intPointer(value int) *int {
	return &value
}

func TestStopFallbackLadderForGracefulFailure(t *testing.T) {
	forceCalled := false
	interruptSent, forceNeeded, err := stopProcessWith(
		42,
		func(int) error { return context.DeadlineExceeded },
		func(int) bool { return true },
		func(int) error { forceCalled = true; return nil },
	)

	if err != nil {
		t.Fatalf("stop fallback: %v", err)
	}
	if interruptSent || !forceNeeded || !forceCalled {
		t.Fatalf("interruptSent=%v forceNeeded=%v forceCalled=%v", interruptSent, forceNeeded, forceCalled)
	}
}
