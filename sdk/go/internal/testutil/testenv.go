//go:build functional

package testutil

import (
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

// StartTestInfra starts the Docker Compose test infrastructure.
// It locates the test-infra directory, runs the start-env.sh script,
// and registers cleanup with t.Cleanup().
func StartTestInfra(t *testing.T) string {
	t.Helper()

	// Find test-infra directory (relative to this file)
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("Failed to get current file path")
	}

	// Go up from sdk/go/internal/testutil to root, then to test-infra
	sdkGoDir := filepath.Dir(filepath.Dir(filepath.Dir(filename)))
	repoRoot := filepath.Dir(filepath.Dir(sdkGoDir))
	testInfraDir := filepath.Join(repoRoot, "test-infra")
	startScript := filepath.Join(testInfraDir, "scripts", "start-env.sh")

	// Check if script exists
	if _, err := os.Stat(startScript); os.IsNotExist(err) {
		t.Fatalf("Test infrastructure script not found at %s", startScript)
	}

	t.Logf("Starting test infrastructure from %s", testInfraDir)

	// Run start script
	cmd := exec.Command(startScript)
	cmd.Dir = testInfraDir
	cmd.Env = os.Environ()

	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("Failed to start test infrastructure: %v\nOutput:\n%s", err, string(output))
	}

	t.Logf("Test infrastructure started:\n%s", string(output))

	// Get control plane URL
	controlPlaneURL := os.Getenv("AGENTFIELD_SERVER")
	if controlPlaneURL == "" {
		controlPlaneURL = "http://localhost:8080"
	}

	// Wait for control plane to be ready
	if err := WaitForHealth(controlPlaneURL, 60*time.Second); err != nil {
		t.Fatalf("Control plane failed to become healthy: %v", err)
	}

	// Register cleanup
	t.Cleanup(func() {
		CleanupTestInfra(t, testInfraDir)
	})

	return controlPlaneURL
}

// CleanupTestInfra stops the Docker Compose test infrastructure.
func CleanupTestInfra(t *testing.T, testInfraDir string) {
	t.Helper()

	stopScript := filepath.Join(testInfraDir, "scripts", "stop-env.sh")

	t.Log("Stopping test infrastructure...")

	cmd := exec.Command(stopScript)
	cmd.Dir = testInfraDir

	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Logf("Warning: Failed to stop test infrastructure: %v\nOutput:\n%s", err, string(output))
	} else {
		t.Logf("Test infrastructure stopped:\n%s", string(output))
	}
}

// WaitForHealth waits for the control plane health endpoint to respond.
func WaitForHealth(baseURL string, timeout time.Duration) error {
	healthURL := baseURL + "/api/v1/health"
	client := &http.Client{Timeout: 5 * time.Second}

	deadline := time.Now().Add(timeout)

	for time.Now().Before(deadline) {
		resp, err := client.Get(healthURL)
		if err == nil && resp.StatusCode == http.StatusOK {
			resp.Body.Close()
			return nil
		}
		if resp != nil {
			resp.Body.Close()
		}

		time.Sleep(2 * time.Second)
	}

	return fmt.Errorf("control plane at %s did not become healthy within %v", baseURL, timeout)
}

// GetOpenRouterConfig returns OpenRouter configuration from environment variables.
// It skips the test if OPENROUTER_API_KEY is not set.
func GetOpenRouterConfig(t *testing.T) (apiKey, baseURL, model string) {
	t.Helper()

	apiKey = os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		t.Skip("OPENROUTER_API_KEY not set. Set it in test-infra/.env.test or as an environment variable.")
	}

	baseURL = os.Getenv("OPENROUTER_BASE_URL")
	if baseURL == "" {
		baseURL = "https://openrouter.ai/api/v1"
	}

	model = os.Getenv("OPENROUTER_MODEL")
	if model == "" {
		model = "google/gemini-flash-1.5"
	}

	return apiKey, baseURL, model
}

// AllocatePort returns a unique port for testing.
// Starting from 9000, it increments for each call.
var portCounter = 9000

func AllocatePort() int {
	port := portCounter
	portCounter++
	return port
}
