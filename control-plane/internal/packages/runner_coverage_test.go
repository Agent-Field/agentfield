
package packages

import (
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestRunnerErrorCases(t *testing.T) {
	// t.Parallel() // This causes issues with port allocation in parallel tests

	t.Run("RunAgentNode-load-registry-fails", func(t *testing.T) {
		home := t.TempDir()
		ar := &AgentNodeRunner{AgentFieldHome: home}
		regPath := filepath.Join(home, "installed.yaml")
		if err := os.WriteFile(regPath, []byte("bad:yaml"), 0644); err != nil {
			t.Fatal(err)
		}
		err := ar.RunAgentNode("any")
		if err == nil || !strings.Contains(err.Error(), "failed to load registry") {
			t.Fatalf("expected load registry error, got %v", err)
		}
	})

	t.Run("RunAgentNode-getFreePort-fails", func(t *testing.T) {
		// Rather than trying to exhaust 999 ports (flaky and slow), we test
		// getFreePort directly by binding every port in the range on loopback.
		// If we can't bind them all (e.g. CI contention), we skip — this is
		// fundamentally environment-dependent.
		listeners := make([]net.Listener, 0, 999)
		for port := 8001; port <= 8999; port++ {
			l, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port))
			if err != nil {
				// Can't reliably block all ports — clean up and skip.
				for _, ll := range listeners {
					ll.Close()
				}
				t.Skip("cannot bind all ports in 8001-8999 range; skipping port-exhaustion test")
			}
			listeners = append(listeners, l)
		}
		defer func() {
			for _, l := range listeners {
				l.Close()
			}
		}()

		ar := &AgentNodeRunner{}
		_, err := ar.getFreePort()
		if err == nil {
			t.Fatal("expected getFreePort to fail when all ports are occupied")
		}
		if !strings.Contains(err.Error(), "no free port") {
			t.Fatalf("expected 'no free port' error, got: %v", err)
		}
	})

	t.Run("RunAgentNode-start-process-fails", func(t *testing.T) {
		home := t.TempDir()
		ar := &AgentNodeRunner{AgentFieldHome: home}
		
		pkgPath := filepath.Join(home, "packages", "demo")
		if err := os.MkdirAll(pkgPath, 0755); err != nil {t.Fatal(err)}
		registry := &InstallationRegistry{Installed: map[string]InstalledPackage{
			"demo": { Name: "demo", Path: pkgPath },
		}}
		if err := (&PackageUninstaller{AgentFieldHome: home}).saveRegistry(registry); err != nil {
			t.Fatal(err)
		}

		// Make python not found
		origPath := os.Getenv("PATH")
		os.Setenv("PATH", "")
		defer os.Setenv("PATH", origPath)

		err := ar.RunAgentNode("demo")
		if err == nil || !strings.Contains(err.Error(), "failed to start agent node") {
			t.Fatalf("expected start process error, got %v", err)
		}
	})
	
	t.Run("displayCapabilities-fails", func(t *testing.T) {
		ar := &AgentNodeRunner{}
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request){
			http.Error(w, "error", http.StatusInternalServerError)
		}))
		defer server.Close()

		port := server.Listener.Addr().(*net.TCPAddr).Port
		
		err := ar.displayCapabilities(InstalledPackage{}, port)
		if err == nil {
			t.Fatalf("expected display capabilities to fail")
		}
	})

	t.Run("updateRuntimeInfo-read-only-registry", func(t *testing.T){
		home := t.TempDir()
		ar := &AgentNodeRunner{AgentFieldHome: home}
		regPath := filepath.Join(home, "installed.yaml")
		if err := os.WriteFile(regPath, []byte("installed: { demo: { name: demo } }"), 0444); err != nil {
			t.Fatal(err)
		}
		if err := os.Chmod(regPath, 0444); err != nil {
			t.Fatal(err)
		}

		err := ar.updateRuntimeInfo("demo", 1234, 5678)
		if err == nil || !(strings.Contains(err.Error(), "permission denied") || strings.Contains(err.Error(), "read-only file system")){
			t.Fatalf("expected permission error, got %v", err)
		}
		_ = os.Chmod(regPath, 0644)
	})
	
	t.Run("loadPackageEnvFile-unquoted", func(t *testing.T){
		ar := &AgentNodeRunner{}
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, ".env"), []byte("FOO=bar"), 0644); err != nil {
			t.Fatal(err)
		}
		vars, err := ar.loadPackageEnvFile(dir)
		if err != nil {
			t.Fatal(err)
		}
		if vars["FOO"] != "bar" {
			t.Fatalf("expected bar, got %s", vars["FOO"])
		}
	})

	t.Run("waitForAgentNode-timeout", func(t *testing.T) {
		ar := &AgentNodeRunner{}
		// just use a port that is not listening
		err := ar.waitForAgentNode(1, 10 * time.Millisecond)
		if err == nil {
			t.Fatal("expected timeout error")
		}
	})
}
