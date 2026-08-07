//go:build unix

package furrow

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestEnsureFailsToLockInstallation(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("root can write through directory permissions")
	}
	home := t.TempDir()
	binDir := filepath.Join(home, "bin")
	if err := os.Mkdir(binDir, 0o500); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.Chmod(binDir, 0o700); err != nil {
			t.Errorf("restore bin directory permissions: %v", err)
		}
	})

	err := Ensure(Options{GOOS: "linux", GOARCH: "amd64", Home: home})
	if err == nil || !strings.Contains(err.Error(), "lock furrow installation") {
		t.Fatalf("error = %v", err)
	}
}
