package packages

import (
	"path/filepath"
	"testing"
)

func TestE13ConcurrentReconcileAndRestoreRegistryWritesPreserveBothPackages(t *testing.T) {
	path := filepath.Join(t.TempDir(), "installed.yaml")
	if err := WriteInstallationRegistry(path, &InstallationRegistry{Installed: map[string]InstalledPackage{
		"read-reconcile": {Name: "read-reconcile", Status: "running"},
		"restore":        {Name: "restore", Status: "stopped", DesiredState: DesiredStateRunning},
	}}); err != nil {
		t.Fatal(err)
	}

	enteredSlowWrite := make(chan struct{})
	releaseSlowWrite := make(chan struct{})
	firstDone := make(chan error, 1)
	go func() {
		firstDone <- UpdateInstallationRegistry(path, func(registry *InstallationRegistry) error {
			entry := registry.Installed["read-reconcile"]
			entry.Status = "stopped"
			entry.Runtime.PID = nil
			registry.Installed["read-reconcile"] = entry
			close(enteredSlowWrite)
			<-releaseSlowWrite
			return nil
		})
	}()
	<-enteredSlowWrite

	secondDone := make(chan error, 1)
	go func() {
		secondDone <- UpdateInstallationRegistryAtomic(path, func(registry *InstallationRegistry) error {
			entry := registry.Installed["restore"]
			pid, port := 222, 8222
			entry.Status = "running"
			entry.Runtime.PID = &pid
			entry.Runtime.Port = &port
			registry.Installed["restore"] = entry
			return nil
		})
	}()
	close(releaseSlowWrite)
	if err := <-firstDone; err != nil {
		t.Fatal(err)
	}
	if err := <-secondDone; err != nil {
		t.Fatal(err)
	}

	registry, err := LoadInstallationRegistry(path)
	if err != nil {
		t.Fatal(err)
	}
	if registry.Installed["read-reconcile"].Status != "stopped" {
		t.Fatalf("reconcile update was lost: %+v", registry.Installed["read-reconcile"])
	}
	restored := registry.Installed["restore"]
	if restored.Status != "running" || restored.Runtime.PID == nil || *restored.Runtime.PID != 222 {
		t.Fatalf("restore update was lost: %+v", restored)
	}
}
