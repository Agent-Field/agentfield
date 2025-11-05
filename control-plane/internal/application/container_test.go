package application

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	storagecfg "github.com/Agent-Field/agentfield/control-plane/internal/storage"
)

func TestCreateServiceContainerWithoutDID(t *testing.T) {
	t.Parallel()

	agentfieldHome := t.TempDir()
	cfg := &config.Config{}

	container := CreateServiceContainer(cfg, agentfieldHome)

	if container.PackageService == nil || container.AgentService == nil || container.DevService == nil {
		t.Fatalf("expected core services to be initialised")
	}
	if container.DIDService != nil || container.VCService != nil {
		t.Fatalf("expected DID services to be nil when feature disabled")
	}
}

func TestCreateServiceContainerDIDWithoutStorageFallback(t *testing.T) {
	t.Parallel()

	agentfieldHome := t.TempDir()
	cfg := &config.Config{}
	cfg.Features.DID.Enabled = true
	cfg.Features.DID.Keystore.Path = filepath.Join(agentfieldHome, "keys")
	cfg.Storage.Mode = "invalid"

	container := CreateServiceContainer(cfg, agentfieldHome)

	if container.DIDService != nil || container.VCService != nil {
		t.Fatalf("expected DID services to remain nil when storage initialisation fails")
	}
}

func TestCreateServiceContainerWithLocalDID(t *testing.T) {
	t.Parallel()

	agentfieldHome := t.TempDir()
	cfg := &config.Config{}
	cfg.Storage.Mode = "local"
	cfg.Storage.Local = storagecfg.LocalStorageConfig{
		DatabasePath: filepath.Join(agentfieldHome, "agentfield.db"),
		KVStorePath:  filepath.Join(agentfieldHome, "agentfield.bolt"),
	}
	cfg.Features.DID.Enabled = true
	cfg.Features.DID.Keystore.Path = filepath.Join(agentfieldHome, "keys")

	ctx := context.Background()
	probe := storagecfg.NewLocalStorage(storagecfg.LocalStorageConfig{})
	storageConfig := storagecfg.StorageConfig{
		Mode:  cfg.Storage.Mode,
		Local: cfg.Storage.Local,
	}
	if err := probe.Initialize(ctx, storageConfig); err != nil {
		if strings.Contains(strings.ToLower(err.Error()), "fts5") {
			t.Skip("sqlite3 compiled without FTS5; skipping DID container test")
		}
		t.Fatalf("failed to initialise local storage: %v", err)
	}
	if err := probe.Close(ctx); err != nil {
		t.Fatalf("failed to close probe storage: %v", err)
	}

	container := CreateServiceContainer(cfg, agentfieldHome)

	if container.DIDService == nil {
		t.Fatalf("expected DID service to be initialised when configuration is valid")
	}
	if container.VCService == nil {
		t.Fatalf("expected VC service to be initialised when configuration is valid")
	}
	if container.StorageProvider == nil {
		t.Fatalf("expected storage provider to be initialised for DID services")
	}
}
