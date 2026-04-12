package application

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	storagecfg "github.com/Agent-Field/agentfield/control-plane/internal/storage"
)

func TestCreateServiceContainerWithDefaults(t *testing.T) {
	agentfieldHome := t.TempDir()

	container := CreateServiceContainerWithDefaults(agentfieldHome)

	if container == nil {
		t.Fatal("expected service container")
	}
	if container.PackageService == nil {
		t.Fatal("expected package service to be initialized")
	}
	if container.AgentService == nil {
		t.Fatal("expected agent service to be initialized")
	}
	if container.DevService == nil {
		t.Fatal("expected dev service to be initialized")
	}
	if container.DIDService != nil {
		t.Fatal("expected DID service to be disabled by default")
	}
	if container.VCService != nil {
		t.Fatal("expected VC service to be disabled by default")
	}
}

func TestGenerateAgentFieldServerID(t *testing.T) {
	t.Run("uses absolute path when available", func(t *testing.T) {
		agentfieldHome := t.TempDir()

		got := generateAgentFieldServerID(agentfieldHome)

		absPath, err := filepath.Abs(agentfieldHome)
		if err != nil {
			t.Fatalf("filepath.Abs failed: %v", err)
		}
		sum := sha256.Sum256([]byte(absPath))
		want := hex.EncodeToString(sum[:])[:16]

		if got != want {
			t.Fatalf("expected %q, got %q", want, got)
		}
		if got != generateAgentFieldServerID(agentfieldHome) {
			t.Fatal("expected deterministic server ID")
		}
	})

	t.Run("falls back to original path when absolute lookup fails", func(t *testing.T) {
		originalWD, err := os.Getwd()
		if err != nil {
			t.Fatalf("Getwd failed: %v", err)
		}

		badWD := t.TempDir()
		if err := os.Chdir(badWD); err != nil {
			t.Fatalf("Chdir to temp dir failed: %v", err)
		}

		if err := os.RemoveAll(badWD); err != nil {
			_ = os.Chdir(originalWD)
			t.Fatalf("RemoveAll failed: %v", err)
		}

		t.Cleanup(func() {
			if err := os.Chdir(originalWD); err != nil {
				t.Fatalf("failed to restore working directory: %v", err)
			}
		})

		input := filepath.Join("relative", "agentfield-home")
		got := generateAgentFieldServerID(input)

		sum := sha256.Sum256([]byte(input))
		want := hex.EncodeToString(sum[:])[:16]

		if got != want {
			t.Fatalf("expected fallback hash %q, got %q", want, got)
		}
	})
}

func TestCreateServiceContainerDIDAdditionalBranches(t *testing.T) {
	t.Run("continues without DID service when keystore creation fails", func(t *testing.T) {
		agentfieldHome := t.TempDir()
		keystorePath := filepath.Join(agentfieldHome, "keystore-file")
		if err := os.WriteFile(keystorePath, []byte("not-a-directory"), 0600); err != nil {
			t.Fatalf("WriteFile failed: %v", err)
		}

		cfg := &config.Config{}
		cfg.Storage.Mode = "local"
		cfg.Storage.Local = storagecfg.LocalStorageConfig{
			DatabasePath: filepath.Join(agentfieldHome, "agentfield.db"),
			KVStorePath:  filepath.Join(agentfieldHome, "agentfield.bolt"),
		}
		cfg.Features.DID.Enabled = true
		cfg.Features.DID.Keystore.Path = keystorePath

		container := CreateServiceContainer(cfg, agentfieldHome)

		if container.StorageProvider == nil {
			t.Fatal("expected storage provider to initialize")
		}
		if container.KeystoreService != nil {
			t.Fatal("expected keystore service to be nil after keystore initialization failure")
		}
		if container.DIDRegistry == nil {
			t.Fatal("expected DID registry to initialize with valid storage")
		}
		if container.DIDService != nil {
			t.Fatal("expected DID service to remain nil without a keystore")
		}
		if container.VCService != nil {
			t.Fatal("expected VC service to remain nil without DID service")
		}
	})

	t.Run("initializes DID services with encrypted registry support", func(t *testing.T) {
		agentfieldHome := t.TempDir()

		cfg := &config.Config{}
		cfg.Storage.Mode = "local"
		cfg.Storage.Local = storagecfg.LocalStorageConfig{
			DatabasePath: filepath.Join(agentfieldHome, "agentfield.db"),
			KVStorePath:  filepath.Join(agentfieldHome, "agentfield.bolt"),
		}
		cfg.Features.DID.Enabled = true
		cfg.Features.DID.Keystore.Path = filepath.Join(agentfieldHome, "keys")
		cfg.Features.DID.Keystore.EncryptionPassphrase = "top-secret-passphrase"

		container := CreateServiceContainer(cfg, agentfieldHome)

		if container.KeystoreService == nil {
			t.Fatal("expected keystore service to initialize")
		}
		if container.DIDRegistry == nil {
			t.Fatal("expected DID registry to initialize")
		}
		if container.DIDService == nil {
			t.Fatal("expected DID service to initialize")
		}
		if container.VCService == nil {
			t.Fatal("expected VC service to initialize")
		}
	})
}
