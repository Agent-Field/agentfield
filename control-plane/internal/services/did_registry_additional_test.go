package services

import (
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/encryption"
	"github.com/stretchr/testify/require"
)

func TestDIDRegistrySetEncryptionServiceAndDeleteRegistry(t *testing.T) {
	registry := NewDIDRegistryWithStorage(nil)
	service := &encryption.EncryptionService{}

	registry.SetEncryptionService(service)
	require.Same(t, service, registry.encryptionService)

	registry.registries["server-1"] = nil
	require.NoError(t, registry.DeleteRegistry("server-1"))
	require.NotContains(t, registry.registries, "server-1")
}
