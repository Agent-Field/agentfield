package services

import (
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/stretchr/testify/require"
)

func TestVCServiceGetDIDService(t *testing.T) {
	didService, _, _, _, _ := setupDIDTestEnvironment(t)
	service := NewVCService(&config.DIDConfig{Enabled: true}, didService, nil)
	require.Same(t, didService, service.GetDIDService())
}

func TestVCServiceInitializeAndGetExecutionVCByExecutionID(t *testing.T) {
	vcService, _, provider, ctx := setupVCTestEnvironment(t)
	require.NoError(t, vcService.Initialize())

	require.NoError(t, provider.StoreExecutionVC(ctx, "vc-wrapper", "exec-wrapper", "workflow-wrapper", "session-wrapper", "did:key:issuer", "did:key:target", "did:key:caller", "input", "output", "succeeded", []byte(`{"ok":true}`), "sig", "", 12))

	vc, err := vcService.GetExecutionVCByExecutionID("exec-wrapper")
	require.NoError(t, err)
	require.Equal(t, "vc-wrapper", vc.VCID)

	disabled := NewVCService(&config.DIDConfig{Enabled: false}, nil, nil)
	require.NoError(t, disabled.Initialize())
	_, err = disabled.GetExecutionVCByExecutionID("exec-wrapper")
	require.ErrorContains(t, err, "DID system is disabled")
}
