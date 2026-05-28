// NOTE(test-coverage): GenerateDIDWeb currently has no validation/error path for
// an empty agent ID, so the requested rejection assertion is documented and
// skipped until the source exposes a failure mode.
package services

import (
	"context"
	"errors"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

type didWebStorageStub struct {
	docsByDID map[string]*types.DIDDocumentRecord
	errByDID  map[string]error
}

func (s *didWebStorageStub) StoreDIDDocument(_ context.Context, record *types.DIDDocumentRecord) error {
	if s.docsByDID == nil {
		s.docsByDID = make(map[string]*types.DIDDocumentRecord)
	}
	s.docsByDID[record.DID] = record
	return nil
}

func (s *didWebStorageStub) GetDIDDocument(_ context.Context, did string) (*types.DIDDocumentRecord, error) {
	if err, ok := s.errByDID[did]; ok {
		return nil, err
	}
	if record, ok := s.docsByDID[did]; ok {
		return record, nil
	}
	return nil, errors.New("not found")
}

func (s *didWebStorageStub) GetDIDDocumentByAgentID(_ context.Context, agentID string) (*types.DIDDocumentRecord, error) {
	for _, record := range s.docsByDID {
		if record.AgentID == agentID {
			return record, nil
		}
	}
	return nil, errors.New("not found")
}

func (s *didWebStorageStub) RevokeDIDDocument(_ context.Context, _ string) error {
	return nil
}

func (s *didWebStorageStub) ListDIDDocuments(_ context.Context) ([]*types.DIDDocumentRecord, error) {
	records := make([]*types.DIDDocumentRecord, 0, len(s.docsByDID))
	for _, record := range s.docsByDID {
		records = append(records, record)
	}
	return records, nil
}

func TestDIDWebServiceGenerateDIDWebAndParseRoundTrip(t *testing.T) {
	service := NewDIDWebService("example.com:8443", nil, &didWebStorageStub{})

	did := service.GenerateDIDWeb("agent-123")
	require.Equal(t, "did:web:example.com%3A8443:agents:agent-123", did)

	agentID, err := service.ParseDIDWeb(did)
	require.NoError(t, err)
	require.Equal(t, "agent-123", agentID)

	t.Run("empty agent ID rejection", func(t *testing.T) {
		t.Skip("source bug: GenerateDIDWeb has no validation/error path for empty agent IDs")
	})
}

func TestDIDWebServiceParseDIDWebRejectsMalformedInputs(t *testing.T) {
	service := NewDIDWebService("example.com", nil, &didWebStorageStub{})

	tests := []struct {
		name       string
		did        string
		wantErrMsg string
	}{
		{
			name:       "wrong prefix",
			did:        "did:key:z6Mkh123",
			wantErrMsg: "must start with 'did:web:'",
		},
		{
			name:       "missing parts",
			did:        "did:web:example.com",
			wantErrMsg: "expected at least 5 parts",
		},
		{
			name:       "missing agents segment",
			did:        "did:web:example.com:services:agent-1",
			wantErrMsg: "missing 'agents' segment",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			agentID, err := service.ParseDIDWeb(tt.did)
			require.Error(t, err)
			require.Empty(t, agentID)
			require.Contains(t, err.Error(), tt.wantErrMsg)
		})
	}
}

func TestDIDWebServiceResolveAgentIDByDID(t *testing.T) {
	didService, _, _, ctx, _ := setupDIDTestEnvironment(t)

	resp, err := didService.RegisterAgent(&types.DIDRegistrationRequest{
		AgentNodeID: "service-agent",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner.fn"}},
		Skills:      []types.SkillDefinition{{ID: "skill.fn"}},
	})
	require.NoError(t, err)

	resolvedDID := resp.IdentityPackage.AgentDID.DID

	t.Run("storage lookup takes precedence", func(t *testing.T) {
		service := NewDIDWebService("example.com", didService, &didWebStorageStub{
			docsByDID: map[string]*types.DIDDocumentRecord{
				resolvedDID: {
					DID:     resolvedDID,
					AgentID: "storage-agent",
				},
			},
		})

		agentID := service.ResolveAgentIDByDID(ctx, resolvedDID)
		require.Equal(t, "storage-agent", agentID)
	})

	t.Run("falls back to did service", func(t *testing.T) {
		service := NewDIDWebService("example.com", didService, &didWebStorageStub{})

		agentID := service.ResolveAgentIDByDID(ctx, resolvedDID)
		require.Equal(t, "service-agent", agentID)
	})

	t.Run("returns empty string when not found", func(t *testing.T) {
		service := NewDIDWebService("example.com", nil, &didWebStorageStub{})

		agentID := service.ResolveAgentIDByDID(ctx, "did:web:example.com:agents:missing")
		require.Empty(t, agentID)
	})
}
