package services

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"path/filepath"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

func TestG3VCFallbackCallerIssuer(t *testing.T) {
	env := newContinuityEnv(t)
	env.restart()
	svc := env.service(env.registry(), "server-a")
	pkg := register(t, svc, "node", nil, nil).IdentityPackage
	vc := NewVCService(svc.config, svc, env.provider)
	execution, err := vc.GenerateExecutionVC(&types.ExecutionContext{
		ExecutionID: "external", WorkflowID: "fallback-wf", SessionID: "s",
		AgentNodeDID: pkg.AgentDID.DID, Timestamp: time.Now(),
	}, []byte(`{}`), []byte(`{}`), "succeeded", nil, 1)
	require.NoError(t, err)
	require.NotNil(t, execution)
	var document types.VCDocument
	require.NoError(t, json.Unmarshal(execution.VCDocument, &document))
	require.Equal(t, pkg.AgentDID.DID, document.Issuer)
	require.Equal(t, execution.IssuerDID, document.Issuer)
	require.Equal(t, document.Issuer+"#key-1", document.Proof.VerificationMethod)
	verified, err := vc.VerifyVC(execution.VCDocument)
	require.NoError(t, err)
	require.True(t, verified.Valid, verified.Reason)
	chain, err := vc.GetWorkflowVCChain("fallback-wf")
	require.NoError(t, err)
	require.True(t, chain.Verifiable)
	require.Empty(t, chain.UnresolvedDIDs)
}

func TestG3VCHistoricalDocumentIssuer(t *testing.T) {
	for _, kind := range []string{"empty", "mismatched-resolvable", "unresolvable", "malformed"} {
		t.Run(kind, func(t *testing.T) {
			env := newContinuityEnv(t)
			env.restart()
			svc := env.service(env.registry(), "server-a")
			pkg := register(t, svc, "node", nil, nil).IdentityPackage
			vc := NewVCService(svc.config, svc, env.provider)
			execution, err := vc.GenerateExecutionVC(&types.ExecutionContext{
				ExecutionID: "historical", WorkflowID: "wf", SessionID: "s",
				CallerDID: pkg.AgentDID.DID, AgentNodeDID: pkg.AgentDID.DID, Timestamp: time.Now(),
			}, nil, nil, "succeeded", nil, 0)
			require.NoError(t, err)
			var document types.VCDocument
			require.NoError(t, json.Unmarshal(execution.VCDocument, &document))
			document.Issuer = ""
			if kind == "mismatched-resolvable" {
				document.Issuer, err = svc.GetControlPlaneIssuerDID()
				require.NoError(t, err)
			} else if kind == "unresolvable" {
				document.Issuer = "did:web:127.0.0.1:missing"
			}
			// Reproduce the old issuance defect: a valid signature over an empty
			// document issuer, with a valid signer retained in execution metadata.
			identity, err := svc.ResolveDID(pkg.AgentDID.DID)
			require.NoError(t, err)
			document.Proof.ProofValue, err = vc.signVC(&document, identity)
			require.NoError(t, err)
			raw, err := json.Marshal(document)
			require.NoError(t, err)
			if kind == "malformed" {
				raw = []byte(`{"issuer":`)
			}
			db, err := sql.Open("sqlite3", filepath.Join(env.stateDir, "agentfield.db"))
			require.NoError(t, err)
			defer db.Close()
			_, err = db.Exec("UPDATE execution_vcs SET vc_document = ? WHERE vc_id = ?", string(raw), execution.VCID)
			require.NoError(t, err)
			chain, err := vc.GetWorkflowVCChain("wf")
			require.NoError(t, err)
			require.False(t, chain.Verifiable)
			expected := document.Issuer
			if kind == "empty" || kind == "malformed" {
				expected = fmt.Sprintf("execution-vc:%s:document-issuer:empty-or-invalid", execution.VCID)
			}
			require.Equal(t, []string{expected}, chain.UnresolvedDIDs)
			require.Contains(t, chain.DIDResolutionBundle, pkg.AgentDID.DID)
			if kind == "mismatched-resolvable" {
				require.Contains(t, chain.DIDResolutionBundle, document.Issuer)
			}
		})
	}
}

func TestG3VCNormalTwoStepChain(t *testing.T) {
	env := newContinuityEnv(t)
	env.restart()
	svc := env.service(env.registry(), "server-a")
	pkg := register(t, svc, "node", []string{"r"}, []string{"s"}).IdentityPackage
	vc := NewVCService(svc.config, svc, env.provider)
	for step, caller := range []string{pkg.ReasonerDIDs["r"].DID, pkg.SkillDIDs["s"].DID} {
		execution, err := vc.GenerateExecutionVC(&types.ExecutionContext{
			ExecutionID: fmt.Sprintf("step-%d", step), WorkflowID: "wf", SessionID: "s",
			CallerDID: caller, AgentNodeDID: pkg.AgentDID.DID, Timestamp: time.Now(),
		}, nil, nil, "succeeded", nil, 0)
		require.NoError(t, err)
		verified, err := vc.VerifyVC(execution.VCDocument)
		require.NoError(t, err)
		require.True(t, verified.Valid, verified.Reason)
	}
	chain, err := vc.GetWorkflowVCChain("wf")
	require.NoError(t, err)
	require.Len(t, chain.ComponentVCs, 2)
	require.True(t, chain.Verifiable)
	require.Empty(t, chain.UnresolvedDIDs)
}
