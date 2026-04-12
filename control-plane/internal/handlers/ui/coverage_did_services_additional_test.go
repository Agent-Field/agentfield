package ui

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func setupDIDHandlerServices(t *testing.T) (*storage.LocalStorage, *services.DIDRegistry, *services.DIDService, *services.VCService, *services.DIDWebService, context.Context) {
	t.Helper()

	ctx := context.Background()
	cfg := storage.StorageConfig{
		Mode: "local",
		Local: storage.LocalStorageConfig{
			DatabasePath: filepath.Join(t.TempDir(), "agentfield.db"),
			KVStorePath:  filepath.Join(t.TempDir(), "agentfield.bolt"),
		},
	}

	ls := storage.NewLocalStorage(storage.LocalStorageConfig{})
	err := ls.Initialize(ctx, cfg)
	if err != nil && strings.Contains(strings.ToLower(err.Error()), "fts5") {
		t.Skip("sqlite3 compiled without FTS5")
	}
	require.NoError(t, err)
	t.Cleanup(func() {
		_ = ls.Close(ctx)
	})

	registry := services.NewDIDRegistryWithStorage(ls)
	require.NoError(t, registry.Initialize())

	keystoreDir := filepath.Join(t.TempDir(), "keys")
	didCfg := &config.DIDConfig{
		Enabled: true,
		Keystore: config.KeystoreConfig{
			Path: keystoreDir,
			Type: "local",
		},
		VCRequirements: config.VCRequirements{
			RequireVCForExecution: true,
			PersistExecutionVC:    true,
			HashSensitiveData:     true,
		},
	}
	keystore, err := services.NewKeystoreService(&didCfg.Keystore)
	require.NoError(t, err)

	didService := services.NewDIDService(didCfg, keystore, registry)
	require.NoError(t, didService.Initialize("srv-ui"))

	vcService := services.NewVCService(didCfg, didService, ls)
	require.NoError(t, vcService.Initialize())

	didWebService := services.NewDIDWebService("example.test", didService, ls)

	return ls, registry, didService, vcService, didWebService, ctx
}

func seedDIDHandlerData(t *testing.T, ls *storage.LocalStorage, registry *services.DIDRegistry, vcService *services.VCService, didWebService *services.DIDWebService, ctx context.Context) string {
	t.Helper()

	now := time.Date(2026, 4, 8, 12, 0, 0, 0, time.UTC)

	reg, err := registry.GetRegistry("srv-ui")
	require.NoError(t, err)
	require.NotNil(t, reg)
	reg.AgentNodes["node-1"] = types.AgentDIDInfo{
		DID:                "did:agent:node-1",
		AgentNodeID:        "node-1",
		AgentFieldServerID: "srv-ui",
		PublicKeyJWK:       json.RawMessage(`{"kty":"OKP","crv":"Ed25519","x":"abc"}`),
		DerivationPath:     "m/44'/0'/1'",
		Reasoners: map[string]types.ReasonerDIDInfo{
			"plan": {
				DID:            "did:reasoner:plan",
				FunctionName:   "plan",
				Capabilities:   []string{"search", "draft"},
				ExposureLevel:  "public",
				DerivationPath: "m/44'/0'/1'/1'",
				CreatedAt:      now,
			},
		},
		Skills: map[string]types.SkillDIDInfo{
			"toolbox": {
				DID:            "did:skill:toolbox",
				FunctionName:   "toolbox",
				Tags:           []string{"ops"},
				ExposureLevel:  "private",
				DerivationPath: "m/44'/0'/1'/2'",
				CreatedAt:      now,
			},
		},
		Status:       types.AgentDIDStatusActive,
		RegisteredAt: now,
	}
	require.NoError(t, registry.StoreRegistry(reg))

	didDoc, _, err := didWebService.GetOrCreateDIDDocument(ctx, "node-1")
	require.NoError(t, err)
	require.Equal(t, didWebService.GenerateDIDWeb("node-1"), didDoc.ID)

	require.NoError(t, ls.StoreExecutionVC(ctx,
		"vc-good", "exec-good", "wf-1", "session-1",
		"did:agent:node-1", "did:target:node-1", "did:caller:client",
		"hash-in-1", "hash-out-1", string(types.ExecutionStatusSucceeded),
		[]byte(`{"id":"vc-good","issuer":"did:agent:node-1"}`), "sig", "", 48,
	))
	require.NoError(t, ls.StoreExecutionVC(ctx,
		"vc-failed", "exec-failed", "wf-1", "session-1",
		"did:agent:node-1", "did:target:node-1", "did:caller:client",
		"hash-in-2", "hash-out-2", string(types.ExecutionStatusFailed),
		[]byte(`{"id":"vc-failed"}`), "sig", "", 24,
	))
	require.NoError(t, ls.StoreExecutionVC(ctx,
		"vc-malformed", "exec-malformed", "wf-2", "session-2",
		"did:issuer:outside", "did:target:node-1", "did:caller:client",
		"hash-in-3", "hash-out-3", "verified",
		[]byte(`{invalid`), "sig", "", 7,
	))
	require.NoError(t, ls.StoreExecutionVC(ctx,
		"vc-external", "exec-external", "wf-3", "session-3",
		"did:issuer:outside", "did:target:node-1", "did:caller:client",
		"hash-in-4", "hash-out-4", "pending",
		[]byte{}, "sig", "s3://bucket/vc-external.json", 99,
	))
	require.NoError(t, ls.StoreExecutionVC(ctx,
		"vc-empty", "exec-empty", "wf-4", "session-4",
		"did:issuer:outside", "did:target:other", "did:caller:client",
		"hash-in-5", "hash-out-5", "pending",
		[]byte{}, "sig", "", 0,
	))
	_, err = vcService.CreateWorkflowVC("wf-1", "session-1", []string{"vc-good", "vc-failed"})
	require.NoError(t, err)

	return didWebService.GenerateDIDWeb("node-1")
}

func TestDIDHandlerCoverageWithActiveServices(t *testing.T) {
	gin.SetMode(gin.TestMode)

	ls, registry, didService, vcService, didWebService, ctx := setupDIDHandlerServices(t)
	didWeb := seedDIDHandlerData(t, ls, registry, vcService, didWebService, ctx)

	handler := NewDIDHandler(ls, didService, vcService, didWebService)
	router := gin.New()
	router.GET("/api/ui/v1/nodes/:nodeId/did", handler.GetNodeDIDHandler)
	router.GET("/api/ui/v1/nodes/:nodeId/vc-status", handler.GetNodeVCStatusHandler)
	router.GET("/api/ui/v1/executions/:executionId/vc-status", handler.GetExecutionVCStatusHandler)
	router.GET("/api/ui/v1/agent-api/executions/:execution_id/vc-status", handler.GetExecutionVCStatusHandler)
	router.GET("/api/ui/v1/executions/:executionId/vc", handler.GetExecutionVCHandler)
	router.POST("/api/ui/v1/workflows/vc-status", handler.GetWorkflowVCStatusBatchHandler)
	router.GET("/api/ui/v1/workflows/:workflowId/vc-chain", handler.GetWorkflowVCChainHandler)
	router.GET("/api/ui/v1/vc/:vc_id/download", handler.DownloadVCHandler)
	router.POST("/api/ui/v1/vc/verify", handler.VerifyVCHandler)
	router.POST("/api/ui/v1/did/verify-audit", handler.VerifyAuditBundleHandler)
	router.POST("/api/ui/v1/executions/:executionId/verify-vc", handler.VerifyExecutionVCComprehensiveHandler)
	router.POST("/api/ui/v1/workflows/:workflowId/verify-vc", handler.VerifyWorkflowVCComprehensiveHandler)
	router.GET("/api/ui/v1/did/export/vcs", handler.ExportVCsHandler)
	router.GET("/api/ui/v1/did/status", handler.GetDIDSystemStatusHandler)
	router.GET("/api/ui/v1/did/:did/resolution-bundle", handler.GetDIDResolutionBundleHandler)
	router.GET("/api/ui/v1/did/:did/resolution-bundle/download", handler.DownloadDIDResolutionBundleHandler)

	t.Run("node did and vc summaries", func(t *testing.T) {
		rec := performJSONRequest(router, http.MethodGet, "/api/ui/v1/nodes/node-1/did", nil)
		require.Equal(t, http.StatusOK, rec.Code)
		body := decodeJSONResponse[map[string]any](t, rec)
		require.Equal(t, "did:agent:node-1", body["did"])
		require.Equal(t, didWeb, body["did_web"])
		require.Equal(t, "active", body["status"])

		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/nodes/node-1/vc-status", nil)
		require.Equal(t, http.StatusOK, rec.Code)
		body = decodeJSONResponse[map[string]any](t, rec)
		require.Equal(t, true, body["has_vcs"])
		require.Equal(t, float64(4), body["vc_count"])
		require.Equal(t, float64(2), body["verified_count"])
		require.Equal(t, "failed", body["verification_status"])
	})

	t.Run("execution status variants", func(t *testing.T) {
		tests := []struct {
			name   string
			target string
			status string
		}{
			{name: "parsed document", target: "/api/ui/v1/executions/exec-good/vc-status", status: string(types.ExecutionStatusSucceeded)},
			{name: "malformed document", target: "/api/ui/v1/executions/exec-malformed/vc-status", status: "malformed"},
			{name: "external document", target: "/api/ui/v1/agent-api/executions/exec-external/vc-status", status: "external"},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				rec := performJSONRequest(router, http.MethodGet, tc.target, nil)
				require.Equal(t, http.StatusOK, rec.Code)
				body := decodeJSONResponse[map[string]any](t, rec)
				require.Equal(t, true, body["has_vc"])
				require.Equal(t, tc.status, body["status"])
			})
		}
	})

	t.Run("execution vc retrieval and downloads", func(t *testing.T) {
		rec := performJSONRequest(router, http.MethodGet, "/api/ui/v1/executions/exec-good/vc", nil)
		require.Equal(t, http.StatusOK, rec.Code)
		require.Contains(t, rec.Body.String(), "vc-good")

		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/executions/exec-empty/vc", nil)
		require.Equal(t, http.StatusNotFound, rec.Code)

		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/vc/vc-good/download", nil)
		require.Equal(t, http.StatusOK, rec.Code)
		require.Contains(t, rec.Header().Get("Content-Disposition"), "vc-vc-good.json")

		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/vc/unknown/download", nil)
		require.Equal(t, http.StatusNotFound, rec.Code)
	})

	t.Run("workflow and export handlers", func(t *testing.T) {
		rec := performJSONRequest(router, http.MethodPost, "/api/ui/v1/workflows/vc-status", map[string]any{
			"workflow_ids": []string{"wf-1", "wf-missing"},
		})
		require.Equal(t, http.StatusInternalServerError, rec.Code)

		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/workflows/wf-1/vc-chain", nil)
		require.Equal(t, http.StatusOK, rec.Code)
		chain := decodeJSONResponse[map[string]any](t, rec)
		require.Equal(t, "wf-1", chain["workflow_id"])
		require.Equal(t, float64(2), chain["total_steps"])

		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/did/export/vcs?workflow_id=wf-1&status=failed&limit=2&offset=0", nil)
		require.Equal(t, http.StatusOK, rec.Code)
		exported := decodeJSONResponse[map[string]any](t, rec)
		require.Equal(t, float64(1), exported["total_count"])
	})

	t.Run("verification wrappers and did bundle endpoints", func(t *testing.T) {
		rec := performJSONRequest(router, http.MethodPost, "/api/ui/v1/vc/verify", map[string]any{
			"vc_document": map[string]any{"issuer": "did:unknown:issuer"},
		})
		require.Equal(t, http.StatusOK, rec.Code)
		verifyBody := decodeJSONResponse[map[string]any](t, rec)
		require.Equal(t, false, verifyBody["valid"])

		req := bytes.NewBufferString(`{"bundle":true}`)
		httpRec := performJSONRequest(router, http.MethodPost, "/api/ui/v1/did/verify-audit?resolve_web=true", json.RawMessage(req.Bytes()))
		require.Equal(t, http.StatusBadRequest, httpRec.Code)

		rec = performJSONRequest(router, http.MethodPost, "/api/ui/v1/executions/missing/verify-vc", nil)
		require.Equal(t, http.StatusOK, rec.Code)

		rec = performJSONRequest(router, http.MethodPost, "/api/ui/v1/workflows/missing/verify-vc", nil)
		require.Equal(t, http.StatusOK, rec.Code)

		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/did/status", nil)
		require.Equal(t, http.StatusOK, rec.Code)
		status := decodeJSONResponse[map[string]any](t, rec)
		require.Equal(t, "active", status["status"])

		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/did/did:agent:node-1/resolution-bundle", nil)
		require.Equal(t, http.StatusOK, rec.Code)
		bundle := decodeJSONResponse[map[string]any](t, rec)
		require.Equal(t, "resolved", bundle["resolution_status"])
		require.NotEmpty(t, bundle["related_vcs"])

		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/did/did:reasoner:plan/resolution-bundle", nil)
		require.Equal(t, http.StatusOK, rec.Code)
		componentBundle := decodeJSONResponse[map[string]any](t, rec)
		require.Equal(t, "resolved", componentBundle["resolution_status"])

		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/did/did:agent:node-1/resolution-bundle/download", nil)
		require.Equal(t, http.StatusOK, rec.Code)
		require.Contains(t, rec.Header().Get("Content-Disposition"), "did-resolution-bundle-")
		require.Contains(t, rec.Body.String(), `"status":"resolved"`)
	})
}
