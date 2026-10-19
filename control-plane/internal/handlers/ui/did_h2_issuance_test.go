package ui

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/cli"
	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/handlers"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

// Exercise SDK-facing issuance, both chain export handlers and the offline CLI
// verifier before and after storage is reopened. Gin recorders need no listener.
func TestH2DIDExecutionVCIssuanceExportVerification(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, mode := range []string{"local", "postgres"} {
		t.Run(mode, func(t *testing.T) {
			dir := t.TempDir()
			off := false
			storageCfg := storage.StorageConfig{Mode: mode, Vector: storage.VectorStoreConfig{Enabled: &off}}
			if mode == "local" {
				storageCfg.Local = storage.LocalStorageConfig{DatabasePath: filepath.Join(dir, "agentfield.db"), KVStorePath: filepath.Join(dir, "kv.bolt")}
			} else {
				dsn := os.Getenv("POSTGRES_TEST_URL")
				if dsn == "" {
					t.Skip("POSTGRES_TEST_URL unset")
				}
				admin, err := sql.Open("pgx", dsn)
				require.NoError(t, err)
				defer admin.Close()
				name := fmt.Sprintf("fix3_h2_http_%d", time.Now().UnixNano())
				_, err = admin.Exec("CREATE DATABASE " + name)
				require.NoError(t, err)
				defer func() {
					_, err := admin.Exec("DROP DATABASE " + name + " WITH (FORCE)")
					require.NoError(t, err)
				}()
				u, err := url.Parse(dsn)
				require.NoError(t, err)
				u.Path = "/" + name
				storageCfg.Postgres.DSN = u.String()
			}
			cfg := &config.DIDConfig{Enabled: true,
				Keystore:       config.KeystoreConfig{Type: "local", Path: filepath.Join(dir, "keys")},
				VCRequirements: config.VCRequirements{RequireVCForExecution: true, PersistExecutionVC: true, HashSensitiveData: true},
			}
			var provider *storage.LocalStorage
			open := func() *gin.Engine {
				if mode == "local" {
					provider = storage.NewLocalStorage(storageCfg.Local)
				} else {
					provider = storage.NewPostgresStorage(storageCfg.Postgres)
				}
				require.NoError(t, provider.Initialize(context.Background(), storageCfg))
				registry := services.NewDIDRegistryWithStorage(provider)
				require.NoError(t, registry.Initialize())
				ks, err := services.NewKeystoreService(&cfg.Keystore)
				require.NoError(t, err)
				did := services.NewDIDService(cfg, ks, registry)
				require.NoError(t, did.Initialize("server-a"))
				vc := services.NewVCService(cfg, did, provider)
				h := handlers.NewDIDHandlers(did, vc)
				r := gin.New()
				r.POST("/api/v1/did/register", h.RegisterAgent)
				r.POST("/api/v1/execution/vc", h.CreateExecutionVC)
				r.GET("/api/v1/did/workflow/:workflow_id/vc-chain", h.GetWorkflowVCChain)
				r.GET("/api/ui/v1/workflows/:workflowId/vc-chain", NewDIDHandler(provider, did, vc, nil).GetWorkflowVCChainHandler)
				return r
			}
			router := open()
			defer func() { require.NoError(t, provider.Close(context.Background())) }()
			var pkg types.DIDIdentityPackage
			originalDocuments := make(map[string]json.RawMessage)
			for restart := 0; restart < 2; restart++ {
				request := types.DIDRegistrationRequest{AgentNodeID: "node", Reasoners: []types.ReasonerDefinition{{ID: "r"}}}
				resp := performJSONRequest(router, http.MethodPost, "/api/v1/did/register", request)
				require.Equal(t, http.StatusOK, resp.Code)
				var registered types.DIDRegistrationResponse
				require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &registered))
				require.True(t, registered.Success, registered.Error)
				if restart == 0 {
					pkg = registered.IdentityPackage
					for _, name := range []string{"explicit", "fallback"} {
						executionContext := map[string]any{
							"execution_id": name, "workflow_id": name, "session_id": "s",
							"agent_node_did": pkg.AgentDID.DID, "timestamp": time.Now().UTC().Format(time.RFC3339),
						}
						if name == "explicit" {
							executionContext["caller_did"] = pkg.ReasonerDIDs["r"].DID
						}
						issued := performJSONRequest(router, http.MethodPost, "/api/v1/execution/vc", map[string]any{
							"execution_context": executionContext,
							"input_data":        "e30=", "output_data": "e30=", "status": "succeeded", "duration_ms": 1,
						})
						require.Equal(t, http.StatusOK, issued.Code, issued.Body.String())
					}
				} else {
					require.Equal(t, pkg.AgentDID.DID, registered.IdentityPackage.AgentDID.DID)
					require.Equal(t, pkg.ReasonerDIDs["r"].DID, registered.IdentityPackage.ReasonerDIDs["r"].DID)
				}
				for _, name := range []string{"explicit", "fallback"} {
					for _, path := range []string{"/api/v1/did/workflow/" + name + "/vc-chain", "/api/ui/v1/workflows/" + name + "/vc-chain"} {
						chain := performJSONRequest(router, http.MethodGet, path, nil)
						require.Equal(t, http.StatusOK, chain.Code, chain.Body.String())
						require.Empty(t, chain.Header().Get(handlers.VCChainVerifiableHeader))
						var parsed types.WorkflowVCChainResponse
						require.NoError(t, json.Unmarshal(chain.Body.Bytes(), &parsed))
						require.True(t, parsed.Verifiable)
						verified := cli.VerifyProvenanceJSON(chain.Body.Bytes(), cli.VerifyOptions{})
						require.True(t, verified.Valid, "%+v", verified)
						require.Len(t, parsed.ComponentVCs, 1)
						execution := parsed.ComponentVCs[0]
						caller := pkg.AgentDID.DID
						if name == "explicit" {
							caller = pkg.ReasonerDIDs["r"].DID
						}
						require.Equal(t, caller, execution.CallerDID)
						require.Equal(t, caller, execution.IssuerDID)
						var document types.VCDocument
						require.NoError(t, json.Unmarshal(execution.VCDocument, &document))
						require.Equal(t, caller, document.CredentialSubject.Caller.DID)
						require.Equal(t, caller, document.Issuer)
						require.Equal(t, caller+"#key-1", document.Proof.VerificationMethod)
						if original, ok := originalDocuments[name]; ok {
							require.Equal(t, original, execution.VCDocument, "export and reload must preserve signed bytes")
						} else {
							originalDocuments[name] = append(json.RawMessage(nil), execution.VCDocument...)
						}
					}
				}
				if restart == 0 {
					require.NoError(t, provider.Close(context.Background()))
					router = open()
				}
			}
		})
	}
}
