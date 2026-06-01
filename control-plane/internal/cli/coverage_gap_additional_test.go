package cli

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/spf13/cobra"
	"github.com/stretchr/testify/require"

	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"gopkg.in/yaml.v3"
)

func TestStopAgentNodeShutdownPaths(t *testing.T) {
	t.Run("http shutdown success updates registry", func(t *testing.T) {
		home := t.TempDir()
		ln, err := net.Listen("tcp", "127.0.0.1:0")
		require.NoError(t, err)
		port := ln.Addr().(*net.TCPAddr).Port

		server := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			require.Equal(t, "/shutdown", r.URL.Path)
			require.Equal(t, http.MethodPost, r.Method)
			w.WriteHeader(http.StatusOK)
		})}
		go func() { _ = server.Serve(ln) }()
		t.Cleanup(func() {
			_ = server.Close()
		})

		cmd := exec.Command("sleep", "30")
		require.NoError(t, cmd.Start())
		t.Cleanup(func() {
			_ = cmd.Process.Kill()
			_, _ = cmd.Process.Wait()
		})

		pid := cmd.Process.Pid
		stopper := &AgentNodeStopper{AgentFieldHome: home}
		require.NoError(t, stopper.saveRegistry(makeRegistry("demo", "running", &port, &pid)))

		output := captureOutput(t, func() {
			require.NoError(t, stopper.StopAgentNode("demo"))
		})
		require.Contains(t, output, "HTTP shutdown request accepted")
		require.Contains(t, output, "stopped successfully")

		registry, err := stopper.loadRegistry()
		require.NoError(t, err)
		require.Equal(t, "stopped", registry.Installed["demo"].Status)
		require.Nil(t, registry.Installed["demo"].Runtime.Port)
		require.Nil(t, registry.Installed["demo"].Runtime.PID)
	})

	t.Run("fallback force kill succeeds when interrupt is ignored", func(t *testing.T) {
		home := t.TempDir()
		ln, err := net.Listen("tcp", "127.0.0.1:0")
		require.NoError(t, err)
		port := ln.Addr().(*net.TCPAddr).Port

		server := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusServiceUnavailable)
		})}
		go func() { _ = server.Serve(ln) }()
		t.Cleanup(func() {
			_ = server.Close()
		})

		cmd := exec.Command("sh", "-c", "trap '' INT; sleep 30")
		require.NoError(t, cmd.Start())
		t.Cleanup(func() {
			_ = cmd.Process.Kill()
			_, _ = cmd.Process.Wait()
		})

		pid := cmd.Process.Pid
		stopper := &AgentNodeStopper{AgentFieldHome: home}
		require.NoError(t, stopper.saveRegistry(makeRegistry("demo", "running", &port, &pid)))

		output := captureOutput(t, func() {
			require.NoError(t, stopper.StopAgentNode("demo"))
		})
		require.Contains(t, output, "Falling back to process signal shutdown")
		require.Contains(t, output, "force killing agent demo")
		require.Contains(t, output, "stopped successfully")
	})
}

func TestConfigPromptRetryAndVerifyProvenance(t *testing.T) {
	t.Run("verify provenance covers invalid and legacy workflow inputs", func(t *testing.T) {
		invalid := VerifyProvenanceJSON([]byte(`{"hello":"world"}`), VerifyOptions{})
		require.False(t, invalid.Valid)
		require.False(t, invalid.FormatValid)
		require.Contains(t, invalid.Error, "Invalid VC format")

		legacy := types.WorkflowVCChainResponse{
			WorkflowID: "wf-legacy",
			ComponentVCs: []types.ExecutionVC{
				{VCID: "vc-1", WorkflowID: "wf-legacy", Status: "completed"},
			},
			WorkflowVC: types.WorkflowVC{WorkflowID: "wf-legacy", Status: "completed"},
			Status:     "completed",
		}
		raw, err := json.Marshal(legacy)
		require.NoError(t, err)

		result := VerifyProvenanceJSON(raw, VerifyOptions{Verbose: true})
		require.Equal(t, "workflow", result.Type)
		require.Equal(t, "wf-legacy", result.WorkflowID)
		require.True(t, result.FormatValid)
		require.NotEmpty(t, result.VerificationSteps)
		require.Equal(t, 1, result.Summary.TotalComponents)
		require.GreaterOrEqual(t, result.Summary.TotalDIDs, 0)
	})
}

func TestLogsFollowAndConfigExitPaths(t *testing.T) {
	t.Run("follow logs returns error for missing file", func(t *testing.T) {
		lv := &LogViewer{}
		err := lv.followLogs(filepath.Join(t.TempDir(), "missing.log"))
		require.Error(t, err)
	})

	t.Run("config command exit paths", func(t *testing.T) {
		cases := []string{
			"config-invalid-set",
			"config-list-missing-package",
			"agent-unknown-command",
			"agent-query-missing-resource",
			"agent-run-missing-id",
			"proxy-invalid-json",
			"proxy-http-error-with-default-error",
			"vc-missing-file",
			"output-json-invalid",
			"output-pretty-invalid",
		}

		for _, name := range cases {
			t.Run(name, func(t *testing.T) {
				out, err := runCLITestHelper(t, name)
				require.Error(t, err)
				exitErr := &exec.ExitError{}
				require.ErrorAs(t, err, &exitErr)
				require.NotEmpty(t, out)
			})
		}
	})
}

func runCLITestHelper(t *testing.T, mode string) (string, error) {
	t.Helper()

	cmd := exec.Command(os.Args[0], "-test.run=TestCLIExitHelper", "--", mode)
	cmd.Env = append(os.Environ(), "GO_WANT_CLI_EXIT_HELPER=1")
	out, err := cmd.CombinedOutput()
	return string(out), err
}

func TestCLIExitHelper(t *testing.T) {
	if os.Getenv("GO_WANT_CLI_EXIT_HELPER") != "1" {
		return
	}

	mode := os.Args[len(os.Args)-1]
	switch mode {
	case "config-invalid-set":
		home := t.TempDir()
		prepareConfigFixture(t, home)
		os.Setenv("AGENTFIELD_HOME", home)
		configList, configSet, configUnset = false, "INVALID", ""
		runConfigCommand(nil, []string{"demo"})
	case "config-list-missing-package":
		home := t.TempDir()
		os.Setenv("AGENTFIELD_HOME", home)
		configList, configSet, configUnset = true, "", ""
		runConfigCommand(nil, []string{"missing"})
	case "agent-unknown-command":
		cmd := NewAgentCommand()
		cmd.SetArgs([]string{"unknown"})
		_ = cmd.Execute()
	case "agent-query-missing-resource":
		cmd := NewAgentCommand()
		cmd.SetArgs([]string{"query"})
		_ = cmd.Execute()
	case "agent-run-missing-id":
		cmd := NewAgentCommand()
		cmd.SetArgs([]string{"run"})
		_ = cmd.Execute()
	case "proxy-invalid-json":
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte("not-json"))
		}))
		defer server.Close()
		serverURL, outputFormat, requestTimeout = server.URL, "json", 1
		proxyToServer(http.MethodGet, "/bad", nil)
	case "proxy-http-error-with-default-error":
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"ok":false,"meta":{"upstream":"x"}}`))
		}))
		defer server.Close()
		serverURL, outputFormat, requestTimeout = server.URL, "json", 1
		proxyToServer(http.MethodGet, "/bad-request", nil)
	case "vc-missing-file":
		_ = verifyVC(filepath.Join(t.TempDir(), "missing.json"), VerifyOptions{OutputFormat: "json"})
	case "output-json-invalid":
		_ = outputJSON(VCVerificationResult{
			Valid:      false,
			Type:       "credential",
			FormatValid: false,
			Message:    "bad",
		})
	case "output-pretty-invalid":
		_ = outputPretty(VCVerificationResult{
			Valid:          false,
			Type:           "workflow",
			WorkflowID:     "wf-1",
			FormatValid:    true,
			SignatureValid: false,
			Message:        "failed",
			Error:          "broken",
			VerifiedAt:     "2026-01-02T03:04:05Z",
			Summary:        VerificationSummary{TotalComponents: 1, TotalDIDs: 1, TotalSignatures: 1},
		}, true)
	default:
		os.Exit(2)
	}
}

func prepareConfigFixture(t *testing.T, home string) {
	t.Helper()

	pkgDir := filepath.Join(home, "pkg")
	require.NoError(t, os.MkdirAll(pkgDir, 0o755))

	registry := &packages.InstallationRegistry{
		Installed: map[string]packages.InstalledPackage{
			"demo": {Name: "demo", Path: pkgDir},
		},
	}
	data, err := yaml.Marshal(registry)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(filepath.Join(home, "installed.yaml"), data, 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(pkgDir, "agentfield-package.yaml"), []byte(strings.TrimSpace(`
name: demo
user_environment:
  required:
    - name: API_KEY
      description: api key
`)), 0o644))
}

func TestAgentCommandExitOutputs(t *testing.T) {
	cases := []struct {
		name string
		want string
	}{
		{name: "config-invalid-set", want: "Invalid format. Use KEY=VALUE"},
		{name: "config-list-missing-package", want: "Failed to list configuration"},
		{name: "agent-unknown-command", want: `"code": "unknown_command"`},
		{name: "agent-query-missing-resource", want: `"code": "missing_required_flag"`},
		{name: "agent-run-missing-id", want: `"--id is required"`},
		{name: "proxy-invalid-json", want: `"code": "invalid_response"`},
		{name: "proxy-http-error-with-default-error", want: `"request failed with status 400"`},
		{name: "vc-missing-file", want: `"error": "Failed to read VC file:`},
		{name: "output-json-invalid", want: `"valid": false`},
		{name: "output-pretty-invalid", want: "AgentField VC Verification: ❌ INVALID"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			output, err := runCLITestHelper(t, tc.name)
			require.Error(t, err)
			require.Contains(t, output, tc.want)
		})
	}
}

func TestRunLogsCommandError(t *testing.T) {
	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)

	logsFollow = false
	logsTail = 3
	err := runLogsCommand(nil, []string{"missing"})
	require.ErrorContains(t, err, "failed to view logs")
}

func TestUtilityAndExecutionCommandCoverage(t *testing.T) {
	t.Run("agentfield home falls back to HOME and ensures subdirs", func(t *testing.T) {
		home := t.TempDir()
		t.Setenv("HOME", home)
		t.Setenv("AGENTFIELD_HOME", "")

		got := getAgentFieldHomeDir()
		require.Equal(t, filepath.Join(home, ".agentfield"), got)
		ensureSubdirs(got)
		for _, name := range []string{"packages", "logs", "config"} {
			info, err := os.Stat(filepath.Join(got, name))
			require.NoError(t, err)
			require.True(t, info.IsDir())
		}
	})

	t.Run("cancel and pause execution commands hit their run paths", func(t *testing.T) {
		tests := []struct {
			name     string
			newCmd   func() *cobra.Command
			path     string
			reason   string
			wantText string
		}{
			{name: "cancel", newCmd: newCancelExecutionCommand, path: "/api/v1/executions/ex-10/cancel", reason: "operator", wantText: "Execution ex-10 cancelled"},
			{name: "pause", newCmd: newPauseExecutionCommand, path: "/api/v1/executions/ex-11/pause", reason: "maintenance", wantText: "Execution ex-11 paused"},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					require.Equal(t, tc.path, r.URL.Path)
					var payload map[string]string
					require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
					require.Equal(t, tc.reason, payload["reason"])
					parts := strings.Split(strings.Trim(r.URL.Path, "/"), "/")
					executionID := parts[3]
					w.Header().Set("Content-Type", "application/json")
					_, _ = w.Write([]byte(`{"execution_id":"` + executionID + `","previous_status":"running"}`))
				}))
				defer server.Close()

				id := "ex-10"
				if tc.name == "pause" {
					id = "ex-11"
				}
				cmd := tc.newCmd()
				cmd.SetArgs([]string{id, "--server", server.URL, "--timeout", "1s", "--reason", tc.reason})
				output := captureOutput(t, func() {
					require.NoError(t, cmd.Execute())
				})
				require.Contains(t, output, tc.wantText)
			})
		}
	})
}

func TestAgentHelpOutputAndInteractiveConfig(t *testing.T) {
	oldServer := serverURL
	serverURL = "http://agent.test"
	defer func() { serverURL = oldServer }()

	output := captureOutput(t, func() {
		newAgentHelpCmd().Run(&cobra.Command{}, nil)
	})
	require.Contains(t, output, `"command": "af agent"`)
	require.Contains(t, output, `"server": "http://agent.test"`)

	t.Run("interactive config covers required optional and defaults", func(t *testing.T) {
		home := t.TempDir()
		pkgDir := filepath.Join(home, "pkg")
		require.NoError(t, os.MkdirAll(pkgDir, 0o755))
		prepareConfigFixture(t, home)
		require.NoError(t, os.WriteFile(filepath.Join(pkgDir, "agentfield-package.yaml"), []byte(strings.TrimSpace(`
name: demo
user_environment:
  required:
    - name: API_KEY
      description: api key
  optional:
    - name: REGION
      description: deployment region
      default: us-east-1
    - name: EMPTY
      description: empty optional
`)), 0o644))
		require.NoError(t, os.WriteFile(filepath.Join(pkgDir, ".env"), []byte("API_KEY=current\n"), 0o600))

		pcm := &PackageConfigManager{AgentFieldHome: home}
		withStreamingStdin(t, []string{"\n", "\n", "\n"}, func() {
			require.NoError(t, pcm.InteractiveConfig("demo"))
		})

		envVars, err := pcm.loadEnvFile(pkgDir)
		require.NoError(t, err)
		require.Equal(t, "current", envVars["API_KEY"])
		require.Equal(t, "us-east-1", envVars["REGION"])
		require.NotContains(t, envVars, "EMPTY")
	})

	t.Run("prompt retries after validation failure", func(t *testing.T) {
		pcm := &PackageConfigManager{}
		withStreamingStdin(t, []string{"bad\n", "GOOD\n"}, func() {
			output := captureOutput(t, func() {
				value, err := pcm.promptForVariable(packages.UserEnvironmentVar{
					Name:        "API_KEY",
					Description: "uppercase only",
					Validation:  "^[A-Z]+$",
				}, "")
				require.NoError(t, err)
				require.Equal(t, "GOOD", value)
			})
			require.Contains(t, output, "Invalid format. Please try again.")
		})
	})
}

func TestOutputAgentJSONDefaultAndReadBatchInputSpacing(t *testing.T) {
	oldFormat := outputFormat
	outputFormat = "  "
	defer func() {
		outputFormat = oldFormat
	}()

	output := captureOutput(t, func() {
		require.NoError(t, outputAgentJSON(map[string]int{"value": 1}))
	})
	require.Contains(t, output, "\n  ")

	file := filepath.Join(t.TempDir(), "ops.json")
	require.NoError(t, os.WriteFile(file, []byte(" \n\t[1]\n"), 0o644))
	data, err := readBatchInput("  " + file + "  ")
	require.NoError(t, err)
	require.Equal(t, "[1]", strings.TrimSpace(string(data)))
}

func TestVCFormattingAndSignatureErrors(t *testing.T) {
	t.Run("output pretty renders verbose sections", func(t *testing.T) {
		output := captureOutput(t, func() {
			require.NoError(t, outputPretty(VCVerificationResult{
				Valid:          true,
				Type:           "workflow",
				WorkflowID:     "wf-1",
				FormatValid:    true,
				SignatureValid: true,
				Message:        "ok",
				VerifiedAt:     "2026-01-02T03:04:05Z",
				Summary:        VerificationSummary{ValidComponents: 1, TotalComponents: 1, ResolvedDIDs: 1, TotalDIDs: 1, ValidSignatures: 1, TotalSignatures: 1},
				VerificationSteps: []VerificationStep{
					{Step: 1, Description: "read", Success: true, Details: "done"},
				},
				DIDResolutions: []DIDResolutionResult{
					{DID: "did:key:issuer", Method: "key", ResolvedFrom: "bundled", Success: true},
				},
				ComponentResults: []ComponentVerification{
					{VCID: "vc-1", ExecutionID: "exec-1", Status: "completed", Valid: true},
				},
			}, true))
		})
		require.Contains(t, output, "Verification Steps:")
		require.Contains(t, output, "DID Resolutions:")
		require.Contains(t, output, "Component Verification:")
	})

	t.Run("signature helpers return decoding errors", func(t *testing.T) {
		var workflowDoc types.WorkflowVCDocument
		workflowDoc.Context = []string{"https://www.w3.org/2018/credentials/v1"}
		workflowDoc.Type = []string{"VerifiableCredential"}
		workflowDoc.ID = "wf-vc"
		workflowDoc.Issuer = "did:key:issuer"
		workflowDoc.IssuanceDate = "2026-01-02T03:04:05Z"
		workflowDoc.Proof = types.VCProof{ProofValue: "%%%"}

		valid, err := verifyWorkflowVCSignature(workflowDoc, DIDResolutionInfo{
			PublicKeyJWK: map[string]interface{}{"x": "%%%"},
		})
		require.False(t, valid)
		require.Error(t, err)

		vcDoc := types.VCDocument{
			Context:      []string{"https://www.w3.org/2018/credentials/v1"},
			Type:         []string{"VerifiableCredential"},
			ID:           "vc-1",
			Issuer:       "did:key:issuer",
			IssuanceDate: "2026-01-02T03:04:05Z",
			Proof:        types.VCProof{ProofValue: "%%%"},
		}
		valid, err = verifyVCSignature(vcDoc, DIDResolutionInfo{
			PublicKeyJWK: map[string]interface{}{"x": "%%%"},
		})
		require.False(t, valid)
		require.Error(t, err)
	})
}

func TestTryParseEnhancedChainMergesDIDBundle(t *testing.T) {
	input := []byte(`{
		"workflow_id":"wf-1",
		"did_resolution_bundle":{
			"did:key:issuer":{
				"method":"key",
				"public_key_jwk":{"x":"abc"},
				"resolved_from":"bundle",
				"resolved_at":"2026-01-01T00:00:00Z"
			}
		}
	}`)

	chain, ok := tryParseEnhancedChain(input)
	require.True(t, ok)
	require.Equal(t, "wf-1", chain.WorkflowID)
	require.Contains(t, chain.DIDResolutionBundle, "did:key:issuer")
	require.Equal(t, "bundle", chain.DIDResolutionBundle["did:key:issuer"].ResolvedFrom)
}

func TestVCCommandCoverageHelpers(t *testing.T) {
	execVC, raw, resolution := signedExecutionVC(t, "did:key:issuer")

	t.Run("collect unique dids includes workflow issuer and skips duplicates", func(t *testing.T) {
		workflowDoc := types.WorkflowVCDocument{
			Issuer: "did:key:workflow",
		}
		workflowRaw, err := json.Marshal(workflowDoc)
		require.NoError(t, err)

		dids := collectUniqueDIDs(EnhancedVCChain{
			ExecutionVCs: []types.ExecutionVC{
				execVC,
				{VCDocument: raw},
				{VCDocument: []byte(`not-json`)},
			},
			WorkflowVC: types.WorkflowVC{VCDocument: workflowRaw},
		})
		require.Equal(t, []string{"did:key:issuer", "did:key:workflow"}, dids)
	})

	t.Run("vc verify command executes with pretty and json output", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "vc.json")
		payload, err := json.Marshal(EnhancedVCChain{
			WorkflowID:          "wf-1",
			GeneratedAt:         "2026-01-02T03:04:05Z",
			TotalExecutions:     1,
			CompletedExecutions: 1,
			WorkflowStatus:      "completed",
			ExecutionVCs:        []types.ExecutionVC{execVC},
			ComponentVCs:        []types.ExecutionVC{execVC},
			DIDResolutionBundle: map[string]DIDResolutionInfo{"did:key:issuer": resolution},
		})
		require.NoError(t, err)
		require.NoError(t, os.WriteFile(path, payload, 0o644))

		for _, args := range [][]string{
			{"verify", path, "--format", "pretty"},
			{"verify", path, "--format", "json"},
		} {
			cmd := NewVCCommand()
			cmd.SetArgs(args)
			output := captureOutput(t, func() {
				require.NoError(t, cmd.Execute())
			})
			require.NotEmpty(t, output)
		}
	})
}

func TestEnhancedWorkflowVerificationBranches(t *testing.T) {
	execVC, _, _ := signedExecutionVC(t, "did:key:issuer")

	t.Run("valid workflow verification succeeds", func(t *testing.T) {
		workflowVC, resolution := signedWorkflowVCForTest(t, "did:key:issuer", []string{"vc-1"})
		verifier := NewEnhancedVCVerifier(map[string]DIDResolutionInfo{
			"did:key:issuer": resolution,
		}, false)

		result := verifier.verifyWorkflowVC(workflowVC, []types.ExecutionVC{execVC})
		require.True(t, result.Valid)
		require.True(t, result.SignatureValid)
		require.True(t, result.ComponentConsistency)
		require.True(t, result.TimestampConsistency)
		require.True(t, result.StatusConsistency)
		require.True(t, result.ChainIntegrity)
		require.Empty(t, result.Issues)
	})

	t.Run("workflow signature invalid is reported", func(t *testing.T) {
		workflowVC, resolution := signedWorkflowVCForTest(t, "did:key:issuer", []string{"vc-1"})
		var doc types.WorkflowVCDocument
		require.NoError(t, json.Unmarshal(workflowVC.VCDocument, &doc))
		doc.Proof.ProofValue = "invalid-signature"
		raw, err := json.Marshal(doc)
		require.NoError(t, err)
		workflowVC.VCDocument = raw

		verifier := NewEnhancedVCVerifier(map[string]DIDResolutionInfo{
			"did:key:issuer": resolution,
		}, false)

		result := verifier.verifyWorkflowVC(workflowVC, []types.ExecutionVC{execVC})
		require.False(t, result.Valid)
		require.False(t, result.SignatureValid)
		require.NotEmpty(t, result.Issues)
		require.Equal(t, "workflow_signature_error", result.Issues[0].Type)
	})

	t.Run("validate vc structure covers required fields", func(t *testing.T) {
		verifier := NewEnhancedVCVerifier(nil, false)
		cases := []struct {
			name  string
			doc   types.VCDocument
			want  string
		}{
			{name: "missing context", doc: types.VCDocument{}, want: "missing @context"},
			{name: "missing type", doc: types.VCDocument{Context: []string{"ctx"}}, want: "missing type"},
			{name: "missing id", doc: types.VCDocument{Context: []string{"ctx"}, Type: []string{"VerifiableCredential"}}, want: "missing id"},
			{name: "missing issuer", doc: types.VCDocument{Context: []string{"ctx"}, Type: []string{"VerifiableCredential"}, ID: "vc-1"}, want: "missing issuer"},
			{name: "missing issuanceDate", doc: types.VCDocument{Context: []string{"ctx"}, Type: []string{"VerifiableCredential"}, ID: "vc-1", Issuer: "did:key:issuer"}, want: "missing issuanceDate"},
		}

		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				err := verifier.validateVCStructure(tc.doc)
				require.ErrorContains(t, err, tc.want)
			})
		}
	})
}

func TestAgentHTTPReadResponseError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hj, ok := w.(http.Hijacker)
		require.True(t, ok)
		conn, _, err := hj.Hijack()
		require.NoError(t, err)
		_, _ = conn.Write([]byte("HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhi"))
		_ = conn.Close()
	}))
	defer server.Close()

	oldServer := serverURL
	serverURL = server.URL
	defer func() { serverURL = oldServer }()

	_, status, err := agentHTTP(http.MethodGet, "/broken", nil)
	require.ErrorContains(t, err, "read response")
	require.Equal(t, http.StatusOK, status)
}

func signedWorkflowVCForTest(t *testing.T, issuer string, componentIDs []string) (types.WorkflowVC, DIDResolutionInfo) {
	t.Helper()

	publicKey, privateKey, err := ed25519.GenerateKey(rand.Reader)
	require.NoError(t, err)

	doc := types.WorkflowVCDocument{
		Context:      []string{"https://www.w3.org/2018/credentials/v1"},
		Type:         []string{"VerifiableCredential"},
		ID:           "workflow-vc-1",
		Issuer:       issuer,
		IssuanceDate: "2026-01-02T03:04:05Z",
	}
	canonical, err := json.Marshal(doc)
	require.NoError(t, err)
	doc.Proof = types.VCProof{
		ProofValue: base64.RawURLEncoding.EncodeToString(ed25519.Sign(privateKey, canonical)),
	}
	raw, err := json.Marshal(doc)
	require.NoError(t, err)

	return types.WorkflowVC{
		WorkflowID:   "wf-1",
		ComponentVCs: componentIDs,
		Status:       "completed",
		VCDocument:   raw,
	}, DIDResolutionInfo{
		DID:    issuer,
		Method: "key",
		PublicKeyJWK: map[string]interface{}{
			"x": base64.RawURLEncoding.EncodeToString(publicKey),
		},
	}
}

func TestPrepareConfigFixtureShape(t *testing.T) {
	home := t.TempDir()
	prepareConfigFixture(t, home)

	data, err := os.ReadFile(filepath.Join(home, "installed.yaml"))
	require.NoError(t, err)
	require.Contains(t, string(data), "demo")

	metadata, err := os.ReadFile(filepath.Join(home, "pkg", "agentfield-package.yaml"))
	require.NoError(t, err)
	require.Contains(t, string(metadata), "API_KEY")
}

func withStreamingStdin(t *testing.T, chunks []string, fn func()) {
	t.Helper()

	oldStdin := os.Stdin
	r, w, err := os.Pipe()
	require.NoError(t, err)
	os.Stdin = r
	defer func() {
		os.Stdin = oldStdin
		require.NoError(t, r.Close())
	}()

	done := make(chan struct{})
	go func() {
		defer close(done)
		for _, chunk := range chunks {
			_, _ = w.WriteString(chunk)
			time.Sleep(10 * time.Millisecond)
		}
		_ = w.Close()
	}()

	fn()
	<-done
}


func TestProxyErrorOutputIncludesHint(t *testing.T) {
	output, err := runCLITestHelper(t, "proxy-http-error-with-default-error")
	require.Error(t, err)
	require.Contains(t, output, `"hint": "Review command flags and payload shape, then retry."`)
	require.Contains(t, output, `"status_code": 400`)
}

func TestAgentCommandBatchInvalidJSON(t *testing.T) {
	file := filepath.Join(t.TempDir(), "batch.json")
	require.NoError(t, os.WriteFile(file, []byte("{"), 0o644))

	cmd := exec.Command(os.Args[0], "-test.run=TestCLIExitHelperBatch", "--", file)
	cmd.Env = append(os.Environ(), "GO_WANT_CLI_EXIT_HELPER=1")
	out, err := cmd.CombinedOutput()
	require.Error(t, err)
	require.Contains(t, string(out), `"code": "invalid_batch_json"`)
}

func TestCLIExitHelperBatch(t *testing.T) {
	if os.Getenv("GO_WANT_CLI_EXIT_HELPER") != "1" {
		return
	}
	if len(os.Args) < 2 || os.Args[len(os.Args)-2] != "--" {
		return
	}
	file := os.Args[len(os.Args)-1]
	if _, err := strconv.Atoi(filepath.Base(file)); err == nil {
		return
	}
	cmd := NewAgentCommand()
	cmd.SetArgs([]string{"batch", "--file", file})
	_ = cmd.Execute()
}
