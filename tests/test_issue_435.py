"""Regression test for issue #435: invalid DID JWK must not disable signing silently."""

from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest


def test_issue_435(tmp_path: Path):
    """
    Validate that explicitly requested DID auth reports invalid key material.

    The Go SDK previously logged a warning from WithDIDAuth when JWK parsing
    failed, returned a usable client, and then sent requests without DID
    signature headers. This test runs a small external Go module against the
    local SDK so it only exercises the public client package API.
    """
    if shutil.which("go") is None:
        pytest.skip("Go toolchain is required for issue #435 regression test")

    repo_root = Path(__file__).resolve().parents[1]
    go_sdk = repo_root / "sdk" / "go"

    (tmp_path / "go.mod").write_text(
        textwrap.dedent(
            f"""
            module issue435

            go 1.21

            require github.com/Agent-Field/agentfield/sdk/go v0.0.0

            replace github.com/Agent-Field/agentfield/sdk/go => {go_sdk}
            """
        ).strip()
        + "\n"
    )
    (tmp_path / "issue_435_test.go").write_text(
        textwrap.dedent(
            r"""
            package issue435

            import (
                "context"
                "crypto/ed25519"
                "crypto/rand"
                "encoding/base64"
                "fmt"
                "net/http"
                "net/http/httptest"
                "testing"
                "time"

                afclient "github.com/Agent-Field/agentfield/sdk/go/client"
                "github.com/Agent-Field/agentfield/sdk/go/types"
            )

            func validJWK(t *testing.T) string {
                t.Helper()

                pub, priv, err := ed25519.GenerateKey(rand.Reader)
                if err != nil {
                    t.Fatalf("generate key: %v", err)
                }

                return fmt.Sprintf(
                    `{"kty":"OKP","crv":"Ed25519","d":%q,"x":%q}`,
                    base64.RawURLEncoding.EncodeToString(priv.Seed()),
                    base64.RawURLEncoding.EncodeToString(pub),
                )
            }

            func registrationPayload() types.NodeRegistrationRequest {
                return types.NodeRegistrationRequest{
                    ID:        "node-1",
                    TeamID:    "team-1",
                    BaseURL:   "https://agent.example",
                    Version:   "1.0.0",
                    Reasoners: []types.ReasonerDefinition{},
                    Skills:    []types.SkillDefinition{},
                    CommunicationConfig: types.CommunicationConfig{
                        Protocols: []string{"http"},
                    },
                    HealthStatus:  "healthy",
                    LastHeartbeat: time.Now(),
                    RegisteredAt:  time.Now(),
                }
            }

            func TestIssue435(t *testing.T) {
                badJWK := `{"kty":"OKP","crv":"Ed25519","d":"not-valid-base64url"}`

                badClient, err := afclient.New(
                    "https://control-plane.example",
                    afclient.WithDIDAuth("did:web:example.com:agents:bad", badJWK),
                )
                if err == nil && badClient == nil {
                    t.Fatalf("New(WithDIDAuth(badJWK)) returned nil client and nil error")
                }
                if badClient != nil && badClient.DIDAuthConfigured() {
                    t.Fatalf("client with invalid DID key reports DID auth configured")
                }
                if err == nil {
                    sawUnsignedRequest := false
                    server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
                        if r.Header.Get(afclient.HeaderCallerDID) == "" ||
                            r.Header.Get(afclient.HeaderDIDSignature) == "" ||
                            r.Header.Get(afclient.HeaderDIDTimestamp) == "" ||
                            r.Header.Get(afclient.HeaderDIDNonce) == "" {
                            sawUnsignedRequest = true
                        }
                        w.WriteHeader(http.StatusOK)
                    }))
                    defer server.Close()

                    badClient, err = afclient.New(
                        server.URL,
                        afclient.WithDIDAuth("did:web:example.com:agents:bad", badJWK),
                    )
                    if err == nil {
                        _, err = badClient.RegisterNode(context.Background(), registrationPayload())
                    }
                    if err == nil {
                        t.Fatalf("WithDIDAuth(badJWK) produced no construction or signed-call error")
                    }
                    if sawUnsignedRequest {
                        t.Fatalf("client with failed DID auth initialization sent an unsigned request")
                    }
                }

                goodClient, err := afclient.New(
                    "https://control-plane.example",
                    afclient.WithDIDAuth("did:web:example.com:agents:good", validJWK(t)),
                )
                if err != nil {
                    t.Fatalf("New(WithDIDAuth(goodJWK)) error = %v, want nil", err)
                }
                if goodClient == nil || !goodClient.DIDAuthConfigured() {
                    t.Fatalf("client with valid DID key did not configure DID auth")
                }
            }
            """
        ).strip()
        + "\n"
    )

    result = subprocess.run(
        ["go", "test", "./..."],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
