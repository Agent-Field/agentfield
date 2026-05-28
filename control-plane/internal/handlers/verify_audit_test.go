package handlers

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestHandleVerifyAuditBundle_RejectsRemoteResolutionParams(t *testing.T) {
	gin.SetMode(gin.TestMode)

	router := gin.New()
	router.POST("/verify", HandleVerifyAuditBundle)

	req := httptest.NewRequest(
		http.MethodPost,
		"/verify?resolve_web=true&did_resolver=https://resolver.example",
		strings.NewReader(`{"workflow_id":"wf-1","workflow_vc":{"workflow_id":"wf-1"}}`),
	)
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "remote DID resolution is only supported in the CLI") {
		t.Fatalf("unexpected body: %s", rec.Body.String())
	}
}
