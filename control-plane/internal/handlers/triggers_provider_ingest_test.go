package handlers

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"

	_ "github.com/Agent-Field/agentfield/control-plane/internal/sources/all"
)

func TestIngestProviderWebhookFixtures_LinearAndSentry(t *testing.T) {
	provider, _, ctx := setupAPIContractTestEnv(t)
	h := NewTriggerHandlers(provider, nil, nil)
	r := triggerCoverageRouter(h)

	tests := []struct {
		name               string
		sourceName         string
		triggerID          string
		secretEnv          string
		secret             string
		config             json.RawMessage
		body               []byte
		headers            map[string]string
		expectedType       string
		expectedIdempotent string
		normalizedContains []string
	}{
		{
			name:       "linear issue create",
			sourceName: "linear",
			triggerID:  "linear-ingest-fixture",
			secretEnv:  "LINEAR_FIXTURE_SECRET",
			secret:     "linear-fixture-secret",
			config:     json.RawMessage(`{"tolerance_seconds":60}`),
			body: []byte(fmt.Sprintf(
				`{"action":"create","type":"Issue","createdAt":"2026-06-15T12:00:00Z","webhookTimestamp":%d,"webhookId":"linear-hook-1","data":{"id":"issue-1","identifier":"AF-1"}}`,
				time.Now().UnixMilli(),
			)),
			headers: map[string]string{
				"Linear-Delivery": "linear-delivery-1",
				"Linear-Event":    "Issue",
			},
			expectedType:       "issue.create",
			expectedIdempotent: "linear-delivery-1",
			normalizedContains: []string{
				`"delivery":"linear-delivery-1"`,
				`"identifier":"AF-1"`,
			},
		},
		{
			name:       "sentry issue created",
			sourceName: "sentry",
			triggerID:  "sentry-ingest-fixture",
			secretEnv:  "SENTRY_FIXTURE_SECRET",
			secret:     "sentry-fixture-secret",
			config:     json.RawMessage(`{}`),
			body:       []byte(`{"action":"created","installation":{"uuid":"inst-1"},"data":{"issue":{"id":"123","shortId":"ORG-1"}},"actor":{"type":"user","name":"Ada"}}`),
			headers: map[string]string{
				"Request-ID":             "sentry-request-1",
				"Sentry-Hook-Resource":   "issue",
				"Sentry-Hook-Timestamp":  time.Now().UTC().Format(time.RFC3339),
				"Sentry-Hook-Signature":  "",
				"X-Unused-Provider-Test": "kept",
			},
			expectedType:       "issue.created",
			expectedIdempotent: "sentry-request-1",
			normalizedContains: []string{
				`"request_id":"sentry-request-1"`,
				`"shortId":"ORG-1"`,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv(tt.secretEnv, tt.secret)
			trig := &types.Trigger{
				ID:             tt.triggerID,
				SourceName:     tt.sourceName,
				Config:         tt.config,
				SecretEnvVar:   tt.secretEnv,
				TargetNodeID:   "provider-target",
				TargetReasoner: "handle_provider_event",
				ManagedBy:      types.ManagedByUI,
				Enabled:        true,
				CreatedAt:      time.Now().UTC(),
				UpdatedAt:      time.Now().UTC(),
			}
			require.NoError(t, provider.CreateTrigger(ctx, trig))

			req := httptest.NewRequest(http.MethodPost, "/sources/"+tt.triggerID, strings.NewReader(string(tt.body)))
			req.Header.Set("Content-Type", "application/json")
			for key, value := range tt.headers {
				if value != "" {
					req.Header.Set(key, value)
				}
			}
			switch tt.sourceName {
			case "linear":
				req.Header.Set("Linear-Signature", signProviderFixture(tt.body, tt.secret))
			case "sentry":
				req.Header.Set("Sentry-Hook-Signature", signProviderFixture(tt.body, tt.secret))
			}

			w := httptest.NewRecorder()
			r.ServeHTTP(w, req)
			require.Equalf(t, http.StatusOK, w.Code, "body=%s", w.Body.String())

			var resp struct {
				Status   string `json:"status"`
				Received int    `json:"received"`
			}
			require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
			require.Equal(t, "ok", resp.Status)
			require.Equal(t, 1, resp.Received)

			events, err := provider.ListInboundEvents(ctx, tt.triggerID, 10)
			require.NoError(t, err)
			require.Len(t, events, 1)
			require.Equal(t, tt.sourceName, events[0].SourceName)
			require.Equal(t, tt.expectedType, events[0].EventType)
			require.Equal(t, tt.expectedIdempotent, events[0].IdempotencyKey)
			for _, want := range tt.normalizedContains {
				require.Contains(t, string(events[0].NormalizedPayload), want)
			}
		})
	}
}

func signProviderFixture(body []byte, secret string) string {
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write(body)
	return hex.EncodeToString(mac.Sum(nil))
}
