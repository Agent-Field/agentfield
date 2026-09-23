package agent

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestVerification_EvaluateConstraintsOperators(t *testing.T) {
	for _, tc := range []struct {
		name     string
		operator string
		input    float64
		want     bool
	}{
		{name: "less_equal/below", operator: "<=", input: 2, want: true},
		{name: "less_equal/equal", operator: "<=", input: 3, want: true},
		{name: "less_equal/above", operator: "<=", input: 4, want: false},
		{name: "greater_equal/below", operator: ">=", input: 2, want: false},
		{name: "greater_equal/equal", operator: ">=", input: 3, want: true},
		{name: "greater_equal/above", operator: ">=", input: 4, want: true},
		{name: "less/below", operator: "<", input: 2, want: true},
		{name: "less/equal", operator: "<", input: 3, want: false},
		{name: "less/above", operator: "<", input: 4, want: false},
		{name: "greater/below", operator: ">", input: 2, want: false},
		{name: "greater/equal", operator: ">", input: 3, want: false},
		{name: "greater/above", operator: ">", input: 4, want: true},
		{name: "equal/below", operator: "==", input: 2, want: false},
		{name: "equal/equal", operator: "==", input: 3, want: true},
		{name: "equal/above", operator: "==", input: 4, want: false},
		{name: "equal/within_tolerance", operator: "==", input: 3 + 5e-10, want: true},
		{name: "equal/outside_tolerance", operator: "==", input: 3 + 2e-9, want: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			constraints := map[string]ConstraintEntry{
				"value": {Operator: tc.operator, Value: 3},
			}
			got := evaluateConstraints(constraints, "agent.read", map[string]any{"value": tc.input})
			assert.Equal(t, tc.want, got)
		})
	}
}

func TestVerification_EvaluateConstraintsFailClosed(t *testing.T) {
	for _, tc := range []struct {
		name  string
		input map[string]any
	}{
		{name: "nil_input", input: nil},
		{name: "empty_input", input: map[string]any{}},
		{name: "missing_parameter", input: map[string]any{"other": 2}},
		{name: "nonnumeric_string", input: map[string]any{"value": "not-a-number"}},
		{name: "unsupported_type", input: map[string]any{"value": true}},
		{name: "nil_value", input: map[string]any{"value": nil}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			constraints := map[string]ConstraintEntry{
				"value": {Operator: "<=", Value: 3},
			}
			assert.False(t, evaluateConstraints(constraints, "agent.read", tc.input))
		})
	}
}

func TestLocalVerifier_FetchPoliciesInvalidURL(t *testing.T) {
	v := NewLocalVerifier("http://example.invalid/\n", 0, "")

	policies, err := v.fetchPolicies(&http.Client{})
	require.Error(t, err)
	assert.Nil(t, policies)

	// A parse error distinguishes NewRequest failure from a transport error.
	var urlErr *url.Error
	require.ErrorAs(t, err, &urlErr)
	assert.Equal(t, "parse", urlErr.Op)
	assert.Equal(t, v.agentFieldURL+"/api/v1/policies", urlErr.URL)
	assert.Contains(t, urlErr.Err.Error(), "invalid control character in URL")
}

func TestLocalVerifier_DoRequestAPIKeyHeader(t *testing.T) {
	for _, tc := range []struct {
		name        string
		apiKey      string
		wantPresent bool
	}{
		{name: "present", apiKey: "test-api-key", wantPresent: true},
		{name: "absent", apiKey: "", wantPresent: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				assert.Equal(t, http.MethodGet, r.Method)
				assert.Equal(t, "/api/v1/policies", r.URL.Path)
				// Header.Get alone cannot distinguish omission from an empty header.
				_, present := r.Header[http.CanonicalHeaderKey("X-API-Key")]
				assert.Equal(t, tc.wantPresent, present)
				assert.Equal(t, tc.apiKey, r.Header.Get("X-API-Key"))
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"policies":[{"name":"test-policy"}]}`))
			}))
			defer server.Close()

			v := NewLocalVerifier(server.URL, 0, tc.apiKey)
			policies, err := v.fetchPolicies(server.Client())
			require.NoError(t, err)
			assert.Equal(t, []PolicyEntry{{Name: "test-policy"}}, policies)
		})
	}
}
