package embedding

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// makeEmbedding returns a deterministic float32 slice of the given dimension so
// tests can assert the parsed vector matches what the (fake) API returned.
func makeEmbedding(dim int, seed float32) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = seed + float32(i)
	}
	return v
}

// TestOpenAIEmbedder_Embed_HappyPath stands up a real httptest server that
// returns a canned OpenAI embeddings response. It asserts:
//   - the request hits the configured endpoint with the right method, model and
//     input shape, and an Authorization bearer header;
//   - the response is parsed into one vector per input, in input order;
//   - each parsed vector matches the bytes the server returned.
func TestOpenAIEmbedder_Embed_HappyPath(t *testing.T) {
	wantModel := "text-embedding-3-small"
	inputs := []string{"alpha", "beta"}

	var gotReq openAIEmbedRequest
	var gotAuth, gotContentType, gotMethod string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod = r.Method
		gotAuth = r.Header.Get("Authorization")
		gotContentType = r.Header.Get("Content-Type")
		if err := json.NewDecoder(r.Body).Decode(&gotReq); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		// Return the data deliberately OUT of order to prove the embedder sorts
		// by index back into input order.
		resp := openAIEmbedResponse{}
		resp.Data = append(resp.Data, struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		}{Index: 1, Embedding: makeEmbedding(Dimensions, 1000)})
		resp.Data = append(resp.Data, struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		}{Index: 0, Embedding: makeEmbedding(Dimensions, 0)})
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	e := NewOpenAIEmbedder("sk-test-key", WithEndpoint(srv.URL), WithHTTPClient(srv.Client()))

	got, err := e.Embed(context.Background(), inputs)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	// Request shape.
	if gotMethod != http.MethodPost {
		t.Errorf("method = %q, want POST", gotMethod)
	}
	if gotAuth != "Bearer sk-test-key" {
		t.Errorf("Authorization = %q, want %q", gotAuth, "Bearer sk-test-key")
	}
	if gotContentType != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", gotContentType)
	}
	if gotReq.Model != wantModel {
		t.Errorf("model = %q, want %q", gotReq.Model, wantModel)
	}
	if len(gotReq.Input) != len(inputs) || gotReq.Input[0] != "alpha" || gotReq.Input[1] != "beta" {
		t.Errorf("input = %v, want %v", gotReq.Input, inputs)
	}

	// Response parsing: one vector per input, sorted back into input order.
	if len(got) != len(inputs) {
		t.Fatalf("got %d vectors, want %d", len(got), len(inputs))
	}
	for i, v := range got {
		if len(v) != Dimensions {
			t.Fatalf("vector %d dim = %d, want %d", i, len(v), Dimensions)
		}
	}
	// After sorting by index, vector 0 must be the seed=0 series, vector 1 the
	// seed=1000 series.
	if got[0][0] != 0 || got[1][0] != 1000 {
		t.Errorf("vectors not reordered by index: got[0][0]=%v got[1][0]=%v", got[0][0], got[1][0])
	}
}

// TestOpenAIEmbedder_Embed_EmptyInput returns early with no HTTP call.
func TestOpenAIEmbedder_Embed_EmptyInput(t *testing.T) {
	called := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	}))
	defer srv.Close()

	e := NewOpenAIEmbedder("sk-test", WithEndpoint(srv.URL), WithHTTPClient(srv.Client()))
	got, err := e.Embed(context.Background(), nil)
	if err != nil {
		t.Fatalf("Embed(empty): %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("got %d vectors, want 0", len(got))
	}
	if called {
		t.Fatal("Embed(empty) must not perform an HTTP request")
	}
}

// TestOpenAIEmbedder_Embed_MissingAPIKey fails fast without an HTTP call.
func TestOpenAIEmbedder_Embed_MissingAPIKey(t *testing.T) {
	e := NewOpenAIEmbedder("") // no key
	_, err := e.Embed(context.Background(), []string{"x"})
	if err == nil {
		t.Fatal("expected error when API key is empty")
	}
	if !strings.Contains(err.Error(), "API key") {
		t.Errorf("error = %q, want it to mention the missing API key", err.Error())
	}
}

// TestOpenAIEmbedder_Embed_Non200WithError surfaces the API's error message on a
// non-200 status.
func TestOpenAIEmbedder_Embed_Non200WithError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":{"message":"invalid api key","type":"auth"}}`))
	}))
	defer srv.Close()

	e := NewOpenAIEmbedder("sk-bad", WithEndpoint(srv.URL), WithHTTPClient(srv.Client()))
	_, err := e.Embed(context.Background(), []string{"x"})
	if err == nil {
		t.Fatal("expected error on non-200 response")
	}
	if !strings.Contains(err.Error(), "invalid api key") {
		t.Errorf("error = %q, want it to surface the API error message", err.Error())
	}
}

// TestOpenAIEmbedder_Embed_Non200NoBody errors with a status-only message when
// the non-200 response carries no parseable error object.
func TestOpenAIEmbedder_Embed_Non200NoBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	e := NewOpenAIEmbedder("sk-test", WithEndpoint(srv.URL), WithHTTPClient(srv.Client()))
	_, err := e.Embed(context.Background(), []string{"x"})
	if err == nil {
		t.Fatal("expected error on 500 with no error body")
	}
	if !strings.Contains(err.Error(), "500") {
		t.Errorf("error = %q, want it to mention status 500", err.Error())
	}
}

// TestOpenAIEmbedder_Embed_MalformedBody errors when the body is not valid JSON.
func TestOpenAIEmbedder_Embed_MalformedBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`not json at all`))
	}))
	defer srv.Close()

	e := NewOpenAIEmbedder("sk-test", WithEndpoint(srv.URL), WithHTTPClient(srv.Client()))
	_, err := e.Embed(context.Background(), []string{"x"})
	if err == nil {
		t.Fatal("expected decode error on malformed body")
	}
	if !strings.Contains(err.Error(), "decode") {
		t.Errorf("error = %q, want a decode error", err.Error())
	}
}

// TestOpenAIEmbedder_Embed_CountMismatch errors when the API returns a different
// number of vectors than inputs.
func TestOpenAIEmbedder_Embed_CountMismatch(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := openAIEmbedResponse{}
		resp.Data = append(resp.Data, struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		}{Index: 0, Embedding: makeEmbedding(Dimensions, 0)})
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	e := NewOpenAIEmbedder("sk-test", WithEndpoint(srv.URL), WithHTTPClient(srv.Client()))
	// Two inputs, one vector returned.
	_, err := e.Embed(context.Background(), []string{"a", "b"})
	if err == nil {
		t.Fatal("expected error on vector-count mismatch")
	}
	if !strings.Contains(err.Error(), "expected 2 vectors") {
		t.Errorf("error = %q, want a count-mismatch error", err.Error())
	}
}

// TestOpenAIEmbedder_Embed_DimensionMismatch errors when a returned vector has
// the wrong dimension for the pinned shared index.
func TestOpenAIEmbedder_Embed_DimensionMismatch(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := openAIEmbedResponse{}
		// Wrong dimension (Dimensions-1).
		resp.Data = append(resp.Data, struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		}{Index: 0, Embedding: makeEmbedding(Dimensions-1, 0)})
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	e := NewOpenAIEmbedder("sk-test", WithEndpoint(srv.URL), WithHTTPClient(srv.Client()))
	_, err := e.Embed(context.Background(), []string{"x"})
	if err == nil {
		t.Fatal("expected dimension-mismatch error")
	}
	if !strings.Contains(err.Error(), "dimension") {
		t.Errorf("error = %q, want a dimension-mismatch error", err.Error())
	}
}

// TestOpenAIEmbedder_Embed_TransportError errors when the HTTP request itself
// fails (server closed / unreachable endpoint).
func TestOpenAIEmbedder_Embed_TransportError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	url := srv.URL
	client := srv.Client()
	srv.Close() // close before the request so Do() fails at the transport layer

	e := NewOpenAIEmbedder("sk-test", WithEndpoint(url), WithHTTPClient(client))
	_, err := e.Embed(context.Background(), []string{"x"})
	if err == nil {
		t.Fatal("expected transport error against a closed server")
	}
	if !strings.Contains(err.Error(), "request failed") {
		t.Errorf("error = %q, want a transport-level failure", err.Error())
	}
}

// TestWithModelAndOptions covers the option setters: WithModel overrides the
// model and is sent in the request; empty option values are no-ops.
func TestOpenAIEmbedder_WithModelOption(t *testing.T) {
	var gotModel string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body openAIEmbedRequest
		_ = json.NewDecoder(r.Body).Decode(&body)
		gotModel = body.Model
		resp := openAIEmbedResponse{}
		resp.Data = append(resp.Data, struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		}{Index: 0, Embedding: makeEmbedding(Dimensions, 0)})
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	e := NewOpenAIEmbedder("sk-test",
		WithModel("custom-model"),
		WithModel(""),       // no-op: must not clear the model
		WithEndpoint(""),    // no-op: must keep the real endpoint
		WithEndpoint(srv.URL),
		WithHTTPClient(nil), // no-op: must keep a usable client
		WithHTTPClient(srv.Client()),
	)
	if _, err := e.Embed(context.Background(), []string{"x"}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if gotModel != "custom-model" {
		t.Errorf("model = %q, want custom-model", gotModel)
	}
}
