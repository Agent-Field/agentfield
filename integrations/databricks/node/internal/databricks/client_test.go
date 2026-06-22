package databricks

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestValidateReadOnlySQL(t *testing.T) {
	for _, sql := range []string{"SELECT 1", "WITH x AS (SELECT 1) SELECT * FROM x", "DESCRIBE TABLE t"} {
		if err := ValidateReadOnlySQL(sql); err != nil {
			t.Fatalf("ValidateReadOnlySQL(%q) = %v", sql, err)
		}
	}
	for _, sql := range []string{"DELETE FROM t", "SELECT 1; SELECT 2", ""} {
		if err := ValidateReadOnlySQL(sql); err == nil {
			t.Fatalf("ValidateReadOnlySQL(%q) expected error", sql)
		}
	}
}

func TestQueryReadOnlyCallsStatementAPIAndTruncates(t *testing.T) {
	var gotAuth string
	var gotStatement statementRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		if r.URL.Path != "/api/2.0/sql/statements" {
			t.Fatalf("path = %s", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&gotStatement); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"statement_id":"stmt-1",
			"status":{"state":"SUCCEEDED"},
			"manifest":{"schema":{"columns":[{"name":"id"},{"name":"name"}]}},
			"result":{"data_array":[[1,"a"],[2,"b"]]}
		}`))
	}))
	defer server.Close()

	client, err := NewClient(Config{WorkspaceURL: server.URL, Token: "pat", WarehouseID: "wh", Catalog: "main", Schema: "default", MaxRows: 1, HTTPClient: server.Client()})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	result, err := client.QueryReadOnly(context.Background(), QueryRequest{SQL: "SELECT id, name FROM users"})
	if err != nil {
		t.Fatalf("QueryReadOnly: %v", err)
	}
	if gotAuth != "Bearer pat" {
		t.Fatalf("auth = %q", gotAuth)
	}
	if gotStatement.WarehouseID != "wh" || gotStatement.Catalog != "main" || gotStatement.Schema != "default" {
		t.Fatalf("defaults not sent: %+v", gotStatement)
	}
	if !strings.Contains(gotStatement.Statement, "LIMIT 2") || gotStatement.RowLimit != 2 {
		t.Fatalf("statement was not bounded: %+v", gotStatement)
	}
	if !result.Truncated || result.RowCount != 1 || result.StatementID != "stmt-1" {
		t.Fatalf("bad result = %+v", result)
	}
}

func TestQueryReadOnlyPollsStatement(t *testing.T) {
	var getSeen bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.Method {
		case http.MethodPost:
			_, _ = w.Write([]byte(`{"statement_id":"async-1","status":{"state":"PENDING"}}`))
		case http.MethodGet:
			getSeen = true
			if r.URL.Path != "/api/2.0/sql/statements/async-1" {
				t.Fatalf("poll path = %s", r.URL.Path)
			}
			_, _ = w.Write([]byte(`{
				"statement_id":"async-1",
				"status":{"state":"SUCCEEDED"},
				"manifest":{"schema":{"columns":[{"name":"ok"}]}},
				"result":{"data_array":[[true]]}
			}`))
		default:
			t.Fatalf("unexpected method %s", r.Method)
		}
	}))
	defer server.Close()

	client, err := NewClient(Config{WorkspaceURL: server.URL, Token: "pat", WarehouseID: "wh", TimeoutSeconds: 2, HTTPClient: server.Client()})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	result, err := client.QueryReadOnly(ctx, QueryRequest{SQL: "SELECT true AS ok"})
	if err != nil {
		t.Fatalf("QueryReadOnly: %v", err)
	}
	if !getSeen || result.StatementID != "async-1" {
		t.Fatalf("getSeen=%v result=%+v", getSeen, result)
	}
}

func TestAIQueryUsesDatabricksFunction(t *testing.T) {
	var got statementRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"statement_id":"ai-1",
			"status":{"state":"SUCCEEDED"},
			"manifest":{"schema":{"columns":[{"name":"response"}]}},
			"result":{"data_array":[["answer"]]}
		}`))
	}))
	defer server.Close()
	client, err := NewClient(Config{WorkspaceURL: server.URL, Token: "pat", WarehouseID: "wh", AIEndpoint: "databricks-meta-llama-3-3-70b-instruct", HTTPClient: server.Client()})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	result, err := client.AIQuery(context.Background(), "", "summarize", "", false)
	if err != nil {
		t.Fatalf("AIQuery: %v", err)
	}
	if !strings.Contains(got.Statement, "ai_query") || len(got.Parameters) != 2 || got.Parameters[0].Value != "databricks-meta-llama-3-3-70b-instruct" {
		t.Fatalf("bad ai_query request: %+v", got)
	}
	if result.Rows[0]["response"] != "answer" {
		t.Fatalf("bad ai result: %+v", result)
	}
}

func TestInvokeServingEndpoint(t *testing.T) {
	var gotPath string
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Fatalf("decode body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"predictions":[1]}`))
	}))
	defer server.Close()
	client, err := NewClient(Config{WorkspaceURL: server.URL, Token: "pat", HTTPClient: server.Client()})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	raw, err := client.InvokeServingEndpoint(context.Background(), "fraud-model", map[string]any{"dataframe_records": []any{map[string]any{"x": 1}}})
	if err != nil {
		t.Fatalf("InvokeServingEndpoint: %v", err)
	}
	if gotPath != "/serving-endpoints/fraud-model/invocations" || gotBody["dataframe_records"] == nil {
		t.Fatalf("bad serving request path=%s body=%v", gotPath, gotBody)
	}
	if raw["predictions"] == nil {
		t.Fatalf("bad serving response: %v", raw)
	}
	if _, err := client.InvokeServingEndpoint(context.Background(), "../bad", map[string]any{}); err == nil {
		t.Fatal("expected invalid endpoint error")
	}
}

func TestSearchColumnsRequiresCatalog(t *testing.T) {
	client, err := NewClient(Config{WorkspaceURL: "https://dbc.example.cloud.databricks.com", Token: "pat", WarehouseID: "wh"})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	if _, err := client.SearchColumns(context.Background(), "customer", "", "", 10); err == nil {
		t.Fatal("expected catalog error")
	}
}
