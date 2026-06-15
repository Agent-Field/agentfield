package snowflake

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
	for _, sql := range []string{"SELECT 1", "WITH x AS (SELECT 1) SELECT * FROM x", "DESCRIBE TABLE T"} {
		if err := ValidateReadOnlySQL(sql); err != nil {
			t.Fatalf("ValidateReadOnlySQL(%q) = %v", sql, err)
		}
	}
	for _, sql := range []string{"DELETE FROM T", "SELECT 1; SELECT 2", ""} {
		if err := ValidateReadOnlySQL(sql); err == nil {
			t.Fatalf("ValidateReadOnlySQL(%q) expected error", sql)
		}
	}
}

func TestQueryReadOnlyCallsSQLAPIAndTruncates(t *testing.T) {
	var gotAuth string
	var gotTokenType string
	var gotStatement string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		gotTokenType = r.Header.Get("X-Snowflake-Authorization-Token-Type")
		var req statementRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		gotStatement = req.Statement
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"statementHandle":"01b-456",
			"resultSetMetaData":{"rowType":[{"name":"ID"},{"name":"NAME"}]},
			"data":[[1,"a"],[2,"b"]]
		}`))
	}))
	defer server.Close()

	client, err := NewClient(Config{AccountURL: server.URL, Token: "pat", MaxRows: 1, HTTPClient: server.Client()})
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
	if gotTokenType != "PROGRAMMATIC_ACCESS_TOKEN" {
		t.Fatalf("token type = %q", gotTokenType)
	}
	if !strings.Contains(gotStatement, "LIMIT 2") {
		t.Fatalf("statement was not bounded: %s", gotStatement)
	}
	if !result.Truncated || result.RowCount != 1 {
		t.Fatalf("truncation result = %+v", result)
	}
	if result.QueryID != "01b-456" || result.Columns[0] != "ID" {
		t.Fatalf("bad result = %+v", result)
	}
}

func TestQueryReadOnlyPollsAcceptedStatement(t *testing.T) {
	var getSeen bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.Method {
		case http.MethodPost:
			w.WriteHeader(http.StatusAccepted)
			_, _ = w.Write([]byte(`{"statementHandle":"async-2","statementStatusUrl":"/api/v2/statements/async-2"}`))
		case http.MethodGet:
			getSeen = true
			if r.Header.Get("X-Snowflake-Authorization-Token-Type") != "PROGRAMMATIC_ACCESS_TOKEN" {
				t.Fatalf("missing PAT token type on poll")
			}
			_, _ = w.Write([]byte(`{
				"statementHandle":"async-2",
				"resultSetMetaData":{"rowType":[{"name":"OK"}]},
				"data":[[true]]
			}`))
		default:
			t.Fatalf("unexpected method %s", r.Method)
		}
	}))
	defer server.Close()

	client, err := NewClient(Config{AccountURL: server.URL, Token: "pat", TimeoutSeconds: 2, HTTPClient: server.Client()})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	result, err := client.QueryReadOnly(ctx, QueryRequest{SQL: "SELECT true AS ok"})
	if err != nil {
		t.Fatalf("QueryReadOnly: %v", err)
	}
	if !getSeen || result.QueryID != "async-2" {
		t.Fatalf("getSeen=%v result=%+v", getSeen, result)
	}
}

func TestCortexChatCompleteCallsRESTAPI(t *testing.T) {
	var gotPath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		if r.Header.Get("X-Snowflake-Authorization-Token-Type") != "PROGRAMMATIC_ACCESS_TOKEN" {
			t.Fatalf("missing PAT token type")
		}
		w.Header().Set("X-Snowflake-Request-Id", "req-1")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"hello"}}],"usage":{"total_tokens":3}}`))
	}))
	defer server.Close()

	client, err := NewClient(Config{AccountURL: server.URL, Token: "pat", HTTPClient: server.Client()})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	text, requestID, raw, err := client.CortexChatComplete(context.Background(), "claude-sonnet-4-5", "sys", "hi", 10, 0)
	if err != nil {
		t.Fatalf("CortexChatComplete: %v", err)
	}
	if gotPath != "/api/v2/cortex/v1/chat/completions" || text != "hello" || requestID != "req-1" || raw["usage"] == nil {
		t.Fatalf("bad REST response path=%s text=%q requestID=%q raw=%v", gotPath, text, requestID, raw)
	}
}

func TestSearchColumnsRequiresDatabase(t *testing.T) {
	client, err := NewClient(Config{AccountURL: "https://acct", Token: "pat"})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	if _, err := client.SearchColumns(context.Background(), "customer", "", "", 10); err == nil {
		t.Fatal("expected database error")
	}
}
