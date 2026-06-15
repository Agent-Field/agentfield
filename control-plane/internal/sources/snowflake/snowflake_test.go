package snowflake

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/sources"
)

func TestMetadataAndSchema(t *testing.T) {
	s := &source{}
	if s.Name() != "snowflake" {
		t.Fatalf("Name() = %q, want snowflake", s.Name())
	}
	if s.Kind() != sources.KindLoop {
		t.Fatalf("Kind() = %v, want loop", s.Kind())
	}
	if !s.SecretRequired() {
		t.Fatal("snowflake should require a secret")
	}
	var schema map[string]any
	if err := json.Unmarshal(s.ConfigSchema(), &schema); err != nil {
		t.Fatalf("schema is not valid JSON: %v", err)
	}
}

func TestValidateEventTableConfig(t *testing.T) {
	valid := []byte(`{
		"account_url":"https://acct.snowflakecomputing.com",
		"database":"OBSERVABILITY",
		"schema":"AGENTFIELD",
		"table":"AGENTFIELD_EVENTS",
		"interval_seconds":5
	}`)
	if err := (&source{}).Validate(valid); err != nil {
		t.Fatalf("Validate(valid) = %v", err)
	}

	for _, raw := range []string{
		`{"account_url":"https://acct.snowflakecomputing.com","database":"DB","schema":"S","table":"bad-name"}`,
		`{"account_url":"https://acct.snowflakecomputing.com","mode":"unknown"}`,
		`{"account_url":"https://acct.snowflakecomputing.com","database":"DB","schema":"S","table":"T","interval_seconds":1}`,
		`{"account_url":"https://acct.snowflakecomputing.com","mode":"custom_query_poll","sql":"DELETE FROM T"}`,
		`{"account_url":"https://acct.snowflakecomputing.com","mode":"custom_query_poll","sql":"SELECT 1; SELECT 2"}`,
	} {
		if err := (&source{}).Validate([]byte(raw)); err == nil {
			t.Fatalf("Validate(%s) expected error", raw)
		}
	}
}

func TestValidateRejectsMalformedConfigAndBoundaries(t *testing.T) {
	cases := []string{
		`{`,
		`{"account_url":"acct","database":"DB","schema":"S","table":"T"}`,
		`{"account_url":"https://acct.snowflakecomputing.com","database":"DB","schema":"S","table":"T","max_batch_size":-1}`,
		`{"account_url":"https://acct.snowflakecomputing.com","database":"DB","schema":"S","table":"T","max_batch_size":501}`,
		`{"account_url":"https://acct.snowflakecomputing.com","database":"DB","schema":"S","table":"T","timeout_seconds":-1}`,
		`{"account_url":"https://acct.snowflakecomputing.com","database":"DB","schema":"S","table":"T","timeout_seconds":61}`,
		`{"account_url":"https://acct.snowflakecomputing.com","mode":"custom_query_poll","sql":""}`,
	}
	for _, raw := range cases {
		if err := (&source{}).Validate([]byte(raw)); err == nil {
			t.Fatalf("Validate(%s) expected error", raw)
		}
	}
}

func TestValidateCustomQueryAllowsReadOnlySQL(t *testing.T) {
	for _, sql := range []string{"SELECT * FROM EVENTS", "WITH recent AS (SELECT 1) SELECT * FROM recent"} {
		raw := `{"account_url":"https://acct.snowflakecomputing.com","mode":"custom_query_poll","sql":` + strconvQuote(sql) + `}`
		if err := (&source{}).Validate([]byte(raw)); err != nil {
			t.Fatalf("Validate(%s) = %v", sql, err)
		}
	}
}

func TestRunRejectsBeforeStartingPollLoop(t *testing.T) {
	err := (&source{}).Run(context.Background(), json.RawMessage(`{`), "pat", func(sources.Event) {})
	if err == nil {
		t.Fatal("Run invalid config expected error")
	}

	err = (&source{}).Run(context.Background(), json.RawMessage(`{"account_url":"https://acct.snowflakecomputing.com"}`), " ", func(sources.Event) {})
	if err == nil || !strings.Contains(err.Error(), "secret PAT is required") {
		t.Fatalf("Run blank secret error = %v", err)
	}
}

func TestParseConfigTrimsAndDefaults(t *testing.T) {
	cfg, err := parseConfig([]byte(`{
		"mode":" custom_query_poll ",
		"account_url":" https://acct.snowflakecomputing.com/ ",
		"database":" DB ",
		"schema":" S ",
		"table":" T ",
		"event_id_column":" ID ",
		"event_type_column":" TYPE ",
		"payload_column":" BODY ",
		"watermark_column":" CREATED_AT "
	}`))
	if err != nil {
		t.Fatalf("parseConfig: %v", err)
	}
	if cfg.AccountURL != "https://acct.snowflakecomputing.com" {
		t.Fatalf("AccountURL = %q", cfg.AccountURL)
	}
	if cfg.Mode != modeCustomQueryPoll || cfg.Database != "DB" || cfg.EventIDColumn != "ID" {
		t.Fatalf("config was not normalized: %+v", cfg)
	}
	if cfg.IntervalSeconds != defaultIntervalSeconds || cfg.MaxBatchSize != defaultMaxBatchSize || cfg.TimeoutSeconds != 20 {
		t.Fatalf("defaults not applied: %+v", cfg)
	}
}

func TestValidateAccountURLAndAPIURL(t *testing.T) {
	endpoint, err := validateAccountURL(" https://ACCT.SNOWFLAKECOMPUTING.COM ")
	if err != nil {
		t.Fatalf("validateAccountURL(valid): %v", err)
	}
	got, err := endpoint.apiURL("/api/v2/statements/stmt")
	if err != nil {
		t.Fatalf("apiURL(valid): %v", err)
	}
	if got != "https://snowflakecomputing.com/api/v2/statements/stmt" {
		t.Fatalf("apiURL = %q", got)
	}

	for _, raw := range []string{
		"http://acct.snowflakecomputing.com",
		"https://user@acct.snowflakecomputing.com",
		"https://acct.snowflakecomputing.com/path",
		"https://acct.snowflakecomputing.com?x=1",
		"https://acct.snowflakecomputing.com#frag",
		"https://acct.snowflakecomputing.com:443",
		"https://metadata.internal",
	} {
		if _, err := validateAccountURL(raw); err == nil {
			t.Fatalf("validateAccountURL(%q) expected error", raw)
		}
	}

	if _, err := endpoint.apiURL("/api/v1/other"); err == nil {
		t.Fatal("apiURL outside SQL API path expected error")
	}
}

func TestStatementStatusPathValidation(t *testing.T) {
	got, err := statementStatusPath("", "stmt/with space")
	if err != nil {
		t.Fatalf("statementStatusPath(default): %v", err)
	}
	if got != "/api/v2/statements/stmt%2Fwith%20space" {
		t.Fatalf("default status path = %q", got)
	}
	got, err = statementStatusPath("/api/v2/statements/async-1", "ignored")
	if err != nil {
		t.Fatalf("statementStatusPath(relative): %v", err)
	}
	if got != "/api/v2/statements/async-1" {
		t.Fatalf("relative status path = %q", got)
	}

	for _, raw := range []string{
		"https://metadata.internal/api/v2/statements/stmt",
		"//metadata.internal/api/v2/statements/stmt",
		"/api/v2/statements/stmt?x=1",
		"/api/v2/statements/stmt#frag",
		"/metadata",
		"%zz",
	} {
		if _, err := statementStatusPath(raw, "stmt"); err == nil {
			t.Fatalf("statementStatusPath(%q) expected error", raw)
		}
	}
	if _, err := statementStatusPath("", ""); err == nil {
		t.Fatal("statementStatusPath missing handle expected error")
	}
}

func TestSQLAPIClientDefaultTransportUsesValidatedAccountHost(t *testing.T) {
	httpClient := (&sqlAPIClient{}).httpClientFor(snowflakeAccountEndpoint{host: "acct.snowflakecomputing.com"})
	transport, ok := httpClient.Transport.(snowflakeAccountTransport)
	if !ok {
		t.Fatalf("Transport = %T, want snowflakeAccountTransport", httpClient.Transport)
	}
	if transport.host != "acct.snowflakecomputing.com" {
		t.Fatalf("transport host = %q", transport.host)
	}
}

func TestSnowflakeAccountTransportRewritesHost(t *testing.T) {
	transport := snowflakeAccountTransport{
		host: "acct.snowflakecomputing.com",
		base: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			if req.URL.Host != "acct.snowflakecomputing.com" {
				t.Fatalf("request host = %q", req.URL.Host)
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader("{}")),
				Header:     make(http.Header),
			}, nil
		}),
	}
	req := httptest.NewRequest(http.MethodGet, "https://snowflakecomputing.com/api/v2/statements/stmt", nil)
	resp, err := transport.RoundTrip(req)
	if err != nil {
		t.Fatalf("RoundTrip: %v", err)
	}
	defer resp.Body.Close()
	if req.URL.Host != "snowflakecomputing.com" {
		t.Fatalf("original request host mutated to %q", req.URL.Host)
	}
}

func TestRunDefaultClientUsesValidatedAccountHost(t *testing.T) {
	originalTransport := http.DefaultTransport
	defer func() { http.DefaultTransport = originalTransport }()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var gotHost string
	http.DefaultTransport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
		gotHost = req.URL.Host
		return &http.Response{
			StatusCode: http.StatusOK,
			Body: io.NopCloser(strings.NewReader(`{
				"statementHandle":"run-default",
				"resultSetMetaData":{"rowType":[
					{"name":"EVENT_ID"},
					{"name":"EVENT_TYPE"},
					{"name":"PAYLOAD"},
					{"name":"OCCURRED_AT"}
				]},
				"data":[["evt_1","snowflake.default","{}", "2026-06-11T10:00:00.000"]]
			}`)),
			Header: make(http.Header),
		}, nil
	})

	err := (&source{}).Run(ctx, json.RawMessage(`{
		"account_url":"https://acct.snowflakecomputing.com",
		"database":"OBSERVABILITY",
		"schema":"AGENTFIELD",
		"table":"AGENTFIELD_EVENTS",
		"interval_seconds":5
	}`), "pat", func(sources.Event) {
		cancel()
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Run error = %v, want context.Canceled", err)
	}
	if gotHost != "acct.snowflakecomputing.com" {
		t.Fatalf("default Run request host = %q", gotHost)
	}
}

func TestPollOnceCallsSnowflakeSQLAPIAndEmitsEvents(t *testing.T) {
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
			"statementHandle":"01b-123",
			"resultSetMetaData":{"rowType":[
				{"name":"EVENT_ID"},
				{"name":"EVENT_TYPE"},
				{"name":"PAYLOAD"},
				{"name":"OCCURRED_AT"}
			]},
			"data":[[
				"evt_1",
				"snowflake.alert.fired",
				"{\"severity\":\"high\",\"count\":7}",
				"2026-06-11 10:00:00.000 -0700"
			]]
		}`))
	}))
	defer server.Close()

	cfg, err := parseConfig([]byte(`{
			"account_url":"https://acct.snowflakecomputing.com",
		"database":"OBSERVABILITY",
		"schema":"AGENTFIELD",
		"table":"AGENTFIELD_EVENTS",
		"warehouse":"COMPUTE_WH",
		"role":"AGENTFIELD_TRIGGER",
		"max_batch_size":10
	}`))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	var emitted []sources.Event
	next, err := (&source{}).pollOnce(context.Background(), &sqlAPIClient{httpClient: snowflakeTestHTTPClient(t, server)}, cfg, "pat-secret", pollCursor{}, func(e sources.Event) {
		emitted = append(emitted, e)
	})
	if err != nil {
		t.Fatalf("pollOnce: %v", err)
	}
	if gotAuth != "Bearer pat-secret" {
		t.Fatalf("Authorization = %q", gotAuth)
	}
	if gotTokenType != "PROGRAMMATIC_ACCESS_TOKEN" {
		t.Fatalf("token type = %q", gotTokenType)
	}
	if !strings.Contains(gotStatement, `FROM "OBSERVABILITY"."AGENTFIELD"."AGENTFIELD_EVENTS"`) {
		t.Fatalf("statement did not target configured table: %s", gotStatement)
	}
	if !strings.Contains(gotStatement, `TO_VARCHAR("OCCURRED_AT", 'YYYY-MM-DD"T"HH24:MI:SS.FF9') AS OCCURRED_AT`) {
		t.Fatalf("statement did not normalize watermark column: %s", gotStatement)
	}
	if next.Timestamp == "" || next.EventID == "" {
		t.Fatal("expected cursor")
	}
	if len(emitted) != 1 {
		t.Fatalf("emitted %d events, want 1", len(emitted))
	}
	if emitted[0].Type != "snowflake.alert.fired" || emitted[0].IdempotencyKey != "evt_1" {
		t.Fatalf("bad event metadata: %+v", emitted[0])
	}
	var normalized map[string]any
	if err := json.Unmarshal(emitted[0].Normalized, &normalized); err != nil {
		t.Fatalf("normalized JSON: %v", err)
	}
	payload := normalized["payload"].(map[string]any)
	if payload["severity"] != "high" {
		t.Fatalf("payload was not decoded: %#v", payload)
	}
}

func TestBuildPollStatementUsesCompositeCursor(t *testing.T) {
	cfg := config{
		Mode:            modeEventTablePoll,
		Database:        "OBSERVABILITY",
		Schema:          "AGENTFIELD",
		Table:           "AGENTFIELD_EVENTS",
		EventIDColumn:   "EVENT_ID",
		EventTypeColumn: "EVENT_TYPE",
		PayloadColumn:   "PAYLOAD",
		WatermarkColumn: "OCCURRED_AT",
		MaxBatchSize:    100,
	}

	stmt := buildPollStatement(cfg, pollCursor{Timestamp: "2026-06-11T18:52:00.123456789", EventID: "evt_42"})
	if !strings.Contains(stmt, `WHERE ("OCCURRED_AT" > TO_TIMESTAMP_NTZ('2026-06-11T18:52:00.123456789') OR ("OCCURRED_AT" = TO_TIMESTAMP_NTZ('2026-06-11T18:52:00.123456789') AND "EVENT_ID" > 'evt_42'))`) {
		t.Fatalf("statement did not use composite cursor predicate: %s", stmt)
	}
	if !strings.Contains(stmt, `ORDER BY "OCCURRED_AT", "EVENT_ID"`) {
		t.Fatalf("statement did not order by timestamp and event id: %s", stmt)
	}
}

func TestBuildPollStatementCustomQueryAndEscapedWatermark(t *testing.T) {
	custom := buildPollStatement(config{
		Mode:         modeCustomQueryPoll,
		SQL:          " SELECT * FROM EVENTS ",
		MaxBatchSize: 25,
	}, pollCursor{})
	if custom != "SELECT * FROM (SELECT * FROM EVENTS) LIMIT 25" {
		t.Fatalf("custom statement = %s", custom)
	}

	cfg := config{
		Mode:            modeEventTablePoll,
		Database:        "DB",
		Schema:          "S",
		Table:           "T",
		EventIDColumn:   "EVENT_ID",
		EventTypeColumn: "EVENT_TYPE",
		PayloadColumn:   "PAYLOAD",
		WatermarkColumn: "OCCURRED_AT",
		MaxBatchSize:    5,
	}
	stmt := buildPollStatement(cfg, pollCursor{Timestamp: "2026-06-11T10:00:00'Z", EventID: "evt'1"})
	if !strings.Contains(stmt, "2026-06-11T10:00:00''Z") {
		t.Fatalf("statement did not escape watermark: %s", stmt)
	}
	if !strings.Contains(stmt, "evt''1") {
		t.Fatalf("statement did not escape event id: %s", stmt)
	}
}

func TestSQLAPIClientPollsAcceptedStatement(t *testing.T) {
	var postSeen bool
	var getSeen bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.Method {
		case http.MethodPost:
			postSeen = true
			w.WriteHeader(http.StatusAccepted)
			_, _ = w.Write([]byte(`{"statementHandle":"async-1","statementStatusUrl":"/api/v2/statements/async-1"}`))
		case http.MethodGet:
			getSeen = true
			if r.Header.Get("X-Snowflake-Authorization-Token-Type") != "PROGRAMMATIC_ACCESS_TOKEN" {
				t.Fatalf("missing PAT token type on poll")
			}
			_, _ = w.Write([]byte(`{
				"statementHandle":"async-1",
				"resultSetMetaData":{"rowType":[{"name":"EVENT_ID"},{"name":"EVENT_TYPE"},{"name":"PAYLOAD"},{"name":"OCCURRED_AT"}]},
				"data":[["evt_async","snowflake.async","{}", "2026-06-11 10:00:00.000 -0700"]]
			}`))
		default:
			t.Fatalf("unexpected method %s", r.Method)
		}
	}))
	defer server.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	res, err := (&sqlAPIClient{httpClient: snowflakeTestHTTPClient(t, server)}).Execute(ctx, config{
		AccountURL: "https://acct.snowflakecomputing.com", TimeoutSeconds: 2,
	}, "pat", "SELECT 1")
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !postSeen || !getSeen {
		t.Fatalf("postSeen=%v getSeen=%v", postSeen, getSeen)
	}
	if res.StatementHandle != "async-1" || len(res.Data) != 1 {
		t.Fatalf("bad async result: %+v", res)
	}
}

func TestSQLAPIClientExecuteErrors(t *testing.T) {
	_, err := (&sqlAPIClient{}).Execute(context.Background(), config{
		AccountURL: "https://metadata.internal", TimeoutSeconds: 1,
	}, "pat", "SELECT 1")
	if err == nil || !strings.Contains(err.Error(), "account_url host") {
		t.Fatalf("Execute invalid account error = %v", err)
	}

	_, err = (&sqlAPIClient{httpClient: &http.Client{Transport: errorTransport{}}}).Execute(context.Background(), config{
		AccountURL: "https://acct.snowflakecomputing.com", TimeoutSeconds: 1,
	}, "pat", "SELECT 1")
	if err == nil || !strings.Contains(err.Error(), "transport down") {
		t.Fatalf("Execute transport error = %v", err)
	}

	tests := []struct {
		name    string
		status  int
		body    string
		wantErr string
	}{
		{name: "http status", status: http.StatusBadRequest, body: `bad request`, wantErr: "SQL API status 400"},
		{name: "invalid json", status: http.StatusOK, body: `{`, wantErr: "decode SQL API response"},
		{name: "api code", status: http.StatusOK, body: `{"code":"123","message":"boom"}`, wantErr: "SQL API code 123"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(tt.status)
				_, _ = w.Write([]byte(tt.body))
			}))
			defer server.Close()

			_, err := (&sqlAPIClient{httpClient: snowflakeTestHTTPClient(t, server)}).Execute(context.Background(), config{
				AccountURL: "https://acct.snowflakecomputing.com", TimeoutSeconds: 1,
			}, "pat", "SELECT 1")
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("Execute error = %v, want %q", err, tt.wantErr)
			}
		})
	}
}

func TestSQLAPIClientPollStatementErrors(t *testing.T) {
	client := &sqlAPIClient{}
	_, err := client.pollStatement(context.Background(), config{
		AccountURL: "https://acct.snowflakecomputing.com", TimeoutSeconds: 1,
	}, "pat", statementResponse{})
	if err == nil || !strings.Contains(err.Error(), "missing statementHandle") {
		t.Fatalf("missing handle error = %v", err)
	}

	_, err = client.pollStatement(context.Background(), config{
		AccountURL: "https://acct.snowflakecomputing.com", TimeoutSeconds: 1,
	}, "pat", statementResponse{
		StatementHandle:    "stmt",
		StatementStatusURL: "https://metadata.internal/api/v2/statements/stmt",
	})
	if err == nil || !strings.Contains(err.Error(), "relative SQL API path") {
		t.Fatalf("absolute status URL error = %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = client.pollStatement(ctx, config{AccountURL: "https://acct.snowflakecomputing.com", TimeoutSeconds: 1}, "pat", statementResponse{StatementHandle: "stmt"})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("pollStatement canceled error = %v", err)
	}

	_, err = (&sqlAPIClient{}).pollStatement(context.Background(), config{
		AccountURL: "https://metadata.internal", TimeoutSeconds: 1,
	}, "pat", statementResponse{StatementHandle: "stmt"})
	if err == nil || !strings.Contains(err.Error(), "account_url host") {
		t.Fatalf("pollStatement invalid account error = %v", err)
	}

	ctx, cancel = context.WithTimeout(context.Background(), 1500*time.Millisecond)
	defer cancel()
	_, err = (&sqlAPIClient{httpClient: &http.Client{Transport: errorTransport{}}}).pollStatement(ctx, config{
		AccountURL: "https://acct.snowflakecomputing.com", TimeoutSeconds: 1,
	}, "pat", statementResponse{StatementHandle: "stmt"})
	if err == nil || !strings.Contains(err.Error(), "transport down") {
		t.Fatalf("pollStatement transport error = %v", err)
	}
}

func TestResultToEventsRequiresStandardColumns(t *testing.T) {
	_, _, err := resultToEvents(config{}, statementResponse{
		ResultSetMeta: resultSetMetadata{RowType: []columnMeta{{Name: "EVENT_ID"}}},
		Data:          [][]any{{"evt"}},
	})
	if err == nil || !strings.Contains(err.Error(), "missing EVENT_TYPE") {
		t.Fatalf("expected missing column error, got %v", err)
	}
}

func TestResultToEventsNormalizesValueTypes(t *testing.T) {
	res := statementResponse{
		StatementHandle: "query-1",
		ResultSetMeta: resultSetMetadata{RowType: []columnMeta{
			{Name: "event_id"},
			{Name: "event_type"},
			{Name: "payload"},
			{Name: "occurred_at"},
		}},
		Data: [][]any{
			{json.Number("42"), "snowflake.number", "not-json", nil},
			{map[string]string{"id": "nested"}, "snowflake.object", map[string]any{"ok": true}, "2026-06-11T18:00:00Z"},
		},
	}
	events, cursor, err := resultToEvents(config{AccountURL: "https://acct", Database: "DB", Schema: "S", Table: "T"}, res)
	if err != nil {
		t.Fatalf("resultToEvents: %v", err)
	}
	if len(events) != 2 {
		t.Fatalf("events = %d, want 2", len(events))
	}
	if events[0].IdempotencyKey != "42" {
		t.Fatalf("number id was not stringified: %+v", events[0])
	}
	if events[1].IdempotencyKey != `{"id":"nested"}` {
		t.Fatalf("object id was not JSON stringified: %+v", events[1])
	}
	if cursor.Timestamp != "2026-06-11T18:00:00Z" || cursor.EventID != `{"id":"nested"}` {
		t.Fatalf("cursor = %+v", cursor)
	}
}

func strconvQuote(v string) string {
	b, _ := json.Marshal(v)
	return string(b)
}

type rewriteTransport struct {
	target *url.URL
}

func (t rewriteTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	clone := req.Clone(req.Context())
	clone.URL.Scheme = t.target.Scheme
	clone.URL.Host = t.target.Host
	return http.DefaultTransport.RoundTrip(clone)
}

func snowflakeTestHTTPClient(t *testing.T, server *httptest.Server) *http.Client {
	t.Helper()
	target, err := url.Parse(server.URL)
	if err != nil {
		t.Fatalf("parse test server URL: %v", err)
	}
	return &http.Client{Transport: rewriteTransport{target: target}}
}

type errorTransport struct{}

func (errorTransport) RoundTrip(*http.Request) (*http.Response, error) {
	return nil, errors.New("transport down")
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}
