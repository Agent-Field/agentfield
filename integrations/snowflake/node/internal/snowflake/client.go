package snowflake

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"time"
)

const DefaultMaxRows = 100

var identifierPathRE = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_$]*(\.[A-Za-z_][A-Za-z0-9_$]*){0,2}$`)

type Config struct {
	AccountURL     string
	Token          string
	Database       string
	Schema         string
	Warehouse      string
	Role           string
	TimeoutSeconds int
	MaxRows        int
	HTTPClient     *http.Client
}

type Client struct {
	cfg        Config
	httpClient *http.Client
}

func NewClient(cfg Config) (*Client, error) {
	cfg.AccountURL = strings.TrimRight(strings.TrimSpace(cfg.AccountURL), "/")
	if cfg.AccountURL == "" {
		return nil, errors.New("snowflake account URL is required")
	}
	if cfg.Token == "" {
		return nil, errors.New("snowflake token is required")
	}
	if cfg.TimeoutSeconds <= 0 {
		cfg.TimeoutSeconds = 30
	}
	if cfg.MaxRows <= 0 {
		cfg.MaxRows = DefaultMaxRows
	}
	httpClient := cfg.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{Timeout: time.Duration(cfg.TimeoutSeconds+10) * time.Second}
	}
	return &Client{cfg: cfg, httpClient: httpClient}, nil
}

type QueryRequest struct {
	SQL            string `json:"sql"`
	Database       string `json:"database,omitempty"`
	Schema         string `json:"schema,omitempty"`
	Warehouse      string `json:"warehouse,omitempty"`
	Role           string `json:"role,omitempty"`
	MaxRows        int    `json:"max_rows,omitempty"`
	TimeoutSeconds int    `json:"timeout_seconds,omitempty"`
}

type QueryResult struct {
	QueryID    string           `json:"query_id"`
	Columns    []string         `json:"columns"`
	Rows       []map[string]any `json:"rows"`
	RowCount   int              `json:"row_count"`
	Truncated  bool             `json:"truncated"`
	Provenance map[string]any   `json:"provenance"`
}

func (c *Client) QueryReadOnly(ctx context.Context, req QueryRequest) (QueryResult, error) {
	if err := ValidateReadOnlySQL(req.SQL); err != nil {
		return QueryResult{}, err
	}
	maxRows := req.MaxRows
	if maxRows <= 0 || maxRows > c.cfg.MaxRows {
		maxRows = c.cfg.MaxRows
	}
	stmt := applyLimit(req.SQL, maxRows+1)
	resp, err := c.execute(ctx, statementRequest{
		Statement: stmt,
		Timeout:   firstInt(req.TimeoutSeconds, c.cfg.TimeoutSeconds),
		Database:  firstString(req.Database, c.cfg.Database),
		Schema:    firstString(req.Schema, c.cfg.Schema),
		Warehouse: firstString(req.Warehouse, c.cfg.Warehouse),
		Role:      firstString(req.Role, c.cfg.Role),
	})
	if err != nil {
		return QueryResult{}, err
	}
	columns := columnNames(resp.ResultSetMeta.RowType)
	rows := rowsToMaps(columns, resp.Data)
	truncated := len(rows) > maxRows
	if truncated {
		rows = rows[:maxRows]
	}
	return QueryResult{
		QueryID:   resp.StatementHandle,
		Columns:   columns,
		Rows:      rows,
		RowCount:  len(rows),
		Truncated: truncated,
		Provenance: map[string]any{
			"account_url": c.cfg.AccountURL,
			"database":    firstString(req.Database, c.cfg.Database),
			"schema":      firstString(req.Schema, c.cfg.Schema),
			"warehouse":   firstString(req.Warehouse, c.cfg.Warehouse),
			"role":        firstString(req.Role, c.cfg.Role),
		},
	}, nil
}

func (c *Client) Defaults() map[string]string {
	return map[string]string{
		"account_url": c.cfg.AccountURL,
		"database":    c.cfg.Database,
		"schema":      c.cfg.Schema,
		"warehouse":   c.cfg.Warehouse,
		"role":        c.cfg.Role,
	}
}

func (c *Client) DescribeTable(ctx context.Context, object string) (QueryResult, error) {
	if !identifierPathRE.MatchString(strings.TrimSpace(object)) {
		return QueryResult{}, fmt.Errorf("invalid table identifier %q", object)
	}
	return c.QueryReadOnly(ctx, QueryRequest{SQL: "DESCRIBE TABLE " + object, MaxRows: 500})
}

func (c *Client) SearchColumns(ctx context.Context, query, database, schema string, limit int) (QueryResult, error) {
	if limit <= 0 || limit > 100 {
		limit = 25
	}
	if database != "" && !identifierPathRE.MatchString(database) {
		return QueryResult{}, fmt.Errorf("invalid database identifier %q", database)
	}
	if schema != "" && !identifierPathRE.MatchString(schema) {
		return QueryResult{}, fmt.Errorf("invalid schema identifier %q", schema)
	}
	db := firstString(database, c.cfg.Database)
	if db == "" {
		return QueryResult{}, errors.New("database is required for search_columns")
	}
	sql := fmt.Sprintf(`SELECT table_catalog, table_schema, table_name, column_name, data_type, comment
FROM %s.INFORMATION_SCHEMA.COLUMNS
WHERE UPPER(column_name) LIKE UPPER('%s')`, quotePath(db), "%"+escapeSQLLike(query)+"%")
	if schema != "" {
		sql += fmt.Sprintf(" AND table_schema = '%s'", escapeSQL(schema))
	}
	sql += " ORDER BY table_schema, table_name, ordinal_position"
	return c.QueryReadOnly(ctx, QueryRequest{SQL: sql, MaxRows: limit})
}

func (c *Client) CortexComplete(ctx context.Context, model, prompt string) (string, string, error) {
	if strings.TrimSpace(model) == "" {
		return "", "", errors.New("model is required")
	}
	if strings.TrimSpace(prompt) == "" {
		return "", "", errors.New("prompt is required")
	}
	sql := fmt.Sprintf("SELECT SNOWFLAKE.CORTEX.COMPLETE('%s', '%s') AS TEXT", escapeSQL(model), escapeSQL(prompt))
	res, err := c.QueryReadOnly(ctx, QueryRequest{SQL: sql, MaxRows: 1})
	if err != nil {
		return "", "", err
	}
	if len(res.Rows) == 0 {
		return "", res.QueryID, nil
	}
	return valueString(res.Rows[0]["TEXT"]), res.QueryID, nil
}

func (c *Client) CortexChatComplete(ctx context.Context, model, system, prompt string, maxTokens int, temperature float64) (string, string, map[string]any, error) {
	if strings.TrimSpace(model) == "" {
		return "", "", nil, errors.New("model is required")
	}
	if strings.TrimSpace(prompt) == "" {
		return "", "", nil, errors.New("prompt is required")
	}
	messages := []map[string]any{}
	if strings.TrimSpace(system) != "" {
		messages = append(messages, map[string]any{"role": "system", "content": system})
	}
	messages = append(messages, map[string]any{"role": "user", "content": prompt})
	body := map[string]any{
		"model":       model,
		"messages":    messages,
		"temperature": temperature,
	}
	if maxTokens > 0 {
		body["max_tokens"] = maxTokens
	}
	raw, requestID, err := c.PostJSON(ctx, "/api/v2/cortex/v1/chat/completions", body)
	if err != nil {
		return "", requestID, nil, err
	}
	text := ""
	if choices, ok := raw["choices"].([]any); ok && len(choices) > 0 {
		if choice, ok := choices[0].(map[string]any); ok {
			if message, ok := choice["message"].(map[string]any); ok {
				text = valueString(message["content"])
			}
		}
	}
	return text, requestID, raw, nil
}

func (c *Client) PostJSON(ctx context.Context, path string, body any) (map[string]any, string, error) {
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.cfg.AccountURL+path, bytes.NewReader(payload))
	if err != nil {
		return nil, "", err
	}
	req.Header.Set("Authorization", "Bearer "+c.cfg.Token)
	req.Header.Set("X-Snowflake-Authorization-Token-Type", "PROGRAMMATIC_ACCESS_TOKEN")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, "", err
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, "", fmt.Errorf("snowflake REST status %d: %s", resp.StatusCode, strings.TrimSpace(string(data)))
	}
	var out map[string]any
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, "", err
	}
	return out, resp.Header.Get("X-Snowflake-Request-Id"), nil
}

type statementRequest struct {
	Statement string `json:"statement"`
	Timeout   int    `json:"timeout,omitempty"`
	Database  string `json:"database,omitempty"`
	Schema    string `json:"schema,omitempty"`
	Warehouse string `json:"warehouse,omitempty"`
	Role      string `json:"role,omitempty"`
}

type statementResponse struct {
	StatementHandle    string            `json:"statementHandle"`
	StatementStatusURL string            `json:"statementStatusUrl"`
	Data               [][]any           `json:"data"`
	ResultSetMeta      resultSetMetadata `json:"resultSetMetaData"`
	Code               string            `json:"code"`
	Message            string            `json:"message"`
}

type resultSetMetadata struct {
	RowType []columnMeta `json:"rowType"`
}

type columnMeta struct {
	Name string `json:"name"`
}

func (c *Client) execute(ctx context.Context, reqBody statementRequest) (statementResponse, error) {
	var out statementResponse
	payload, err := json.Marshal(reqBody)
	if err != nil {
		return out, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.cfg.AccountURL+"/api/v2/statements", bytes.NewReader(payload))
	if err != nil {
		return out, err
	}
	req.Header.Set("Authorization", "Bearer "+c.cfg.Token)
	req.Header.Set("X-Snowflake-Authorization-Token-Type", "PROGRAMMATIC_ACCESS_TOKEN")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return out, err
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return out, fmt.Errorf("snowflake SQL API status %d: %s", resp.StatusCode, strings.TrimSpace(string(data)))
	}
	if err := json.Unmarshal(data, &out); err != nil {
		return out, err
	}
	if resp.StatusCode == http.StatusAccepted {
		return c.pollStatement(ctx, out)
	}
	if out.Code != "" && out.Code != "090001" {
		return out, fmt.Errorf("snowflake SQL API code %s: %s", out.Code, out.Message)
	}
	return out, nil
}

func (c *Client) pollStatement(ctx context.Context, status statementResponse) (statementResponse, error) {
	var out statementResponse
	if status.StatementHandle == "" {
		return out, errors.New("async SQL response missing statementHandle")
	}
	statusURL := status.StatementStatusURL
	if statusURL == "" {
		statusURL = "/api/v2/statements/" + status.StatementHandle
	}
	deadline := time.Now().Add(time.Duration(c.cfg.TimeoutSeconds) * time.Second)
	for {
		if time.Now().After(deadline) {
			return out, fmt.Errorf("statement %s did not complete before timeout", status.StatementHandle)
		}
		select {
		case <-ctx.Done():
			return out, ctx.Err()
		case <-time.After(time.Second):
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.cfg.AccountURL+statusURL, nil)
		if err != nil {
			return out, err
		}
		req.Header.Set("Authorization", "Bearer "+c.cfg.Token)
		req.Header.Set("X-Snowflake-Authorization-Token-Type", "PROGRAMMATIC_ACCESS_TOKEN")
		req.Header.Set("Accept", "application/json")
		resp, err := c.httpClient.Do(req)
		if err != nil {
			return out, err
		}
		data, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
		resp.Body.Close()
		if resp.StatusCode == http.StatusAccepted || resp.StatusCode == http.StatusTooManyRequests {
			continue
		}
		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			return out, fmt.Errorf("snowflake SQL API status %d: %s", resp.StatusCode, strings.TrimSpace(string(data)))
		}
		if err := json.Unmarshal(data, &out); err != nil {
			return out, err
		}
		if out.Code != "" && out.Code != "090001" {
			return out, fmt.Errorf("snowflake SQL API code %s: %s", out.Code, out.Message)
		}
		return out, nil
	}
}

func ValidateReadOnlySQL(sql string) error {
	sql = strings.TrimSpace(sql)
	if sql == "" {
		return errors.New("sql is required")
	}
	if strings.Contains(sql, ";") {
		return errors.New("sql must contain one statement")
	}
	fields := strings.Fields(sql)
	if len(fields) == 0 {
		return errors.New("sql is required")
	}
	switch strings.ToUpper(fields[0]) {
	case "SELECT", "WITH", "SHOW", "DESCRIBE", "DESC", "EXPLAIN":
		return nil
	default:
		return fmt.Errorf("sql must be read-only, got %s", strings.ToUpper(fields[0]))
	}
}

func applyLimit(sql string, limit int) string {
	upper := strings.ToUpper(strings.TrimSpace(sql))
	if strings.Contains(upper, " LIMIT ") || strings.HasPrefix(upper, "SHOW ") || strings.HasPrefix(upper, "DESCRIBE ") || strings.HasPrefix(upper, "DESC ") || strings.HasPrefix(upper, "EXPLAIN ") {
		return sql
	}
	return fmt.Sprintf("SELECT * FROM (%s) LIMIT %d", sql, limit)
}

func columnNames(cols []columnMeta) []string {
	out := make([]string, 0, len(cols))
	for _, col := range cols {
		out = append(out, col.Name)
	}
	return out
}

func rowsToMaps(cols []string, data [][]any) []map[string]any {
	out := make([]map[string]any, 0, len(data))
	for _, row := range data {
		item := make(map[string]any, len(cols))
		for i, col := range cols {
			if i < len(row) {
				item[col] = row[i]
			}
		}
		out = append(out, item)
	}
	return out
}

func firstString(values ...string) string {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

func firstInt(values ...int) int {
	for _, v := range values {
		if v > 0 {
			return v
		}
	}
	return 0
}

func quotePath(path string) string {
	parts := strings.Split(path, ".")
	quoted := make([]string, 0, len(parts))
	for _, part := range parts {
		quoted = append(quoted, `"`+strings.ReplaceAll(part, `"`, `""`)+`"`)
	}
	return strings.Join(quoted, ".")
}

func URLPath(parts ...string) string {
	escaped := make([]string, 0, len(parts))
	for _, part := range parts {
		escaped = append(escaped, url.PathEscape(part))
	}
	return strings.Join(escaped, "/")
}

func escapeSQL(v string) string {
	return strings.ReplaceAll(v, "'", "''")
}

func escapeSQLLike(v string) string {
	v = escapeSQL(v)
	v = strings.ReplaceAll(v, `%`, `\%`)
	v = strings.ReplaceAll(v, `_`, `\_`)
	return v
}

func valueString(v any) string {
	switch t := v.(type) {
	case string:
		return t
	case json.Number:
		return t.String()
	case float64:
		return strconv.FormatFloat(t, 'f', -1, 64)
	default:
		b, _ := json.Marshal(t)
		return string(b)
	}
}
