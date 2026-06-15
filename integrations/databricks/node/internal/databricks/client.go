package databricks

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
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

var identifierPathRE = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*){0,2}$`)

type Config struct {
	WorkspaceURL   string
	Token          string
	WarehouseID    string
	Catalog        string
	Schema         string
	AIEndpoint     string
	TimeoutSeconds int
	MaxRows        int
	HTTPClient     *http.Client
}

type Client struct {
	cfg        Config
	httpClient *http.Client
}

func NewClient(cfg Config) (*Client, error) {
	cfg.WorkspaceURL = strings.TrimRight(strings.TrimSpace(cfg.WorkspaceURL), "/")
	if cfg.WorkspaceURL == "" {
		return nil, errors.New("databricks workspace URL is required")
	}
	if _, err := url.ParseRequestURI(cfg.WorkspaceURL); err != nil {
		return nil, fmt.Errorf("invalid databricks workspace URL: %w", err)
	}
	if cfg.Token == "" {
		return nil, errors.New("databricks token is required")
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
	SQL            string         `json:"sql"`
	Catalog        string         `json:"catalog,omitempty"`
	Schema         string         `json:"schema,omitempty"`
	WarehouseID    string         `json:"warehouse_id,omitempty"`
	MaxRows        int            `json:"max_rows,omitempty"`
	TimeoutSeconds int            `json:"timeout_seconds,omitempty"`
	Parameters     []SQLParameter `json:"parameters,omitempty"`
}

type SQLParameter struct {
	Name  string `json:"name"`
	Value string `json:"value"`
	Type  string `json:"type,omitempty"`
}

type QueryResult struct {
	StatementID string           `json:"statement_id"`
	Columns     []string         `json:"columns"`
	Rows        []map[string]any `json:"rows"`
	RowCount    int              `json:"row_count"`
	Truncated   bool             `json:"truncated"`
	Provenance  map[string]any   `json:"provenance"`
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
	resp, err := c.executeStatement(ctx, statementRequest{
		Statement:   stmt,
		WarehouseID: firstString(req.WarehouseID, c.cfg.WarehouseID),
		Catalog:     firstString(req.Catalog, c.cfg.Catalog),
		Schema:      firstString(req.Schema, c.cfg.Schema),
		WaitTimeout: waitTimeout(firstInt(req.TimeoutSeconds, c.cfg.TimeoutSeconds)),
		Disposition: "INLINE",
		Format:      "JSON_ARRAY",
		RowLimit:    maxRows + 1,
		Parameters:  req.Parameters,
	}, firstInt(req.TimeoutSeconds, c.cfg.TimeoutSeconds))
	if err != nil {
		return QueryResult{}, err
	}
	columns := columnNames(resp.Manifest.Schema.Columns)
	rows := rowsToMaps(columns, resp.Result.DataArray)
	truncated := resp.Manifest.Truncated || len(rows) > maxRows
	if truncated && len(rows) > maxRows {
		rows = rows[:maxRows]
	}
	return QueryResult{
		StatementID: resp.StatementID,
		Columns:     columns,
		Rows:        rows,
		RowCount:    len(rows),
		Truncated:   truncated,
		Provenance: map[string]any{
			"workspace_url": c.cfg.WorkspaceURL,
			"warehouse_id":  firstString(req.WarehouseID, c.cfg.WarehouseID),
			"catalog":       firstString(req.Catalog, c.cfg.Catalog),
			"schema":        firstString(req.Schema, c.cfg.Schema),
		},
	}, nil
}

func (c *Client) Defaults() map[string]string {
	return map[string]string{
		"workspace_url": c.cfg.WorkspaceURL,
		"warehouse_id":  c.cfg.WarehouseID,
		"catalog":       c.cfg.Catalog,
		"schema":        c.cfg.Schema,
		"ai_endpoint":   c.cfg.AIEndpoint,
	}
}

func (c *Client) DescribeTable(ctx context.Context, object string) (QueryResult, error) {
	if !identifierPathRE.MatchString(strings.TrimSpace(object)) {
		return QueryResult{}, fmt.Errorf("invalid table identifier %q", object)
	}
	return c.QueryReadOnly(ctx, QueryRequest{SQL: "DESCRIBE TABLE " + object, MaxRows: 500})
}

func (c *Client) SearchColumns(ctx context.Context, query, catalog, schema string, limit int) (QueryResult, error) {
	if limit <= 0 || limit > 100 {
		limit = 25
	}
	cat := firstString(catalog, c.cfg.Catalog)
	if cat == "" {
		return QueryResult{}, errors.New("catalog is required for search_columns")
	}
	if !identifierPathRE.MatchString(cat) {
		return QueryResult{}, fmt.Errorf("invalid catalog identifier %q", cat)
	}
	if schema != "" && !identifierPathRE.MatchString(schema) {
		return QueryResult{}, fmt.Errorf("invalid schema identifier %q", schema)
	}
	sql := fmt.Sprintf(`SELECT table_catalog, table_schema, table_name, column_name, data_type, comment
FROM %s.information_schema.columns
WHERE UPPER(column_name) LIKE UPPER(:pattern)`, quotePath(cat))
	params := []SQLParameter{{Name: "pattern", Value: "%" + query + "%", Type: "STRING"}}
	if schema != "" {
		sql += " AND table_schema = :schema_name"
		params = append(params, SQLParameter{Name: "schema_name", Value: schema, Type: "STRING"})
	}
	sql += " ORDER BY table_schema, table_name, ordinal_position"
	return c.QueryReadOnly(ctx, QueryRequest{SQL: sql, MaxRows: limit, Parameters: params})
}

func (c *Client) AIQuery(ctx context.Context, endpoint, prompt string, returnType string, failOnError bool) (QueryResult, error) {
	endpoint = firstString(endpoint, c.cfg.AIEndpoint)
	if endpoint == "" {
		return QueryResult{}, errors.New("endpoint is required for ai_query")
	}
	if strings.TrimSpace(prompt) == "" {
		return QueryResult{}, errors.New("prompt is required for ai_query")
	}
	sql := "SELECT ai_query(:endpoint, :prompt) AS response"
	params := []SQLParameter{
		{Name: "endpoint", Value: endpoint, Type: "STRING"},
		{Name: "prompt", Value: prompt, Type: "STRING"},
	}
	if strings.TrimSpace(returnType) != "" {
		sql = "SELECT ai_query(:endpoint, :prompt, :return_type, :fail_on_error) AS response"
		params = append(params,
			SQLParameter{Name: "return_type", Value: returnType, Type: "STRING"},
			SQLParameter{Name: "fail_on_error", Value: strconv.FormatBool(failOnError), Type: "BOOLEAN"},
		)
	}
	return c.QueryReadOnly(ctx, QueryRequest{SQL: sql, MaxRows: 1, Parameters: params})
}

func (c *Client) InvokeServingEndpoint(ctx context.Context, endpoint string, body any) (map[string]any, error) {
	endpoint = strings.Trim(strings.TrimSpace(endpoint), "/")
	if endpoint == "" {
		return nil, errors.New("endpoint is required")
	}
	if strings.Contains(endpoint, "/") || strings.Contains(endpoint, "..") {
		return nil, fmt.Errorf("invalid serving endpoint name %q", endpoint)
	}
	var raw map[string]any
	if value, ok := body.(map[string]any); ok {
		raw = value
	} else {
		data, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		if err := json.Unmarshal(data, &raw); err != nil {
			return nil, err
		}
	}
	return c.PostJSON(ctx, "/serving-endpoints/"+url.PathEscape(endpoint)+"/invocations", raw)
}

func (c *Client) PostJSON(ctx context.Context, path string, body any) (map[string]any, error) {
	if !strings.HasPrefix(path, "/api/") && !strings.HasPrefix(path, "/serving-endpoints/") {
		return nil, errors.New("databricks API path must stay under /api or /serving-endpoints")
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.cfg.WorkspaceURL+path, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+c.cfg.Token)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("databricks REST status %d: %s", resp.StatusCode, strings.TrimSpace(string(data)))
	}
	var out map[string]any
	if len(bytes.TrimSpace(data)) == 0 {
		return map[string]any{}, nil
	}
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, err
	}
	return out, nil
}

type statementRequest struct {
	Statement   string         `json:"statement"`
	WarehouseID string         `json:"warehouse_id,omitempty"`
	Catalog     string         `json:"catalog,omitempty"`
	Schema      string         `json:"schema,omitempty"`
	Parameters  []SQLParameter `json:"parameters,omitempty"`
	WaitTimeout string         `json:"wait_timeout,omitempty"`
	Disposition string         `json:"disposition,omitempty"`
	Format      string         `json:"format,omitempty"`
	RowLimit    int            `json:"row_limit,omitempty"`
}

type statementResponse struct {
	StatementID string `json:"statement_id"`
	Status      struct {
		State string `json:"state"`
		Error *struct {
			ErrorCode string `json:"error_code"`
			Message   string `json:"message"`
		} `json:"error,omitempty"`
	} `json:"status"`
	Manifest struct {
		Truncated bool `json:"truncated"`
		Schema    struct {
			Columns []columnMeta `json:"columns"`
		} `json:"schema"`
	} `json:"manifest"`
	Result struct {
		DataArray [][]any `json:"data_array"`
		RowCount  int     `json:"row_count"`
	} `json:"result"`
}

type columnMeta struct {
	Name string `json:"name"`
}

func (c *Client) executeStatement(ctx context.Context, reqBody statementRequest, timeoutSeconds int) (statementResponse, error) {
	var out statementResponse
	if strings.TrimSpace(reqBody.WarehouseID) == "" {
		return out, errors.New("warehouse_id is required")
	}
	raw, err := c.PostJSON(ctx, "/api/2.0/sql/statements", reqBody)
	if err != nil {
		return out, err
	}
	data, _ := json.Marshal(raw)
	if err := json.Unmarshal(data, &out); err != nil {
		return out, err
	}
	return c.finishStatement(ctx, out, timeoutSeconds)
}

func (c *Client) finishStatement(ctx context.Context, status statementResponse, timeoutSeconds int) (statementResponse, error) {
	state := strings.ToUpper(status.Status.State)
	switch state {
	case "SUCCEEDED":
		return status, nil
	case "FAILED", "CANCELED", "CLOSED":
		if status.Status.Error != nil {
			return status, fmt.Errorf("databricks statement %s: %s", status.Status.Error.ErrorCode, status.Status.Error.Message)
		}
		return status, fmt.Errorf("databricks statement ended in %s", state)
	}
	if status.StatementID == "" {
		return status, errors.New("databricks statement response missing statement_id")
	}
	if timeoutSeconds <= 0 {
		timeoutSeconds = c.cfg.TimeoutSeconds
	}
	deadline := time.Now().Add(time.Duration(timeoutSeconds) * time.Second)
	for {
		if time.Now().After(deadline) {
			return status, fmt.Errorf("statement %s did not complete before timeout", status.StatementID)
		}
		select {
		case <-ctx.Done():
			return status, ctx.Err()
		case <-time.After(time.Second):
		}
		next, err := c.getStatement(ctx, status.StatementID)
		if err != nil {
			return status, err
		}
		state = strings.ToUpper(next.Status.State)
		switch state {
		case "SUCCEEDED":
			return next, nil
		case "FAILED", "CANCELED", "CLOSED":
			if next.Status.Error != nil {
				return next, fmt.Errorf("databricks statement %s: %s", next.Status.Error.ErrorCode, next.Status.Error.Message)
			}
			return next, fmt.Errorf("databricks statement ended in %s", state)
		}
		status = next
	}
}

func (c *Client) getStatement(ctx context.Context, statementID string) (statementResponse, error) {
	var out statementResponse
	if statementID == "" {
		return out, errors.New("statement_id is required")
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.cfg.WorkspaceURL+"/api/2.0/sql/statements/"+url.PathEscape(statementID), nil)
	if err != nil {
		return out, err
	}
	req.Header.Set("Authorization", "Bearer "+c.cfg.Token)
	req.Header.Set("Accept", "application/json")
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return out, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return out, fmt.Errorf("databricks SQL API status %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}
	if err := json.Unmarshal(body, &out); err != nil {
		return out, err
	}
	return out, nil
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

func EventID(value any) string {
	data, _ := json.Marshal(value)
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:16])
}

func applyLimit(sql string, limit int) string {
	trimmed := strings.TrimSpace(sql)
	upper := strings.ToUpper(trimmed)
	if strings.Contains(upper, " LIMIT ") || strings.HasPrefix(upper, "SHOW ") || strings.HasPrefix(upper, "DESCRIBE ") || strings.HasPrefix(upper, "DESC ") || strings.HasPrefix(upper, "EXPLAIN ") {
		return sql
	}
	return fmt.Sprintf("SELECT * FROM (%s) LIMIT %d", sql, limit)
}

func waitTimeout(seconds int) string {
	if seconds <= 0 {
		seconds = 30
	}
	if seconds < 5 {
		seconds = 5
	}
	if seconds > 50 {
		seconds = 50
	}
	return strconv.Itoa(seconds) + "s"
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
		quoted = append(quoted, "`"+strings.ReplaceAll(part, "`", "``")+"`")
	}
	return strings.Join(quoted, ".")
}
