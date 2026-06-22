package capabilities

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/Agent-Field/agentfield/integrations/snowflake/node/internal/config"
	"github.com/Agent-Field/agentfield/integrations/snowflake/node/internal/prompts"
	"github.com/Agent-Field/agentfield/integrations/snowflake/node/internal/snowflake"
	afagent "github.com/Agent-Field/agentfield/sdk/go/agent"
)

type Runtime struct {
	Config    config.Config
	Snowflake *snowflake.Client
	Prompts   *prompts.Store
}

func Register(a *afagent.Agent, rt Runtime) {
	a.RegisterReasoner("health", rt.health, afagent.WithDescription("Report Snowflake node configuration health"), afagent.WithInputSchema(schema(`{"type":"object","additionalProperties":false}`)))
	a.RegisterReasoner("query_readonly", rt.queryReadOnly, afagent.WithDescription("Run a guarded read-only Snowflake SQL statement"))
	a.RegisterReasoner("describe_table", rt.describeTable, afagent.WithDescription("Describe a Snowflake table or view"))
	a.RegisterReasoner("search_columns", rt.searchColumns, afagent.WithDescription("Search Snowflake information_schema columns"))
	a.RegisterReasoner("cortex_complete", rt.cortexComplete, afagent.WithDescription("Call Snowflake Cortex COMPLETE through SQL API"))
	a.RegisterReasoner("cortex_analyst_message", rt.cortexAnalystMessage, afagent.WithDescription("Call Snowflake Cortex Analyst message API"))
	a.RegisterReasoner("cortex_search_query", rt.cortexSearchQuery, afagent.WithDescription("Query a Snowflake Cortex Search service"))
	a.RegisterReasoner("semantic_ask", rt.semanticAsk, afagent.WithDescription("Answer a data question using configured Snowflake semantic context and prompts"))
	a.RegisterReasoner("explain_result", rt.explainResult, afagent.WithDescription("Explain a Snowflake query result using config-loaded prompts"))
	a.RegisterReasoner("investigate_metric_change", rt.investigateMetricChange, afagent.WithDescription("Plan a bounded Snowflake metric-change investigation"))
}

func schema(raw string) json.RawMessage {
	return json.RawMessage(raw)
}

func (rt Runtime) health(_ context.Context, _ map[string]any) (any, error) {
	defaults := map[string]string{}
	if rt.Snowflake != nil {
		defaults = rt.Snowflake.Defaults()
	}
	return map[string]any{
		"status":               "ok",
		"node_id":              rt.Config.NodeID,
		"prompt_source":        rt.Prompts.Source(),
		"prompt_version":       rt.Prompts.Version(),
		"default_cortex_model": rt.Config.DefaultCortexModel,
		"snowflake":            defaults,
	}, nil
}

func (rt Runtime) queryReadOnly(ctx context.Context, input map[string]any) (any, error) {
	sql, err := requiredString(input, "sql")
	if err != nil {
		return nil, err
	}
	return rt.Snowflake.QueryReadOnly(ctx, snowflake.QueryRequest{
		SQL:            sql,
		Database:       stringInput(input, "database"),
		Schema:         stringInput(input, "schema"),
		Warehouse:      stringInput(input, "warehouse"),
		Role:           stringInput(input, "role"),
		MaxRows:        intInput(input, "max_rows"),
		TimeoutSeconds: intInput(input, "timeout_seconds"),
	})
}

func (rt Runtime) describeTable(ctx context.Context, input map[string]any) (any, error) {
	object := firstNonBlank(stringInput(input, "object"), stringInput(input, "table"))
	if object == "" {
		return nil, errors.New("object is required")
	}
	return rt.Snowflake.DescribeTable(ctx, object)
}

func (rt Runtime) searchColumns(ctx context.Context, input map[string]any) (any, error) {
	query, err := requiredString(input, "query")
	if err != nil {
		return nil, err
	}
	return rt.Snowflake.SearchColumns(ctx, query, stringInput(input, "database"), stringInput(input, "schema"), intInput(input, "limit"))
}

func (rt Runtime) cortexComplete(ctx context.Context, input map[string]any) (any, error) {
	model := firstNonBlank(stringInput(input, "model"), rt.Config.DefaultCortexModel)
	system := stringInput(input, "system")
	prompt, err := requiredString(input, "prompt")
	if err != nil {
		return nil, err
	}
	text, requestID, raw, err := rt.Snowflake.CortexChatComplete(ctx, model, system, prompt, intInput(input, "max_tokens"), floatInput(input, "temperature"))
	if err != nil {
		return nil, err
	}
	return map[string]any{"text": text, "model": model, "snowflake_request_id": requestID, "usage": raw["usage"]}, nil
}

func (rt Runtime) cortexAnalystMessage(ctx context.Context, input map[string]any) (any, error) {
	question, err := requiredString(input, "question")
	if err != nil {
		return nil, err
	}
	body := map[string]any{
		"messages": []map[string]any{{
			"role": "user",
			"content": []map[string]any{{
				"type": "text",
				"text": question,
			}},
		}},
	}
	if semanticModelFile := stringInput(input, "semantic_model_file"); semanticModelFile != "" {
		body["semantic_model_file"] = semanticModelFile
	}
	if semanticModel := stringInput(input, "semantic_model"); semanticModel != "" {
		if strings.HasPrefix(semanticModel, "@") {
			body["semantic_model_file"] = semanticModel
		} else {
			body["semantic_model"] = semanticModel
		}
	}
	if semanticView := stringInput(input, "semantic_view"); semanticView != "" {
		body["semantic_view"] = semanticView
	}
	if conversation, ok := input["conversation"]; ok {
		body["messages"] = conversation
	}
	raw, requestID, err := rt.Snowflake.PostJSON(ctx, "/api/v2/cortex/analyst/message", body)
	if err != nil {
		return nil, err
	}
	return map[string]any{
		"answer":               extractAnalystText(raw),
		"raw":                  raw,
		"snowflake_request_id": requestID,
	}, nil
}

func (rt Runtime) cortexSearchQuery(ctx context.Context, input map[string]any) (any, error) {
	service, err := requiredString(input, "service")
	if err != nil {
		return nil, err
	}
	query, err := requiredString(input, "query")
	if err != nil {
		return nil, err
	}
	database := firstNonBlank(stringInput(input, "database"), rt.Snowflake.Defaults()["database"])
	schemaName := firstNonBlank(stringInput(input, "schema"), rt.Snowflake.Defaults()["schema"])
	if database == "" || schemaName == "" {
		return nil, errors.New("database and schema are required for cortex_search_query")
	}
	body := map[string]any{
		"query": query,
		"limit": firstInt(intInput(input, "limit"), 10),
	}
	copyIfPresent(body, input, "columns")
	copyIfPresent(body, input, "filter")
	path := fmt.Sprintf("/api/v2/databases/%s/schemas/%s/cortex-search-services/%s:query", snowflake.URLPath(database), snowflake.URLPath(schemaName), snowflake.URLPath(service))
	raw, requestID, err := rt.Snowflake.PostJSON(ctx, path, body)
	if err != nil {
		return nil, err
	}
	return map[string]any{
		"results":              raw["results"],
		"raw":                  raw,
		"service":              service,
		"snowflake_request_id": requestID,
	}, nil
}

func (rt Runtime) semanticAsk(ctx context.Context, input map[string]any) (any, error) {
	question, err := requiredString(input, "question")
	if err != nil {
		return nil, err
	}
	contextBlob := stringInput(input, "semantic_context")
	if contextBlob == "" {
		contextBlob = compactJSON(input)
	}
	system, err := rt.Prompts.Render("semantic_ask.system", map[string]string{"question": question, "semantic_context": contextBlob})
	if err != nil {
		return nil, err
	}
	if stringInput(input, "semantic_model") != "" || stringInput(input, "semantic_view") != "" {
		return rt.cortexAnalystMessage(ctx, input)
	}
	model := firstNonBlank(stringInput(input, "model"), rt.Config.DefaultCortexModel)
	if model == "" {
		return map[string]any{
			"answer":        "NEEDS_REVIEW",
			"reason":        "semantic_ask needs semantic_model, semantic_view, or SNOWFLAKE_CORTEX_MODEL",
			"prompt_source": rt.Prompts.Source(),
		}, nil
	}
	text, queryID, _, err := rt.Snowflake.CortexChatComplete(ctx, model, system, question, 0, 0)
	if err != nil {
		return nil, err
	}
	return map[string]any{
		"answer":        text,
		"model":         model,
		"query_id":      queryID,
		"prompt_keys":   []string{"semantic_ask.system", "semantic_ask.answer_policy"},
		"prompt_source": rt.Prompts.Source(),
	}, nil
}

func (rt Runtime) explainResult(ctx context.Context, input map[string]any) (any, error) {
	question := firstNonBlank(stringInput(input, "question"), "Explain this Snowflake result.")
	preview := firstNonBlank(stringInput(input, "result_preview"), compactJSON(input["result"]))
	system, err := rt.Prompts.Render("explain_result.system", map[string]string{"question": question, "result_preview": preview})
	if err != nil {
		return nil, err
	}
	model := firstNonBlank(stringInput(input, "model"), rt.Config.DefaultCortexModel)
	if model == "" {
		return map[string]any{
			"answer":        "NEEDS_REVIEW",
			"reason":        "explain_result needs model or SNOWFLAKE_CORTEX_MODEL",
			"prompt_source": rt.Prompts.Source(),
		}, nil
	}
	text, queryID, _, err := rt.Snowflake.CortexChatComplete(ctx, model, "", system, 0, 0)
	if err != nil {
		return nil, err
	}
	return map[string]any{"answer": text, "model": model, "query_id": queryID, "prompt_source": rt.Prompts.Source()}, nil
}

func (rt Runtime) investigateMetricChange(_ context.Context, input map[string]any) (any, error) {
	metricName, err := requiredString(input, "metric_name")
	if err != nil {
		return nil, err
	}
	timeWindow, err := requiredString(input, "time_window")
	if err != nil {
		return nil, err
	}
	system, err := rt.Prompts.Render("investigate_metric_change.system", map[string]string{"metric_name": metricName, "time_window": timeWindow})
	if err != nil {
		return nil, err
	}
	return map[string]any{
		"status":      "planned",
		"metric_name": metricName,
		"time_window": timeWindow,
		"prompt":      system,
		"recommended_calls": []map[string]any{
			{"capability": "search_columns", "input": map[string]any{"query": metricName, "limit": 25}},
			{"capability": "semantic_ask", "input": map[string]any{"question": "Define " + metricName + " for " + timeWindow}},
			{"capability": "query_readonly", "input": map[string]any{"sql": "SELECT /* fill bounded metric query */ 1", "max_rows": 100}},
		},
		"prompt_source": rt.Prompts.Source(),
	}, nil
}

func requiredString(input map[string]any, key string) (string, error) {
	value := stringInput(input, key)
	if value == "" {
		return "", fmt.Errorf("%s is required", key)
	}
	return value, nil
}

func stringInput(input map[string]any, key string) string {
	if input == nil {
		return ""
	}
	switch value := input[key].(type) {
	case string:
		return strings.TrimSpace(value)
	default:
		return ""
	}
}

func intInput(input map[string]any, key string) int {
	switch value := input[key].(type) {
	case int:
		return value
	case float64:
		return int(value)
	case json.Number:
		n, _ := value.Int64()
		return int(n)
	default:
		return 0
	}
}

func floatInput(input map[string]any, key string) float64 {
	switch value := input[key].(type) {
	case float64:
		return value
	case float32:
		return float64(value)
	case int:
		return float64(value)
	case json.Number:
		n, _ := value.Float64()
		return n
	default:
		return 0
	}
}

func firstNonBlank(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func firstInt(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func copyIfPresent(dst map[string]any, src map[string]any, key string) {
	if value, ok := src[key]; ok {
		dst[key] = value
	}
}

func compactJSON(value any) string {
	data, err := json.Marshal(value)
	if err != nil {
		return ""
	}
	return string(data)
}

func extractAnalystText(raw map[string]any) string {
	if text, ok := raw["text"].(string); ok {
		return text
	}
	if message, ok := raw["message"].(string); ok {
		return message
	}
	if message, ok := raw["message"].(map[string]any); ok {
		if content, ok := message["content"].([]any); ok {
			parts := make([]string, 0, len(content))
			for _, item := range content {
				block, ok := item.(map[string]any)
				if !ok || block["type"] != "text" {
					continue
				}
				if text := valueString(block["text"]); text != "" {
					parts = append(parts, text)
				}
			}
			return strings.Join(parts, "\n")
		}
	}
	data, _ := json.Marshal(raw)
	return string(data)
}

func valueString(v any) string {
	switch value := v.(type) {
	case string:
		return strings.TrimSpace(value)
	default:
		data, _ := json.Marshal(value)
		return strings.TrimSpace(string(data))
	}
}
