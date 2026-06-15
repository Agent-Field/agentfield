package capabilities

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/Agent-Field/agentfield/integrations/databricks/node/internal/config"
	"github.com/Agent-Field/agentfield/integrations/databricks/node/internal/databricks"
	"github.com/Agent-Field/agentfield/integrations/databricks/node/internal/prompts"
	afagent "github.com/Agent-Field/agentfield/sdk/go/agent"
)

type Runtime struct {
	Config     config.Config
	Databricks *databricks.Client
	Prompts    *prompts.Store
}

func Register(a *afagent.Agent, rt Runtime) {
	a.RegisterReasoner("health", rt.health, afagent.WithDescription("Report Databricks node configuration health"), afagent.WithInputSchema(schema(`{"type":"object","additionalProperties":false}`)))
	a.RegisterReasoner("query_readonly", rt.queryReadOnly, afagent.WithDescription("Run a guarded read-only Databricks SQL statement"))
	a.RegisterReasoner("describe_table", rt.describeTable, afagent.WithDescription("Describe a Databricks Unity Catalog table or view"))
	a.RegisterReasoner("search_columns", rt.searchColumns, afagent.WithDescription("Search Unity Catalog information_schema columns"))
	a.RegisterReasoner("ai_query", rt.aiQuery, afagent.WithDescription("Call Databricks AI Functions ai_query through SQL"))
	a.RegisterReasoner("invoke_serving_endpoint", rt.invokeServingEndpoint, afagent.WithDescription("Invoke a Databricks Model Serving endpoint"))
	a.RegisterReasoner("explain_result", rt.explainResult, afagent.WithDescription("Explain a query result using Databricks AI Functions"))
	a.RegisterReasoner("investigate_metric_change", rt.investigateMetricChange, afagent.WithDescription("Plan a bounded Databricks metric-change investigation"))
	a.RegisterReasoner("handle_databricks_event", rt.handleDatabricksEvent, afagent.WithDescription("Receive Databricks trigger events and route to Databricks capabilities"), afagent.WithAcceptsWebhook("true"))
}

func schema(raw string) json.RawMessage {
	return json.RawMessage(raw)
}

func (rt Runtime) health(_ context.Context, _ map[string]any) (any, error) {
	defaults := map[string]string{}
	if rt.Databricks != nil {
		defaults = rt.Databricks.Defaults()
	}
	return map[string]any{
		"status":         "ok",
		"node_id":        rt.Config.NodeID,
		"prompt_source":  rt.Prompts.Source(),
		"prompt_version": rt.Prompts.Version(),
		"databricks":     defaults,
	}, nil
}

func (rt Runtime) queryReadOnly(ctx context.Context, input map[string]any) (any, error) {
	sql, err := requiredString(input, "sql")
	if err != nil {
		return nil, err
	}
	return rt.Databricks.QueryReadOnly(ctx, databricks.QueryRequest{
		SQL:            sql,
		Catalog:        stringInput(input, "catalog"),
		Schema:         stringInput(input, "schema"),
		WarehouseID:    stringInput(input, "warehouse_id"),
		MaxRows:        intInput(input, "max_rows"),
		TimeoutSeconds: intInput(input, "timeout_seconds"),
		Parameters:     sqlParameters(input["parameters"]),
	})
}

func (rt Runtime) describeTable(ctx context.Context, input map[string]any) (any, error) {
	object := firstNonBlank(stringInput(input, "object"), stringInput(input, "table"))
	if object == "" {
		return nil, errors.New("object is required")
	}
	return rt.Databricks.DescribeTable(ctx, object)
}

func (rt Runtime) searchColumns(ctx context.Context, input map[string]any) (any, error) {
	query, err := requiredString(input, "query")
	if err != nil {
		return nil, err
	}
	return rt.Databricks.SearchColumns(ctx, query, stringInput(input, "catalog"), stringInput(input, "schema"), intInput(input, "limit"))
}

func (rt Runtime) aiQuery(ctx context.Context, input map[string]any) (any, error) {
	prompt, err := requiredString(input, "prompt")
	if err != nil {
		return nil, err
	}
	return rt.Databricks.AIQuery(ctx, firstNonBlank(stringInput(input, "endpoint"), stringInput(input, "model")), prompt, stringInput(input, "return_type"), boolInput(input, "fail_on_error"))
}

func (rt Runtime) invokeServingEndpoint(ctx context.Context, input map[string]any) (any, error) {
	endpoint, err := requiredString(input, "endpoint")
	if err != nil {
		return nil, err
	}
	body, ok := input["body"]
	if !ok {
		body = map[string]any{}
	}
	return rt.Databricks.InvokeServingEndpoint(ctx, endpoint, body)
}

func (rt Runtime) explainResult(ctx context.Context, input map[string]any) (any, error) {
	question := firstNonBlank(stringInput(input, "question"), "Explain this Databricks query result.")
	preview := firstNonBlank(stringInput(input, "result_preview"), compactJSON(input["result"]))
	prompt, err := rt.Prompts.Render("explain_result.ai_query_prompt", map[string]string{"question": question, "result_preview": preview})
	if err != nil {
		return nil, err
	}
	result, err := rt.Databricks.AIQuery(ctx, firstNonBlank(stringInput(input, "endpoint"), stringInput(input, "model")), prompt, "", false)
	if err != nil {
		return nil, err
	}
	return map[string]any{
		"answer":        firstResponse(result),
		"query_result":  result,
		"prompt_keys":   []string{"explain_result.ai_query_prompt"},
		"prompt_source": rt.Prompts.Source(),
	}, nil
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
	prompt, err := rt.Prompts.Render("investigate_metric_change.planning_prompt", map[string]string{"metric_name": metricName, "time_window": timeWindow})
	if err != nil {
		return nil, err
	}
	return map[string]any{
		"status":      "planned",
		"metric_name": metricName,
		"time_window": timeWindow,
		"prompt":      prompt,
		"recommended_calls": []map[string]any{
			{"capability": "search_columns", "input": map[string]any{"query": metricName, "limit": 25}},
			{"capability": "ai_query", "input": map[string]any{"prompt": prompt}},
			{"capability": "query_readonly", "input": map[string]any{"sql": "SELECT /* fill bounded metric query */ 1", "max_rows": 100}},
		},
		"prompt_source": rt.Prompts.Source(),
	}, nil
}

func (rt Runtime) handleDatabricksEvent(_ context.Context, input map[string]any) (any, error) {
	event := input
	if nested, ok := input["event"].(map[string]any); ok {
		event = nested
	}
	return map[string]any{
		"received":         true,
		"event_type":       firstNonBlank(stringInput(event, "event_type"), stringInput(event, "type"), stringInput(input, "event_type")),
		"event_id":         firstNonBlank(stringInput(event, "event_id"), stringInput(event, "id"), stringInput(input, "event_id")),
		"recommended_call": "query_readonly",
		"databricks":       event["databricks"],
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

func boolInput(input map[string]any, key string) bool {
	switch value := input[key].(type) {
	case bool:
		return value
	case string:
		return strings.EqualFold(strings.TrimSpace(value), "true")
	default:
		return false
	}
}

func sqlParameters(value any) []databricks.SQLParameter {
	raw, ok := value.([]any)
	if !ok {
		return nil
	}
	out := make([]databricks.SQLParameter, 0, len(raw))
	for _, item := range raw {
		m, ok := item.(map[string]any)
		if !ok {
			continue
		}
		name := stringInput(m, "name")
		if name == "" {
			continue
		}
		out = append(out, databricks.SQLParameter{
			Name:  name,
			Value: valueString(m["value"]),
			Type:  stringInput(m, "type"),
		})
	}
	return out
}

func firstNonBlank(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func compactJSON(value any) string {
	data, err := json.Marshal(value)
	if err != nil {
		return ""
	}
	return string(data)
}

func firstResponse(result databricks.QueryResult) string {
	if len(result.Rows) == 0 {
		return ""
	}
	for _, key := range []string{"response", "RESPONSE"} {
		if value, ok := result.Rows[0][key]; ok {
			return valueString(value)
		}
	}
	return valueString(result.Rows[0])
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
