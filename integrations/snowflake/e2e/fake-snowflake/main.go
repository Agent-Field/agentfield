package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"
	"strings"
)

func main() {
	addr := env("FAKE_SNOWFLAKE_LISTEN", ":19090")
	mux := http.NewServeMux()
	mux.HandleFunc("/api/v2/statements", handleStatements)
	mux.HandleFunc("/api/v2/cortex/v1/chat/completions", handleChatCompletions)
	mux.HandleFunc("/", handleREST)
	log.Printf("fake Snowflake listening on %s", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatal(err)
	}
}

func handleStatements(w http.ResponseWriter, r *http.Request) {
	requirePAT(w, r)
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		Statement string `json:"statement"`
	}
	_ = json.NewDecoder(r.Body).Decode(&req)
	upper := strings.ToUpper(req.Statement)
	switch {
	case strings.Contains(upper, "AGENTFIELD_EVENTS"):
		writeJSON(w, map[string]any{
			"statementHandle": "fake-events-query",
			"resultSetMetaData": map[string]any{"rowType": []map[string]string{
				{"name": "EVENT_ID"}, {"name": "EVENT_TYPE"}, {"name": "PAYLOAD"}, {"name": "OCCURRED_AT"},
			}},
			"data": [][]any{{"evt_fake_1", "snowflake.alert.fake", `{"metric":"revenue","delta":-12}`, "2026-06-11 12:00:00.000 -0700"}},
		})
	case strings.Contains(upper, "INFORMATION_SCHEMA.COLUMNS"):
		writeJSON(w, map[string]any{
			"statementHandle": "fake-columns-query",
			"resultSetMetaData": map[string]any{"rowType": []map[string]string{
				{"name": "TABLE_CATALOG"}, {"name": "TABLE_SCHEMA"}, {"name": "TABLE_NAME"}, {"name": "COLUMN_NAME"}, {"name": "DATA_TYPE"}, {"name": "COMMENT"},
			}},
			"data": [][]any{{"DEMO_DB", "PUBLIC", "REVENUE", "CUSTOMER_ID", "VARCHAR", "Customer identifier"}},
		})
	default:
		writeJSON(w, map[string]any{
			"statementHandle": "fake-readonly-query",
			"resultSetMetaData": map[string]any{"rowType": []map[string]string{
				{"name": "ID"}, {"name": "AMOUNT"},
			}},
			"data": [][]any{{1, 42.5}},
		})
	}
}

func handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	requirePAT(w, r)
	w.Header().Set("X-Snowflake-Request-Id", "fake-cortex-request")
	writeJSON(w, map[string]any{
		"choices": []map[string]any{{
			"message": map[string]any{"content": "Fake Cortex explanation from Snowflake e2e."},
		}},
		"usage": map[string]any{"total_tokens": 7},
	})
}

func handleREST(w http.ResponseWriter, r *http.Request) {
	requirePAT(w, r)
	switch {
	case strings.Contains(r.URL.Path, "/cortex-search-services/") && strings.HasSuffix(r.URL.Path, ":query"):
		w.Header().Set("X-Snowflake-Request-Id", "fake-search-request")
		writeJSON(w, map[string]any{"results": []map[string]any{{"text": "search hit", "score": 0.98}}})
	case r.URL.Path == "/api/v2/cortex/analyst/message":
		w.Header().Set("X-Snowflake-Request-Id", "fake-analyst-request")
		writeJSON(w, map[string]any{
			"message": map[string]any{"content": []map[string]any{{"type": "text", "text": "Fake Analyst answer."}}},
		})
	default:
		http.NotFound(w, r)
	}
}

func requirePAT(w http.ResponseWriter, r *http.Request) {
	if r.Header.Get("Authorization") != "Bearer fake-pat" {
		http.Error(w, "missing fake PAT", http.StatusUnauthorized)
		return
	}
}

func writeJSON(w http.ResponseWriter, value any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(value)
}

func env(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}
