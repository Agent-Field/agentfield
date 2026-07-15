// Package share builds a self-contained, offline HTML artifact for a single
// AgentField workflow run. The bundle schema is versioned so the embedded
// template and any future hosted share server can evolve independently.
package share

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"fmt"
	"strings"
)

// BundleVersion is the schema version of the shareable run bundle. Bump it on
// any breaking change to the JSON shape so consumers (template, share server)
// can gate on it.
const BundleVersion = 1

// PreviewLimit caps input/output preview strings at 2 KiB per node to keep the
// artifact small and avoid leaking large payloads into a shareable file.
const PreviewLimit = 2048

//go:embed template.html
var templateHTML string

// Bundle is the versioned, self-contained description of a single run. It is
// both inlined into the HTML artifact and used as the POST body for the
// (optional) hosted share server.
type Bundle struct {
	Version     int          `json:"version"`
	WorkflowID  string       `json:"workflow_id"`
	Title       string       `json:"title,omitempty"`
	GeneratedAt string       `json:"generated_at"`
	Totals      BundleTotals `json:"totals"`
	Nodes       []BundleNode `json:"nodes"`
	Edges       []BundleEdge `json:"edges"`
}

// BundleTotals is the run-level summary strip. Cost is a pointer so it renders
// as JSON null (not 0) when the control plane does not track cost.
type BundleTotals struct {
	Agents     int      `json:"agents"`
	DurationMS int64    `json:"duration_ms"`
	CostUSD    *float64 `json:"cost_usd"`
}

// BundleNode is one execution in the run. Model and CostUSD are pointers so
// absent data serializes as null rather than a fabricated zero value.
type BundleNode struct {
	ID            string   `json:"id"`
	Agent         string   `json:"agent"`
	Func          string   `json:"func"`
	Status        string   `json:"status"`
	StartedAt     string   `json:"started_at"`
	EndedAt       string   `json:"ended_at"`
	DurationMS    *int64   `json:"duration_ms"`
	CostUSD       *float64 `json:"cost_usd"`
	Model         *string  `json:"model"`
	InputPreview  string   `json:"input_preview"`
	OutputPreview string   `json:"output_preview"`
	Error         *string  `json:"error"`
}

// BundleEdge is a parent->child execution relationship in the DAG.
type BundleEdge struct {
	From string `json:"from"`
	To   string `json:"to"`
}

// TruncatePreview clips a preview string to PreviewLimit bytes, appending an
// ellipsis marker when it had to cut. It is UTF-8 safe (it will not split a
// multi-byte rune).
func TruncatePreview(s string) string {
	if len(s) <= PreviewLimit {
		return s
	}
	cut := PreviewLimit
	// Walk back to a rune boundary.
	for cut > 0 && (s[cut]&0xC0) == 0x80 {
		cut--
	}
	return s[:cut] + "\n… [truncated]"
}

// Redact replaces every input/output preview with a fixed marker so a shared
// artifact carries no payload data. Errors and structure are preserved.
func Redact(b *Bundle) {
	if b == nil {
		return
	}
	for i := range b.Nodes {
		if b.Nodes[i].InputPreview != "" {
			b.Nodes[i].InputPreview = "[redacted]"
		}
		if b.Nodes[i].OutputPreview != "" {
			b.Nodes[i].OutputPreview = "[redacted]"
		}
	}
}

// RenderHTML inlines the bundle JSON into the embedded template and returns a
// fully self-contained HTML document. The JSON is escaped for safe embedding
// inside a <script type="application/json"> block.
func RenderHTML(b *Bundle) ([]byte, error) {
	raw, err := json.Marshal(b)
	if err != nil {
		return nil, fmt.Errorf("marshal bundle: %w", err)
	}
	inline := escapeForScriptTag(raw)
	out := strings.Replace(templateHTML, "__BUNDLE_JSON__", inline, 1)
	return []byte(out), nil
}

// escapeForScriptTag makes JSON safe to place inside an inline
// <script type="application/json"> element: it neutralizes the "</script"
// sequence and escapes HTML-significant runes so the payload can never break
// out of the script block or introduce markup.
func escapeForScriptTag(raw []byte) string {
	var out bytes.Buffer
	for _, r := range string(raw) {
		switch r {
		case '<':
			out.WriteString(`\u003c`)
		case '>':
			out.WriteString(`\u003e`)
		case '&':
			out.WriteString(`\u0026`)
		case '\u2028':
			out.WriteString(`\u2028`)
		case '\u2029':
			out.WriteString(`\u2029`)
		default:
			out.WriteRune(r)
		}
	}
	return out.String()
}
