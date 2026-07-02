package share

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestTruncatePreviewRedactAndRenderHTML(t *testing.T) {
	long := strings.Repeat("x", PreviewLimit+20)
	got := TruncatePreview(long)
	if len(got) <= PreviewLimit || !strings.Contains(got, "[truncated]") {
		t.Fatalf("expected truncated marker, got %q", got)
	}

	bundle := &Bundle{
		Version:     BundleVersion,
		WorkflowID:  "run-1",
		GeneratedAt: "2026-04-08T09:00:00Z",
		Nodes: []BundleNode{{
			ID:            "exec-1",
			Agent:         "agent",
			Func:          "func",
			Status:        "succeeded",
			InputPreview:  `{"secret":"<script>"}`,
			OutputPreview: "ok",
		}},
	}
	Redact(bundle)
	if bundle.Nodes[0].InputPreview != "[redacted]" || bundle.Nodes[0].OutputPreview != "[redacted]" {
		t.Fatalf("expected previews to be redacted: %+v", bundle.Nodes[0])
	}

	html, err := RenderHTML(bundle)
	if err != nil {
		t.Fatalf("RenderHTML returned error: %v", err)
	}
	body := string(html)
	if !strings.Contains(body, "[redacted]") {
		t.Fatalf("rendered HTML should contain redacted bundle: %s", body)
	}
	if strings.Contains(body, "</script") && strings.Contains(body, "__BUNDLE_JSON__") {
		t.Fatalf("bundle placeholder was not replaced")
	}
}

func TestDemoBundleIsSerializableAndRedactable(t *testing.T) {
	bundle := DemoBundle()
	if bundle.Version != BundleVersion || bundle.WorkflowID == "" || len(bundle.Nodes) == 0 || len(bundle.Edges) == 0 {
		t.Fatalf("demo bundle missing expected structure: %+v", bundle)
	}
	if bundle.Title == "" || bundle.GeneratedAt == "" || bundle.Totals.Agents != len(bundle.Nodes) {
		t.Fatalf("demo bundle missing summary fields: %+v", bundle)
	}
	if _, err := json.Marshal(bundle); err != nil {
		t.Fatalf("demo bundle should marshal: %v", err)
	}

	Redact(bundle)
	for _, node := range bundle.Nodes {
		if node.InputPreview != "" && node.InputPreview != "[redacted]" {
			t.Fatalf("input preview was not redacted: %+v", node)
		}
		if node.OutputPreview != "" && node.OutputPreview != "[redacted]" {
			t.Fatalf("output preview was not redacted: %+v", node)
		}
	}
}
