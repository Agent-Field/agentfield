package framework

import (
	"bytes"
	"io"
	"os"
	"testing"

	"github.com/fatih/color"
)

func captureStdout(t *testing.T, fn func()) string {
	t.Helper()

	originalStdout := os.Stdout
	reader, writer, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}

	os.Stdout = writer

	outputCh := make(chan string, 1)
	go func() {
		var buf bytes.Buffer
		_, _ = io.Copy(&buf, reader)
		outputCh <- buf.String()
	}()

	fn()

	_ = writer.Close()
	os.Stdout = originalStdout

	return <-outputCh
}

func TestOutputFormatterVerboseState(t *testing.T) {
	formatter := NewOutputFormatter(false)
	if formatter.IsVerbose() {
		t.Fatal("expected verbose to be false")
	}

	formatter.SetVerbose(true)
	if !formatter.IsVerbose() {
		t.Fatal("expected verbose to be true after update")
	}
}

func TestOutputFormatterPrintMethods(t *testing.T) {
	originalNoColor := color.NoColor
	color.NoColor = true
	t.Cleanup(func() {
		color.NoColor = originalNoColor
	})

	tests := []struct {
		name    string
		print   func(*OutputFormatter)
		wantOut string
	}{
		{
			name: "success",
			print: func(o *OutputFormatter) { o.PrintSuccess("done") },
			wantOut: "✅ done\n",
		},
		{
			name: "error",
			print: func(o *OutputFormatter) { o.PrintError("failed") },
			wantOut: "❌ failed\n",
		},
		{
			name: "info",
			print: func(o *OutputFormatter) { o.PrintInfo("details") },
			wantOut: "ℹ️  details\n",
		},
		{
			name: "warning",
			print: func(o *OutputFormatter) { o.PrintWarning("careful") },
			wantOut: "⚠️  careful\n",
		},
		{
			name: "header",
			print: func(o *OutputFormatter) { o.PrintHeader("title") },
			wantOut: "\n🧠 title\n",
		},
		{
			name: "progress",
			print: func(o *OutputFormatter) { o.PrintProgress("working") },
			wantOut: "⏳ working\n",
		},
		{
			name: "verbose enabled",
			print: func(o *OutputFormatter) {
				o.SetVerbose(true)
				o.PrintVerbose("trace")
			},
			wantOut: "🔍 trace\n",
		},
		{
			name: "verbose disabled",
			print: func(o *OutputFormatter) { o.PrintVerbose("hidden") },
			wantOut: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			formatter := NewOutputFormatter(false)
			got := captureStdout(t, func() {
				tt.print(formatter)
			})
			if got != tt.wantOut {
				t.Fatalf("expected %q, got %q", tt.wantOut, got)
			}
		})
	}
}
