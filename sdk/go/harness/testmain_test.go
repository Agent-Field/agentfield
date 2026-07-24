package harness

import (
	"os"
	"path/filepath"
	"testing"
)

func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "harness-test-bin-")
	if err != nil {
		panic(err)
	}
	defer os.RemoveAll(dir)

	stubs := map[string]string{
		"codex": `#!/bin/sh
printf '%s\n' '{"type":"result","result":"stub codex result","session_id":"stub-session","num_turns":1}'
`,
		"claude": `#!/bin/sh
printf '%s\n' '{"type":"result","result":"stub claude result","session_id":"stub-session","num_turns":1}'
`,
		"gemini": `#!/bin/sh
printf '%s\n' 'stub gemini result'
`,
		"opencode": `#!/bin/sh
printf '%s\n' 'stub opencode result'
`,
	}

	for name, content := range stubs {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o755); err != nil {
			panic(err)
		}
	}

	if err := os.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH")); err != nil {
		panic(err)
	}

	os.Exit(m.Run())
}
