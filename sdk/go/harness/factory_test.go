package harness

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestResolveProviderName(t *testing.T) {
	tests := []struct {
		name     string
		explicit string
		env      string
		want     string
	}{
		{name: "default", want: ProviderCodeAF},
		{name: "environment", env: ProviderCodex, want: ProviderCodex},
		{name: "explicit wins", explicit: ProviderGemini, env: ProviderCodex, want: ProviderGemini},
		{name: "blank explicit", explicit: "   ", want: ProviderCodeAF},
		{name: "blank environment", env: "  ", want: ProviderCodeAF},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv(ProviderEnvVar, tt.env)
			assert.Equal(t, tt.want, ResolveProviderName(tt.explicit))
		})
	}
}

func TestBuildProvider_DefaultsToCodeAF(t *testing.T) {
	t.Setenv(ProviderEnvVar, "")
	provider, err := BuildProvider("", "")
	require.NoError(t, err)
	require.NotNil(t, provider)
	_, ok := provider.(*CodeAFProvider)
	assert.True(t, ok)
}

func TestBuildProvider_ExplicitProviders(t *testing.T) {
	tests := []struct {
		name string
		want any
	}{
		{name: ProviderCodeAF, want: &CodeAFProvider{}},
		{name: ProviderAforge, want: &AforgeProvider{}},
		{name: ProviderClaudeCode, want: &ClaudeCodeProvider{}},
		{name: ProviderOpenCode, want: &OpenCodeProvider{}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider, err := BuildProvider(tt.name, "test-bin")
			require.NoError(t, err)
			require.NotNil(t, provider)
			assert.IsType(t, tt.want, provider)
		})
	}
}

func TestBuildProvider_RejectsUnknownName(t *testing.T) {
	provider, err := BuildProvider("nope", "")
	assert.Nil(t, provider)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "nope")
	assert.True(t, strings.Contains(err.Error(), "supported: codeaf, aforge, claude-code, codex, gemini, opencode, pi, omp, cursor"))
}

// Issue #1181 regression: a generic harness-binary override (BinPath) must not
// drag an implicit provider onto the legacy aforge adapter. The default CodeAF
// provider is selected and honors the custom binary.
func TestBuildProvider_ImplicitWithCustomBinaryUsesCodeAF(t *testing.T) {
	isolateCodeAF(t)
	t.Setenv(ProviderEnvVar, "")
	const customBin = "/opt/custom/codeaf"
	provider, err := BuildProvider("", customBin)
	require.NoError(t, err)
	codeAF, ok := provider.(*CodeAFProvider)
	require.True(t, ok)
	assert.Equal(t, customBin, codeAF.BinPath)
	assert.False(t, codeAF.alias, "an implicit provider with a custom binary is CodeAF itself, not the aforge alias")

	// The runner path resolves the same way.
	r := NewRunner(Options{BinPath: customBin})
	p, err := r.buildProvider(r.DefaultOptions)
	require.NoError(t, err)
	codeAF, ok = p.(*CodeAFProvider)
	require.True(t, ok)
	assert.Equal(t, customBin, codeAF.BinPath)
}

// The compatibility pin stays: AFORGE_BIN alone still selects the legacy
// adapter for an implicit default, and an explicit aforge selection with a
// custom binary remains available for upgrades from pre-CodeAF SDKs.
func TestBuildProvider_LegacySelectionRemainsAvailable(t *testing.T) {
	isolateCodeAF(t)
	t.Setenv(ProviderEnvVar, "")
	const legacyBin = "/opt/legacy/aforge"

	t.Run("implicit with AFORGE_BIN pins legacy", func(t *testing.T) {
		t.Setenv("AFORGE_BIN", legacyBin)
		provider, err := BuildProvider("", "")
		require.NoError(t, err)
		aforge, ok := provider.(*AforgeProvider)
		require.True(t, ok)
		assert.Equal(t, legacyBin, aforge.BinPath)
	})

	t.Run("explicit aforge with a custom binary stays legacy", func(t *testing.T) {
		t.Setenv("AFORGE_BIN", "")
		provider, err := BuildProvider(ProviderAforge, legacyBin)
		require.NoError(t, err)
		aforge, ok := provider.(*AforgeProvider)
		require.True(t, ok)
		assert.Equal(t, legacyBin, aforge.BinPath)
	})
}

// Issue #1181 acceptance through the factory and an owned execution: selecting
// CodeAFProvider with a custom binary retains live watching and steering — the
// run streams events while it is still running and a Steer note reaches its
// steer file mid-run, all through the custom binary.
func TestCodeAFWithCustomBinaryRetainsLiveWatchingAndSteering(t *testing.T) {
	isolateCodeAF(t)
	t.Setenv(ProviderEnvVar, "")
	const customBin = "/opt/custom/codeaf"
	provider, err := BuildProvider("", customBin)
	require.NoError(t, err)
	p := provider.(*CodeAFProvider)
	p.resolveBinary = func(context.Context, string) (codeAFBinary, error) {
		return codeAFBinary{path: p.BinPath}, nil
	}

	steer := make(chan string, 1)
	var mu sync.Mutex
	var lines []string
	sawFirstWhileRunning := make(chan struct{})
	var begun, ended int
	live := &Live{
		Steer: steer,
		Begin: func() { begun++ },
		End:   func() { ended++ },
		Event: func(raw json.RawMessage) {
			mu.Lock()
			lines = append(lines, string(raw))
			n := len(lines)
			mu.Unlock()
			if n == 1 {
				close(sawFirstWhileRunning)
			}
		},
	}
	first := `{"kind":"assistant", "turn":1,"text":"looking at main.go"}`
	p.runCLI = func(ctx context.Context, cmd []string, env map[string]string, cwd string, timeout, idle int, stdin []byte) (*CLIResult, error) {
		// The custom binary runs the watched chat session.
		assert.Equal(t, customBin, cmd[0])
		events, steerPath := flagValue(cmd, "--events"), flagValue(cmd, "--steer")
		require.Equal(t, []string{"chat", "--once", "write it"}, cmd[1:4])
		require.Contains(t, cmd, "--json")
		require.NotEmpty(t, events)
		require.NotEmpty(t, steerPath)
		require.NotEmpty(t, cwd)
		appendLine(t, events, first+"\n")
		select {
		case <-sawFirstWhileRunning:
		case <-time.After(3 * time.Second):
			t.Fatal("the first event was not handed over while the run was working")
		}
		steer <- "also add a doc comment"
		var note string
		for deadline := time.Now().Add(3 * time.Second); time.Now().Before(deadline); time.Sleep(20 * time.Millisecond) {
			raw, _ := os.ReadFile(steerPath)
			if len(raw) > 0 {
				note = string(raw)
				break
			}
		}
		require.Equal(t, `{"text":"also add a doc comment"}`+"\n", note)
		appendLine(t, events, `{"kind":"note","text":"steered: also add a doc comment"}`)
		return &CLIResult{Stdout: `{"ok":true,"stop":"done","answer":"done"}`}, nil
	}

	raw, err := p.Execute(context.Background(), "write it", Options{Live: live})
	require.NoError(t, err)
	assert.Equal(t, "done", raw.Result)
	assert.Equal(t, []string{first, `{"kind":"note","text":"steered: also add a doc comment"}`}, lines)
	assert.Equal(t, 1, begun)
	assert.Equal(t, 1, ended)
}
