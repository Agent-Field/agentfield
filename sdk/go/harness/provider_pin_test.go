package harness

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// #1181: an integrator that already sets a harness binary must be able to
// move to CodeAF without losing that binary, while the legacy adapter stays
// reachable when asked for by name.
func TestBuildProvider_BinaryOverrideSelection(t *testing.T) {
	const bin = "/opt/custom/harness"
	tests := []struct {
		name     string
		provider string
		bin      string
		aforge   string
		want     any
		wantBin  string
	}{
		{name: "implicit provider with binary keeps the legacy pin", bin: bin, want: &AforgeProvider{}, wantBin: bin},
		{name: "implicit provider with AFORGE_BIN keeps the legacy pin", aforge: bin, want: &AforgeProvider{}, wantBin: bin},
		{name: "explicit codeaf keeps the custom binary", provider: ProviderCodeAF, bin: bin, want: &CodeAFProvider{}, wantBin: bin},
		{name: "explicit codeaf wins over AFORGE_BIN", provider: ProviderCodeAF, aforge: bin, want: &CodeAFProvider{}},
		{name: "explicit aforge with binary stays legacy", provider: ProviderAforge, bin: bin, want: &AforgeProvider{}, wantBin: bin},
		{name: "implicit provider without overrides is CodeAF", want: &CodeAFProvider{}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv(ProviderEnvVar, "")
			t.Setenv("AFORGE_BIN", tt.aforge)

			provider, err := BuildProvider(tt.provider, tt.bin)
			require.NoError(t, err)
			require.IsType(t, tt.want, provider)

			switch p := provider.(type) {
			case *CodeAFProvider:
				assert.Equal(t, tt.wantBin, p.BinPath)
				assert.False(t, p.alias)
			case *AforgeProvider:
				assert.Equal(t, tt.wantBin, p.BinPath)
			}
		})
	}
}

func TestBuildProvider_ProviderEnvOverridesLegacyPin(t *testing.T) {
	t.Setenv(ProviderEnvVar, ProviderCodeAF)
	t.Setenv("AFORGE_BIN", "")

	provider, err := BuildProvider("", "/opt/custom/harness")
	require.NoError(t, err)
	codeaf, ok := provider.(*CodeAFProvider)
	require.True(t, ok)
	assert.Equal(t, "/opt/custom/harness", codeaf.BinPath)
}

// The acceptance from #1181: CodeAF chosen through the factory with a custom
// binary runs that binary as a watched chat turn, so live events and steering
// keep working.
func TestBuildProvider_CodeAFWithCustomBinaryStaysWatched(t *testing.T) {
	isolateCodeAF(t)
	const bin = "/opt/custom/codeaf"

	provider, err := BuildProvider(ProviderCodeAF, bin)
	require.NoError(t, err)
	p, ok := provider.(*CodeAFProvider)
	require.True(t, ok)

	var resolvedWith string
	p.resolveBinary = func(_ context.Context, override string) (codeAFBinary, error) {
		resolvedWith = override
		return codeAFBinary{path: override}, nil
	}
	var cmd []string
	p.runCLI = func(_ context.Context, argv []string, _ map[string]string, _ string, _, _ int, _ []byte) (*CLIResult, error) {
		cmd = argv
		return &CLIResult{Stdout: `{"ok":true,"stop":"done","answer":"done"}`}, nil
	}

	steer := make(chan string)
	raw, err := p.Execute(context.Background(), "write it", Options{Live: &Live{Steer: steer}})
	require.NoError(t, err)
	assert.Equal(t, "done", raw.Result)
	assert.Equal(t, bin, resolvedWith)
	require.NotEmpty(t, cmd)
	assert.Equal(t, bin, cmd[0])
	assert.Equal(t, []string{"chat", "--once", "write it"}, cmd[1:4])
	assert.NotEmpty(t, flagValue(cmd, "--events"))
	assert.NotEmpty(t, flagValue(cmd, "--steer"))
}
