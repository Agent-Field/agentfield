package harness

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestCodeAFLiveInfoContractB2(t *testing.T) {
	t.Setenv("AFORGE_BIN", "")
	for _, name := range []string{ProviderCodeAF, ProviderAforge} {
		t.Run(name, func(t *testing.T) {
			provider, err := BuildProvider(name, "")
			require.NoError(t, err)
			live, ok := provider.(LiveProvider)
			require.True(t, ok)
			raw, err := json.Marshal(live.LiveInfo())
			require.NoError(t, err)
			require.Equal(t, []byte(`{"harness":"codeaf","steerable":true}`), raw)
		})
	}
}

func TestRegisterProviderContractB1(t *testing.T) {
	const name = "example-harness"
	var bin string
	factory := func(binPath string) Provider {
		bin = binPath
		return NewCodeAFProvider(binPath)
	}
	require.Error(t, RegisterProvider("  ", factory))
	require.Error(t, RegisterProvider(name, nil))
	for _, builtin := range []string{
		"codeaf", "aforge", "claude-code", "codex", "gemini", "opencode", "pi", "omp", "cursor",
	} {
		require.Error(t, RegisterProvider(builtin, factory))
	}
	require.NoError(t, RegisterProvider(name, factory))
	t.Cleanup(func() { UnregisterProvider(name) })
	require.Error(t, RegisterProvider(name, factory))
	provider, err := BuildProvider(name, "example-binary")
	require.NoError(t, err)
	require.NotNil(t, provider)
	require.Equal(t, "example-binary", bin)
	UnregisterProvider(name)
	_, err = BuildProvider(name, "")
	require.EqualError(t, err, `unknown harness provider: "example-harness" (supported: codeaf, aforge, claude-code, codex, gemini, opencode, pi, omp, cursor)`)
}
