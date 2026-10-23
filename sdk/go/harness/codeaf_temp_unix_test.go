//go:build !windows

package harness

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// A5: a foreign-owned base or child must be rejected, even if writable.
func TestCodeAFTempRejectsForeignOwner(t *testing.T) {
	for _, surface := range []string{"resolution", "doctor", "download", "state"} {
		for _, level := range []string{"base", "leaf"} {
			t.Run(surface+"/"+level, func(t *testing.T) {
				isolateCodeAF(t)
				blockCodeAFHome(t)
				logs := captureCodeAFLogs(t)
				dir := t.TempDir()
				t.Setenv("PATH", dir)
				safe := writeCodeAFCLI(t, filepath.Join(dir, "codeaf"), "{}", 0)
				_, binary := codeAFManagedPaths()
				base := filepath.Dir(filepath.Dir(binary))
				leaf := filepath.Dir(binary)
				if surface == "state" {
					leaf = filepath.Join(base, "state")
				}
				require.NoError(t, os.MkdirAll(leaf, 0o777))
				untrusted := base
				if level == "leaf" {
					untrusted = leaf
				}
				require.NoError(t, os.Chmod(untrusted, 0o777))
				if surface == "resolution" || surface == "doctor" {
					writeManagedCodeAF(t, binary, "codeaf v0.8.2 built planted")
				}
				ownerCheck := codeAFTempOwnedByCurrentUser
				codeAFTempOwnedByCurrentUser = func(info os.FileInfo) bool {
					return info.Name() != filepath.Base(untrusted) && ownerCheck(info)
				}
				t.Cleanup(func() { codeAFTempOwnedByCurrentUser = ownerCheck })
				switch surface {
				case "resolution":
					resolved, err := resolveCodeAF(context.Background(), "")
					require.NoError(t, err)
					assert.Equal(t, safe, resolved.path)
				case "doctor":
					reports, err := Doctor(context.Background(), DoctorOptions{Providers: []string{ProviderCodeAF}})
					require.NoError(t, err)
					assert.Equal(t, safe, reports[0].Binary)
				case "download":
					// A failed trust check must win over even constructing a request.
					t.Setenv("AGENTFIELD_CODEAF_BASE_URL", "http://[invalid")
					_, err := downloadManagedCodeAF(context.Background())
					require.ErrorContains(t, err, "not owned")
				case "state":
					_, err := codeAFEnvironment(Options{}, "")
					require.ErrorContains(t, err, "not owned")
				}
				assert.Equal(t, 1, strings.Count(logs.String(), "WARNING"), logs.String())
				info, err := os.Lstat(untrusted)
				require.NoError(t, err)
				assert.Equal(t, os.FileMode(0o777), info.Mode().Perm(), "never chmod a foreign-owned directory")
			})
		}
	}
}

type codeAFTempStatInfo struct {
	os.FileInfo
	stat any
}

func (info codeAFTempStatInfo) Sys() any { return info.stat }

func TestCodeAFTempOwnerCheck(t *testing.T) {
	info, err := os.Lstat(t.TempDir())
	require.NoError(t, err)
	assert.True(t, codeAFTempOwnedByCurrentUser(info))
	assert.False(t, codeAFTempOwnedByCurrentUser(codeAFTempStatInfo{info, &syscall.Stat_t{Uid: uint32(os.Getuid()) + 1}}))
	assert.False(t, codeAFTempOwnedByCurrentUser(codeAFTempStatInfo{info, nil}))
}
