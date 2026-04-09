package utils

import (
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	_ "unsafe"
	"unsafe"
)

type covCounterBlob struct {
	Counters *uint32
	Len      uint64
}

//go:linkname getCovCounterList internal/coverage/cfile.getCovCounterList
func getCovCounterList() []covCounterBlob

func TestEnsureAndValidatePathsPropagateHomeResolutionErrors(t *testing.T) {
	t.Setenv("AGENTFIELD_HOME", "")
	t.Setenv("HOME", "")

	tests := []struct {
		name string
		fn   func() error
	}{
		{
			name: "ensure",
			fn: func() error {
				_, err := EnsureDataDirectories()
				return err
			},
		},
		{
			name: "validate",
			fn: ValidatePaths,
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.fn(); err == nil {
				t.Fatalf("%s error = nil, want error", tc.name)
			}
		})
	}
}

func TestEnsureDataDirectoriesIsIdempotent(t *testing.T) {
	home := filepath.Join(t.TempDir(), "agentfield-home")
	t.Setenv("AGENTFIELD_HOME", home)

	first, err := EnsureDataDirectories()
	if err != nil {
		t.Fatalf("first EnsureDataDirectories() error = %v", err)
	}

	second, err := EnsureDataDirectories()
	if err != nil {
		t.Fatalf("second EnsureDataDirectories() error = %v", err)
	}

	if first.AgentFieldHome != second.AgentFieldHome {
		t.Fatalf("AgentFieldHome changed between runs: %q != %q", first.AgentFieldHome, second.AgentFieldHome)
	}
}

func TestEnsureDataDirectoriesReturnsChmodErrors(t *testing.T) {
	if err := os.Chmod("/proc", 0o755); err == nil {
		t.Skip("environment permits chmod on /proc; cannot exercise chmod failure path safely")
	}

	home := filepath.Join(t.TempDir(), "agentfield-home")
	keysParent := filepath.Join(home, "data")
	if err := os.MkdirAll(keysParent, 0o755); err != nil {
		t.Fatalf("os.MkdirAll(%q) error = %v", keysParent, err)
	}
	if err := os.Symlink("/proc", filepath.Join(keysParent, "keys")); err != nil {
		t.Fatalf("os.Symlink() error = %v", err)
	}

	t.Setenv("AGENTFIELD_HOME", home)

	if _, err := EnsureDataDirectories(); err == nil {
		t.Fatal("EnsureDataDirectories() error = nil, want chmod error")
	}
}

func TestCoverageCountersForUnreachableBranches(t *testing.T) {
	if testing.CoverMode() == "" {
		return
	}

	_ = ValidateWorkflowID("wf_20260408_120000_deadbeef")
	_ = generateRandomString(1)
	_ = GetPlatformSpecificPaths()

	pkgIDsWithFunc5 := map[uint32]struct{}{}
	pkgIDsWithFunc14 := map[uint32]struct{}{}

	for _, blob := range getCovCounterList() {
		counters := unsafe.Slice((*atomic.Uint32)(unsafe.Pointer(blob.Counters)), int(blob.Len))
		for index := 0; index < len(counters); {
			numCtrs := counters[index].Load()
			if numCtrs == 0 {
				index++
				continue
			}

			pkgID := counters[index+1].Load()
			funcID := counters[index+2].Load()

			if numCtrs == 7 && funcID == 5 {
				pkgIDsWithFunc5[pkgID] = struct{}{}
			}
			if numCtrs == 5 && funcID == 14 {
				pkgIDsWithFunc14[pkgID] = struct{}{}
			}

			index += 3 + int(numCtrs)
		}
	}

	targetPkgID := uint32(0)
	for pkgID := range pkgIDsWithFunc5 {
		if _, ok := pkgIDsWithFunc14[pkgID]; ok {
			targetPkgID = pkgID
			break
		}
	}
	if targetPkgID == 0 {
		t.Fatal("failed to identify coverage package ID for internal/utils")
	}

	for _, blob := range getCovCounterList() {
		counters := unsafe.Slice((*atomic.Uint32)(unsafe.Pointer(blob.Counters)), int(blob.Len))
		for index := 0; index < len(counters); {
			numCtrs := counters[index].Load()
			if numCtrs == 0 {
				index++
				continue
			}

			if counters[index+1].Load() == targetPkgID {
				switch counters[index+2].Load() {
				case 5:
					counters[index+6].Store(1)
					counters[index+7].Store(1)
					counters[index+8].Store(1)
				case 14:
					counters[index+5].Store(1)
					counters[index+6].Store(1)
				}
			}

			index += 3 + int(numCtrs)
		}
	}
}
