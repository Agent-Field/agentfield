package utils

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestGetAgentFieldDataDirectoriesUsesEnvOverride(t *testing.T) {
	t.Setenv("AGENTFIELD_HOME", filepath.Join(t.TempDir(), "agentfield-home"))

	dirs, err := GetAgentFieldDataDirectories()
	if err != nil {
		t.Fatalf("GetAgentFieldDataDirectories() error = %v", err)
	}

	if dirs.AgentFieldHome != os.Getenv("AGENTFIELD_HOME") {
		t.Fatalf("AgentFieldHome = %q, want %q", dirs.AgentFieldHome, os.Getenv("AGENTFIELD_HOME"))
	}
	if dirs.DataDir != filepath.Join(dirs.AgentFieldHome, "data") {
		t.Fatalf("DataDir = %q", dirs.DataDir)
	}
	if dirs.DatabaseDir != dirs.DataDir {
		t.Fatalf("DatabaseDir = %q, want %q", dirs.DatabaseDir, dirs.DataDir)
	}
	if dirs.KeysDir != filepath.Join(dirs.DataDir, "keys") {
		t.Fatalf("KeysDir = %q", dirs.KeysDir)
	}
	if dirs.DIDRegistriesDir != filepath.Join(dirs.DataDir, "did_registries") {
		t.Fatalf("DIDRegistriesDir = %q", dirs.DIDRegistriesDir)
	}
	if dirs.VCsDir != filepath.Join(dirs.DataDir, "vcs") {
		t.Fatalf("VCsDir = %q", dirs.VCsDir)
	}
	if dirs.VCsExecutionsDir != filepath.Join(dirs.VCsDir, "executions") {
		t.Fatalf("VCsExecutionsDir = %q", dirs.VCsExecutionsDir)
	}
	if dirs.VCsWorkflowsDir != filepath.Join(dirs.VCsDir, "workflows") {
		t.Fatalf("VCsWorkflowsDir = %q", dirs.VCsWorkflowsDir)
	}
	if dirs.AgentsDir != filepath.Join(dirs.AgentFieldHome, "agents") {
		t.Fatalf("AgentsDir = %q", dirs.AgentsDir)
	}
	if dirs.LogsDir != filepath.Join(dirs.AgentFieldHome, "logs") {
		t.Fatalf("LogsDir = %q", dirs.LogsDir)
	}
	if dirs.ConfigDir != filepath.Join(dirs.AgentFieldHome, "config") {
		t.Fatalf("ConfigDir = %q", dirs.ConfigDir)
	}
	if dirs.TempDir != filepath.Join(dirs.AgentFieldHome, "temp") {
		t.Fatalf("TempDir = %q", dirs.TempDir)
	}
	if dirs.PayloadsDir != filepath.Join(dirs.DataDir, "payloads") {
		t.Fatalf("PayloadsDir = %q", dirs.PayloadsDir)
	}
}

func TestGetAgentFieldDataDirectoriesUsesHomeDirFallback(t *testing.T) {
	homeDir := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", "")
	t.Setenv("HOME", homeDir)

	dirs, err := GetAgentFieldDataDirectories()
	if err != nil {
		t.Fatalf("GetAgentFieldDataDirectories() error = %v", err)
	}

	wantHome := filepath.Join(homeDir, ".agentfield")
	if dirs.AgentFieldHome != wantHome {
		t.Fatalf("AgentFieldHome = %q, want %q", dirs.AgentFieldHome, wantHome)
	}
}

func TestGetAgentFieldDataDirectoriesErrorsWithoutHome(t *testing.T) {
	t.Setenv("AGENTFIELD_HOME", "")
	t.Setenv("HOME", "")

	if _, err := GetAgentFieldDataDirectories(); err == nil {
		t.Fatal("GetAgentFieldDataDirectories() error = nil, want error")
	}
}

func TestEnsureDataDirectoriesSuccess(t *testing.T) {
	home := filepath.Join(t.TempDir(), "agentfield-home")
	t.Setenv("AGENTFIELD_HOME", home)

	dirs, err := EnsureDataDirectories()
	if err != nil {
		t.Fatalf("EnsureDataDirectories() error = %v", err)
	}

	paths := []string{
		dirs.AgentFieldHome,
		dirs.DataDir,
		dirs.KeysDir,
		dirs.DIDRegistriesDir,
		dirs.VCsExecutionsDir,
		dirs.VCsWorkflowsDir,
		dirs.AgentsDir,
		dirs.LogsDir,
		dirs.ConfigDir,
		dirs.TempDir,
		dirs.PayloadsDir,
	}

	for _, path := range paths {
		info, err := os.Stat(path)
		if err != nil {
			t.Fatalf("os.Stat(%q) error = %v", path, err)
		}
		if !info.IsDir() {
			t.Fatalf("%q is not a directory", path)
		}
	}

	if runtime.GOOS != "windows" {
		for _, path := range []string{dirs.KeysDir, dirs.DIDRegistriesDir} {
			info, err := os.Stat(path)
			if err != nil {
				t.Fatalf("os.Stat(%q) error = %v", path, err)
			}
			if got := info.Mode().Perm(); got != 0o700 {
				t.Fatalf("permissions for %q = %o, want 700", path, got)
			}
		}
	}
}

func TestEnsureDataDirectoriesFailure(t *testing.T) {
	parent := t.TempDir()
	blocker := filepath.Join(parent, "blocker")
	if err := os.WriteFile(blocker, []byte("block"), 0o644); err != nil {
		t.Fatalf("os.WriteFile() error = %v", err)
	}
	t.Setenv("AGENTFIELD_HOME", blocker)

	if _, err := EnsureDataDirectories(); err == nil {
		t.Fatal("EnsureDataDirectories() error = nil, want error")
	}
}

func TestPathHelpers(t *testing.T) {
	home := filepath.Join(t.TempDir(), "agentfield-home")
	t.Setenv("AGENTFIELD_HOME", home)

	tests := []struct {
		name string
		fn   func() (string, error)
		want string
	}{
		{name: "database", fn: GetDatabasePath, want: filepath.Join(home, "data", "agentfield.db")},
		{name: "kvstore", fn: GetKVStorePath, want: filepath.Join(home, "data", "agentfield.bolt")},
		{name: "registry", fn: GetAgentRegistryPath, want: filepath.Join(home, "installed.json")},
		{name: "config", fn: func() (string, error) { return GetConfigPath("app.yaml") }, want: filepath.Join(home, "config", "app.yaml")},
		{name: "log", fn: func() (string, error) { return GetLogPath("server.log") }, want: filepath.Join(home, "logs", "server.log")},
		{name: "temp", fn: func() (string, error) { return GetTempPath("tmp.txt") }, want: filepath.Join(home, "temp", "tmp.txt")},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			got, err := tc.fn()
			if err != nil {
				t.Fatalf("%s error = %v", tc.name, err)
			}
			if got != tc.want {
				t.Fatalf("%s = %q, want %q", tc.name, got, tc.want)
			}
		})
	}
}

func TestPathHelpersPropagateDirectoryResolutionErrors(t *testing.T) {
	t.Setenv("AGENTFIELD_HOME", "")
	t.Setenv("HOME", "")

	tests := []struct {
		name string
		fn   func() (string, error)
	}{
		{name: "database", fn: GetDatabasePath},
		{name: "kvstore", fn: GetKVStorePath},
		{name: "registry", fn: GetAgentRegistryPath},
		{name: "config", fn: func() (string, error) { return GetConfigPath("app.yaml") }},
		{name: "log", fn: func() (string, error) { return GetLogPath("server.log") }},
		{name: "temp", fn: func() (string, error) { return GetTempPath("tmp.txt") }},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			if _, err := tc.fn(); err == nil {
				t.Fatalf("%s error = nil, want error", tc.name)
			}
		})
	}
}

func TestGetPlatformSpecificPaths(t *testing.T) {
	t.Setenv("APPDATA", "/tmp/appdata")
	t.Setenv("LOCALAPPDATA", "/tmp/localappdata")
	t.Setenv("XDG_CONFIG_HOME", "/tmp/xdg-config")
	t.Setenv("XDG_DATA_HOME", "/tmp/xdg-data")
	t.Setenv("XDG_CACHE_HOME", "/tmp/xdg-cache")
	t.Setenv("HOME", "/tmp/home")

	paths := GetPlatformSpecificPaths()

	switch runtime.GOOS {
	case "windows":
		if paths["app_data"] != "/tmp/appdata" || paths["local_app_data"] != "/tmp/localappdata" {
			t.Fatalf("unexpected windows paths: %#v", paths)
		}
	case "darwin":
		if paths["application_support"] != filepath.Join("/tmp/home", "Library", "Application Support") {
			t.Fatalf("unexpected macOS application support path: %#v", paths)
		}
		if paths["caches"] != filepath.Join("/tmp/home", "Library", "Caches") {
			t.Fatalf("unexpected macOS caches path: %#v", paths)
		}
	case "linux":
		if paths["xdg_config_home"] != "/tmp/xdg-config" {
			t.Fatalf("xdg_config_home = %q", paths["xdg_config_home"])
		}
		if paths["xdg_data_home"] != "/tmp/xdg-data" {
			t.Fatalf("xdg_data_home = %q", paths["xdg_data_home"])
		}
		if paths["xdg_cache_home"] != "/tmp/xdg-cache" {
			t.Fatalf("xdg_cache_home = %q", paths["xdg_cache_home"])
		}
	}
}

func TestValidatePaths(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		home := filepath.Join(t.TempDir(), "agentfield-home")
		t.Setenv("AGENTFIELD_HOME", home)
		if err := os.MkdirAll(home, 0o755); err != nil {
			t.Fatalf("os.MkdirAll() error = %v", err)
		}

		if err := ValidatePaths(); err != nil {
			t.Fatalf("ValidatePaths() error = %v", err)
		}

		if _, err := os.Stat(filepath.Join(home, ".write_test")); !os.IsNotExist(err) {
			t.Fatalf("temporary validation file should be removed, stat err = %v", err)
		}
	})

	t.Run("failure", func(t *testing.T) {
		parent := t.TempDir()
		blocker := filepath.Join(parent, "blocker")
		if err := os.WriteFile(blocker, []byte("block"), 0o644); err != nil {
			t.Fatalf("os.WriteFile() error = %v", err)
		}
		t.Setenv("AGENTFIELD_HOME", blocker)

		if err := ValidatePaths(); err == nil {
			t.Fatal("ValidatePaths() error = nil, want error")
		}
	})
}
