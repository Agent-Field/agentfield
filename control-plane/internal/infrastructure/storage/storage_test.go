package storage

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/domain"
)

type stubFS struct {
	exists    bool
	readData  []byte
	readErr   error
	writeErr  error
	writtenTo string
	written   []byte
}

func (s *stubFS) ReadFile(path string) ([]byte, error) {
	if s.readErr != nil {
		return nil, s.readErr
	}
	return s.readData, nil
}

func (s *stubFS) WriteFile(path string, data []byte) error {
	s.writtenTo = path
	s.written = append([]byte(nil), data...)
	return s.writeErr
}

func (s *stubFS) Exists(path string) bool { return s.exists }

func (s *stubFS) CreateDirectory(path string) error { return nil }

func (s *stubFS) ListDirectory(path string) ([]string, error) { return nil, nil }

func TestLocalConfigStorageLoadAgentFieldConfig(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		path      string
		fs        *stubFS
		wantHome  string
		wantEnv   map[string]string
		wantErr   error
		assertNil bool
	}{
		{
			name:     "missing file returns defaults",
			path:     "/tmp/agentfield/config/agentfield.yaml",
			fs:       &stubFS{exists: false},
			wantHome: "/tmp/agentfield/config",
			wantEnv:  map[string]string{},
		},
		{
			name: "read error",
			path: "/tmp/agentfield/config/agentfield.yaml",
			fs: &stubFS{
				exists:  true,
				readErr: errors.New("read failed"),
			},
			wantErr:   errors.New("read failed"),
			assertNil: true,
		},
		{
			name: "invalid yaml",
			path: "/tmp/agentfield/config/agentfield.yaml",
			fs: &stubFS{
				exists:   true,
				readData: []byte("home_dir: ["),
			},
			wantErr:   errors.New("yaml"),
			assertNil: true,
		},
		{
			name: "valid yaml",
			path: "/tmp/agentfield/config/agentfield.yaml",
			fs: &stubFS{
				exists:   true,
				readData: []byte("{}\n"),
			},
			wantHome: "",
			wantEnv:  map[string]string{},
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			config, err := NewLocalConfigStorage(tt.fs).LoadAgentFieldConfig(tt.path)
			if tt.wantErr != nil {
				if err == nil || !containsError(err, tt.wantErr.Error()) {
					t.Fatalf("expected error containing %q, got %v", tt.wantErr.Error(), err)
				}
				if tt.assertNil && config != nil {
					t.Fatalf("expected nil config, got %#v", config)
				}
				return
			}

			if err != nil {
				t.Fatalf("LoadAgentFieldConfig returned error: %v", err)
			}
			if config == nil {
				t.Fatal("expected config, got nil")
			}
			if config.HomeDir != tt.wantHome {
				t.Fatalf("HomeDir = %q, want %q", config.HomeDir, tt.wantHome)
			}
			if len(config.Environment) != len(tt.wantEnv) {
				t.Fatalf("Environment len = %d, want %d", len(config.Environment), len(tt.wantEnv))
			}
			for key, want := range tt.wantEnv {
				if got := config.Environment[key]; got != want {
					t.Fatalf("Environment[%q] = %q, want %q", key, got, want)
				}
			}
		})
	}
}

func TestLocalConfigStorageSaveAgentFieldConfig(t *testing.T) {
	t.Parallel()

	t.Run("writes serialized config", func(t *testing.T) {
		t.Parallel()

		fs := &stubFS{}
		path := filepath.Join(t.TempDir(), "config", "agentfield.yaml")
		cfg := &domain.AgentFieldConfig{
			HomeDir:     "/srv/agentfield",
			Environment: map[string]string{"FOO": "bar"},
		}

		if err := NewLocalConfigStorage(fs).SaveAgentFieldConfig(path, cfg); err != nil {
			t.Fatalf("SaveAgentFieldConfig returned error: %v", err)
		}
		if fs.writtenTo != path {
			t.Fatalf("WriteFile path = %q, want %q", fs.writtenTo, path)
		}
		if len(fs.written) == 0 {
			t.Fatal("expected serialized config to be written")
		}
	})

	t.Run("mkdir failure", func(t *testing.T) {
		t.Parallel()

		blocker := filepath.Join(t.TempDir(), "blocker")
		if err := os.WriteFile(blocker, []byte("x"), 0644); err != nil {
			t.Fatalf("WriteFile blocker: %v", err)
		}

		err := NewLocalConfigStorage(&stubFS{}).SaveAgentFieldConfig(
			filepath.Join(blocker, "child", "agentfield.yaml"),
			&domain.AgentFieldConfig{Environment: map[string]string{}},
		)
		if err == nil {
			t.Fatal("expected error from MkdirAll")
		}
	})

	t.Run("write failure", func(t *testing.T) {
		t.Parallel()

		fs := &stubFS{writeErr: errors.New("write failed")}
		err := NewLocalConfigStorage(fs).SaveAgentFieldConfig(
			filepath.Join(t.TempDir(), "config", "agentfield.yaml"),
			&domain.AgentFieldConfig{Environment: map[string]string{}},
		)
		if err == nil || !containsError(err, "write failed") {
			t.Fatalf("expected write failure, got %v", err)
		}
	})
}

func TestLocalRegistryStorageLoadRegistry(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		fs        *stubFS
		wantCount int
		wantErr   error
		assertNil bool
	}{
		{
			name:      "missing file returns empty registry",
			fs:        &stubFS{exists: false},
			wantCount: 0,
		},
		{
			name: "read error",
			fs: &stubFS{
				exists:  true,
				readErr: errors.New("read failed"),
			},
			wantErr:   errors.New("read failed"),
			assertNil: true,
		},
		{
			name: "invalid json",
			fs: &stubFS{
				exists:   true,
				readData: []byte("{"),
			},
			wantErr:   errors.New("unexpected end of JSON"),
			assertNil: true,
		},
		{
			name: "valid json",
			fs: &stubFS{
				exists:   true,
				readData: []byte(`{"installed":{"pkg":{"name":"pkg","version":"1.0.0","path":"/tmp/pkg","environment":{"FOO":"bar"}}}}`),
			},
			wantCount: 1,
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			registry, err := NewLocalRegistryStorage(tt.fs, "/tmp/registry.json").LoadRegistry()
			if tt.wantErr != nil {
				if err == nil || !containsError(err, tt.wantErr.Error()) {
					t.Fatalf("expected error containing %q, got %v", tt.wantErr.Error(), err)
				}
				if tt.assertNil && registry != nil {
					t.Fatalf("expected nil registry, got %#v", registry)
				}
				return
			}

			if err != nil {
				t.Fatalf("LoadRegistry returned error: %v", err)
			}
			if registry == nil {
				t.Fatal("expected registry, got nil")
			}
			if len(registry.Installed) != tt.wantCount {
				t.Fatalf("Installed len = %d, want %d", len(registry.Installed), tt.wantCount)
			}
		})
	}
}

func TestLocalRegistryStorageSaveRegistry(t *testing.T) {
	t.Parallel()

	t.Run("writes serialized registry", func(t *testing.T) {
		t.Parallel()

		fs := &stubFS{}
		path := filepath.Join(t.TempDir(), "registry", "installed.json")
		registry := &domain.InstallationRegistry{
			Installed: map[string]domain.InstalledPackage{
				"pkg": {
					Name:        "pkg",
					Version:     "1.0.0",
					Path:        "/tmp/pkg",
					Environment: map[string]string{"FOO": "bar"},
					InstalledAt: time.Unix(123, 0).UTC(),
				},
			},
		}

		if err := NewLocalRegistryStorage(fs, path).SaveRegistry(registry); err != nil {
			t.Fatalf("SaveRegistry returned error: %v", err)
		}
		if fs.writtenTo != path {
			t.Fatalf("WriteFile path = %q, want %q", fs.writtenTo, path)
		}
		if len(fs.written) == 0 {
			t.Fatal("expected serialized registry to be written")
		}
	})

	t.Run("mkdir failure", func(t *testing.T) {
		t.Parallel()

		blocker := filepath.Join(t.TempDir(), "blocker")
		if err := os.WriteFile(blocker, []byte("x"), 0644); err != nil {
			t.Fatalf("WriteFile blocker: %v", err)
		}

		err := NewLocalRegistryStorage(&stubFS{}, filepath.Join(blocker, "child", "installed.json")).SaveRegistry(
			&domain.InstallationRegistry{Installed: map[string]domain.InstalledPackage{}},
		)
		if err == nil {
			t.Fatal("expected error from MkdirAll")
		}
	})

	t.Run("write failure", func(t *testing.T) {
		t.Parallel()

		err := NewLocalRegistryStorage(
			&stubFS{writeErr: errors.New("write failed")},
			filepath.Join(t.TempDir(), "registry", "installed.json"),
		).SaveRegistry(&domain.InstallationRegistry{Installed: map[string]domain.InstalledPackage{}})
		if err == nil || !containsError(err, "write failed") {
			t.Fatalf("expected write failure, got %v", err)
		}
	})
}

func TestLocalRegistryStorageGetPackage(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		fs        *stubFS
		wantName  string
		wantErr   error
		assertNil bool
	}{
		{
			name: "load error",
			fs: &stubFS{
				exists:  true,
				readErr: errors.New("read failed"),
			},
			wantErr:   errors.New("read failed"),
			assertNil: true,
		},
		{
			name: "package missing",
			fs: &stubFS{
				exists:   true,
				readData: []byte(`{"installed":{}}`),
			},
			wantErr:   os.ErrNotExist,
			assertNil: true,
		},
		{
			name: "package found",
			fs: &stubFS{
				exists:   true,
				readData: []byte(`{"installed":{"pkg":{"name":"pkg","version":"1.0.0"}}}`),
			},
			wantName: "pkg",
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			pkg, err := NewLocalRegistryStorage(tt.fs, "/tmp/registry.json").GetPackage("pkg")
			if tt.wantErr != nil {
				if !errors.Is(err, tt.wantErr) && !containsError(err, tt.wantErr.Error()) {
					t.Fatalf("expected error matching %v, got %v", tt.wantErr, err)
				}
				if tt.assertNil && pkg != nil {
					t.Fatalf("expected nil package, got %#v", pkg)
				}
				return
			}

			if err != nil {
				t.Fatalf("GetPackage returned error: %v", err)
			}
			if pkg == nil || pkg.Name != tt.wantName {
				t.Fatalf("package = %#v, want name %q", pkg, tt.wantName)
			}
		})
	}
}

func TestLocalRegistryStorageSavePackage(t *testing.T) {
	t.Parallel()

	t.Run("load error", func(t *testing.T) {
		t.Parallel()

		err := NewLocalRegistryStorage(&stubFS{
			exists:  true,
			readErr: errors.New("read failed"),
		}, "/tmp/registry.json").SavePackage("pkg", &domain.InstalledPackage{Name: "pkg"})
		if err == nil || !containsError(err, "read failed") {
			t.Fatalf("expected read failure, got %v", err)
		}
	})

	t.Run("save success", func(t *testing.T) {
		t.Parallel()

		fs := &stubFS{
			exists:   true,
			readData: []byte(`{"installed":{"existing":{"name":"existing","version":"0.1.0"}}}`),
		}
		path := filepath.Join(t.TempDir(), "registry", "installed.json")

		err := NewLocalRegistryStorage(fs, path).SavePackage("pkg", &domain.InstalledPackage{
			Name:    "pkg",
			Version: "1.0.0",
			Path:    "/tmp/pkg",
		})
		if err != nil {
			t.Fatalf("SavePackage returned error: %v", err)
		}
		if fs.writtenTo != path {
			t.Fatalf("WriteFile path = %q, want %q", fs.writtenTo, path)
		}
		if len(fs.written) == 0 {
			t.Fatal("expected updated registry to be written")
		}
		if !containsError(errors.New(string(fs.written)), `"pkg"`) {
			t.Fatalf("expected saved registry to contain package entry, got %s", string(fs.written))
		}
	})
}

func TestDefaultFileSystemAdapter(t *testing.T) {
	t.Parallel()

	fs := NewFileSystemAdapter()
	baseDir := t.TempDir()
	filePath := filepath.Join(baseDir, "nested", "file.txt")

	if err := fs.CreateDirectory(filepath.Dir(filePath)); err != nil {
		t.Fatalf("CreateDirectory returned error: %v", err)
	}

	if err := fs.WriteFile(filePath, []byte("hello")); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}

	if !fs.Exists(filePath) {
		t.Fatalf("Exists(%q) = false, want true", filePath)
	}

	data, err := fs.ReadFile(filePath)
	if err != nil {
		t.Fatalf("ReadFile returned error: %v", err)
	}
	if string(data) != "hello" {
		t.Fatalf("ReadFile data = %q, want %q", string(data), "hello")
	}

	entries, err := fs.ListDirectory(filepath.Dir(filePath))
	if err != nil {
		t.Fatalf("ListDirectory returned error: %v", err)
	}
	if len(entries) != 1 || entries[0] != "file.txt" {
		t.Fatalf("ListDirectory entries = %#v, want [\"file.txt\"]", entries)
	}

	if got := fs.Exists(filepath.Join(baseDir, "does-not-exist")); got {
		t.Fatal("Exists returned true for missing path")
	}
}

func TestDefaultFileSystemAdapterListDirectoryError(t *testing.T) {
	t.Parallel()

	_, err := NewFileSystemAdapter().ListDirectory(filepath.Join(t.TempDir(), "missing"))
	if err == nil {
		t.Fatal("expected error for missing directory")
	}
}

func containsError(err error, want string) bool {
	return err != nil && want != "" && (err.Error() == want || strings.Contains(err.Error(), want))
}
