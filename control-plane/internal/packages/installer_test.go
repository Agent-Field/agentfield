package packages

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

func writeTestPackage(t *testing.T, root string, yamlBody string) {
	t.Helper()
	if err := os.MkdirAll(root, 0755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "agentfield-package.yaml"), []byte(yamlBody), 0644); err != nil {
		t.Fatalf("write metadata: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "main.py"), []byte("print('ok')\n"), 0644); err != nil {
		t.Fatalf("write main.py: %v", err)
	}
}

func readRegistryFile(t *testing.T, path string) InstallationRegistry {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read registry: %v", err)
	}
	var registry InstallationRegistry
	if err := yaml.Unmarshal(data, &registry); err != nil {
		t.Fatalf("unmarshal registry: %v", err)
	}
	return registry
}

func TestResolveServerURL(t *testing.T) {
	// t.Setenv registers cleanup to restore original values when test ends.
	// Initial call registers the restore; os.Unsetenv clears for the default case.
	t.Setenv("AGENTFIELD_SERVER", "")
	t.Setenv("AGENTFIELD_SERVER_URL", "")
	os.Unsetenv("AGENTFIELD_SERVER")
	os.Unsetenv("AGENTFIELD_SERVER_URL")
	if got := resolveServerURL(); got != "http://localhost:8080" {
		t.Fatalf("default URL = %q", got)
	}

	t.Setenv("AGENTFIELD_SERVER_URL", "http://from-url")
	if got := resolveServerURL(); got != "http://from-url" {
		t.Fatalf("server URL env = %q", got)
	}

	t.Setenv("AGENTFIELD_SERVER", "http://preferred")
	if got := resolveServerURL(); got != "http://preferred" {
		t.Fatalf("preferred env = %q", got)
	}
}

func TestPackageInstallerValidationAndMetadata(t *testing.T) {
	t.Parallel()

	installer := &PackageInstaller{AgentFieldHome: t.TempDir()}
	base := t.TempDir()

	validYAML := strings.TrimSpace(`
name: demo
version: 1.2.3
description: demo package
user_environment:
  required:
    - name: REQUIRED_TOKEN
      description: api token
  optional:
    - name: OPTIONAL_REGION
      description: region
      default: us
`) + "\n"

	t.Run("validate package", func(t *testing.T) {
		pkg := filepath.Join(base, "valid")
		writeTestPackage(t, pkg, validYAML)

		if err := installer.validatePackage(pkg); err != nil {
			t.Fatalf("validatePackage: %v", err)
		}

		metadata, err := installer.parsePackageMetadata(pkg)
		if err != nil {
			t.Fatalf("parsePackageMetadata: %v", err)
		}
		if metadata.Name != "demo" || metadata.Version != "1.2.3" {
			t.Fatalf("unexpected metadata: %+v", metadata)
		}
		if metadata.Main != "main.py" {
			t.Fatalf("expected default main.py, got %q", metadata.Main)
		}
	})

	cases := []struct {
		name    string
		setup   func(string)
		wantErr string
	}{
		{
			name: "missing yaml",
			setup: func(dir string) {
				_ = os.MkdirAll(dir, 0755)
				_ = os.WriteFile(filepath.Join(dir, "main.py"), []byte("print('ok')\n"), 0644)
			},
			wantErr: "agentfield-package.yaml not found",
		},
		{
			name: "missing main",
			setup: func(dir string) {
				_ = os.MkdirAll(dir, 0755)
				_ = os.WriteFile(filepath.Join(dir, "agentfield-package.yaml"), []byte(validYAML), 0644)
			},
			wantErr: "main.py not found",
		},
		{
			name: "missing name",
			setup: func(dir string) {
				writeTestPackage(t, dir, "version: 1.0.0\n")
			},
			wantErr: "package name is required",
		},
		{
			name: "missing version",
			setup: func(dir string) {
				writeTestPackage(t, dir, "name: demo\n")
			},
			wantErr: "package version is required",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := filepath.Join(base, tc.name)
			tc.setup(dir)

			if strings.Contains(tc.wantErr, "required") {
				_, err := installer.parsePackageMetadata(dir)
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("parsePackageMetadata error = %v, want %q", err, tc.wantErr)
				}
				return
			}

			err := installer.validatePackage(dir)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("validatePackage error = %v, want %q", err, tc.wantErr)
			}
		})
	}
}

func TestPackageInstallerCopyRegistryAndInstall(t *testing.T) {
	t.Parallel()

	home := t.TempDir()
	installer := &PackageInstaller{AgentFieldHome: home}

	source := filepath.Join(t.TempDir(), "source")
	writeTestPackage(t, source, strings.TrimSpace(`
name: installed-demo
version: 0.1.0
description: package under test
dependencies: {}
user_environment:
  required:
    - name: REQUIRED_TOKEN
      description: token
  optional:
    - name: OPTIONAL_REGION
      description: region
      default: us-east-1
`)+"\n")
	if err := os.WriteFile(filepath.Join(source, "requirements.txt"), []byte(""), 0644); err != nil {
		t.Fatalf("write requirements: %v", err)
	}

	if !installer.hasRequirementsFile(source) {
		t.Fatalf("expected requirements.txt to be detected")
	}

	dest := filepath.Join(home, "packages", "copied")
	if err := installer.copyPackage(source, dest); err != nil {
		t.Fatalf("copyPackage: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dest, "main.py")); err != nil {
		t.Fatalf("copied main.py: %v", err)
	}

	singleDst := filepath.Join(home, "single.txt")
	if err := installer.copyFile(filepath.Join(source, "main.py"), singleDst); err != nil {
		t.Fatalf("copyFile: %v", err)
	}

	metadata, err := installer.parsePackageMetadata(source)
	if err != nil {
		t.Fatalf("parsePackageMetadata: %v", err)
	}

	if err := installer.updateRegistry(metadata, source, dest); err != nil {
		t.Fatalf("updateRegistry: %v", err)
	}
	if !installer.isPackageInstalled("installed-demo") {
		t.Fatalf("expected package to be installed")
	}

	registry := readRegistryFile(t, filepath.Join(home, "installed.yaml"))
	if registry.Installed["installed-demo"].Runtime.LogFile == "" {
		t.Fatalf("expected log file to be set")
	}

	if err := installer.InstallPackage(source, true); err != nil {
		t.Fatalf("InstallPackage: %v", err)
	}
	if !installer.isPackageInstalled("installed-demo") {
		t.Fatalf("expected package after InstallPackage")
	}
}

func TestPackageUninstallerLifecycle(t *testing.T) {
	t.Parallel()

	home := t.TempDir()
	pkgPath := filepath.Join(home, "packages", "demo")
	if err := os.MkdirAll(pkgPath, 0755); err != nil {
		t.Fatalf("mkdir package: %v", err)
	}
	logDir := filepath.Join(home, "logs")
	if err := os.MkdirAll(logDir, 0755); err != nil {
		t.Fatalf("mkdir logs: %v", err)
	}
	logPath := filepath.Join(logDir, "demo.log")
	if err := os.WriteFile(logPath, []byte("log"), 0644); err != nil {
		t.Fatalf("write log: %v", err)
	}

	cmd := exec.Command("sleep", "30")
	if err := cmd.Start(); err != nil {
		t.Fatalf("start sleep: %v", err)
	}
	t.Cleanup(func() {
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
	})

	pid := cmd.Process.Pid
	registry := &InstallationRegistry{
		Installed: map[string]InstalledPackage{
			"demo": {
				Name:   "demo",
				Path:   pkgPath,
				Status: "running",
				Runtime: RuntimeInfo{
					PID:     &pid,
					LogFile: logPath,
				},
			},
		},
	}

	uninstaller := &PackageUninstaller{AgentFieldHome: home}
	if err := uninstaller.saveRegistry(registry); err != nil {
		t.Fatalf("saveRegistry: %v", err)
	}

	loaded, err := uninstaller.loadRegistry()
	if err != nil {
		t.Fatalf("loadRegistry: %v", err)
	}
	if _, ok := loaded.Installed["demo"]; !ok {
		t.Fatalf("expected demo in loaded registry")
	}

	if err := uninstaller.UninstallPackage("demo"); err == nil || !strings.Contains(err.Error(), "currently running") {
		t.Fatalf("expected running error, got %v", err)
	}

	uninstaller.Force = true
	if err := uninstaller.UninstallPackage("demo"); err != nil {
		t.Fatalf("forced uninstall: %v", err)
	}
	if _, err := os.Stat(pkgPath); !os.IsNotExist(err) {
		t.Fatalf("expected package path removed, stat err=%v", err)
	}
	if _, err := os.Stat(logPath); !os.IsNotExist(err) {
		t.Fatalf("expected log path removed, stat err=%v", err)
	}

	if err := uninstaller.stopAgentNode(&InstalledPackage{}); err == nil || !strings.Contains(err.Error(), "no PID found") {
		t.Fatalf("expected missing PID error, got %v", err)
	}
}
