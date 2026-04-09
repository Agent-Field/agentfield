package packages

import (
	"archive/zip"
	"bytes"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"gopkg.in/yaml.v3"
)

func TestPackageInstallerRegistryAndMetadataErrors(t *testing.T) {
	t.Parallel()

	t.Run("parsePackageMetadata invalid yaml", func(t *testing.T) {
		installer := &PackageInstaller{AgentFieldHome: t.TempDir()}
		pkgDir := t.TempDir()
		if err := os.WriteFile(filepath.Join(pkgDir, "agentfield-package.yaml"), []byte("name: [\n"), 0644); err != nil {
			t.Fatalf("write yaml: %v", err)
		}

		_, err := installer.parsePackageMetadata(pkgDir)
		if err == nil || !strings.Contains(err.Error(), "failed to parse agentfield-package.yaml") {
			t.Fatalf("parsePackageMetadata error = %v", err)
		}
	})

	t.Run("isPackageInstalled invalid registry", func(t *testing.T) {
		home := t.TempDir()
		installer := &PackageInstaller{AgentFieldHome: home}
		if err := os.WriteFile(filepath.Join(home, "installed.yaml"), []byte("installed: ["), 0644); err != nil {
			t.Fatalf("write registry: %v", err)
		}

		if installer.isPackageInstalled("demo") {
			t.Fatalf("expected invalid registry to report not installed")
		}
	})

	t.Run("updateRegistry invalid registry", func(t *testing.T) {
		home := t.TempDir()
		installer := &PackageInstaller{AgentFieldHome: home}
		if err := os.WriteFile(filepath.Join(home, "installed.yaml"), []byte("installed: ["), 0644); err != nil {
			t.Fatalf("write registry: %v", err)
		}

		err := installer.updateRegistry(&PackageMetadata{Name: "demo", Version: "1.0.0"}, "/src", "/dst")
		if err == nil || !strings.Contains(err.Error(), "failed to parse registry") {
			t.Fatalf("updateRegistry error = %v", err)
		}
	})

	t.Run("updateRegistry logs dir failure", func(t *testing.T) {
		home := t.TempDir()
		installer := &PackageInstaller{AgentFieldHome: home}
		if err := os.WriteFile(filepath.Join(home, "logs"), []byte("not a dir"), 0644); err != nil {
			t.Fatalf("write logs file: %v", err)
		}

		err := installer.updateRegistry(&PackageMetadata{Name: "demo", Version: "1.0.0"}, "/src", "/dst")
		if err == nil || !strings.Contains(err.Error(), "failed to create logs directory") {
			t.Fatalf("updateRegistry error = %v", err)
		}
	})

	t.Run("saveRegistry write failure", func(t *testing.T) {
		uninstaller := &PackageUninstaller{AgentFieldHome: filepath.Join(t.TempDir(), "missing", "home")}
		err := uninstaller.saveRegistry(&InstallationRegistry{Installed: map[string]InstalledPackage{}})
		if err == nil || !strings.Contains(err.Error(), "failed to write registry") {
			t.Fatalf("saveRegistry error = %v", err)
		}
	})
}

func TestPackageUninstallerErrorPaths(t *testing.T) {
	t.Parallel()

	t.Run("package not installed", func(t *testing.T) {
		uninstaller := &PackageUninstaller{AgentFieldHome: t.TempDir()}
		err := uninstaller.UninstallPackage("missing")
		if err == nil || !strings.Contains(err.Error(), "is not installed") {
			t.Fatalf("UninstallPackage error = %v", err)
		}
	})

	t.Run("running package requires force", func(t *testing.T) {
		home := t.TempDir()
		registry := &InstallationRegistry{
			Installed: map[string]InstalledPackage{
				"demo": {Name: "demo", Status: "running"},
			},
		}
		if err := (&PackageUninstaller{AgentFieldHome: home}).saveRegistry(registry); err != nil {
			t.Fatalf("saveRegistry: %v", err)
		}

		err := (&PackageUninstaller{AgentFieldHome: home}).UninstallPackage("demo")
		if err == nil || !strings.Contains(err.Error(), "currently running") {
			t.Fatalf("UninstallPackage error = %v", err)
		}
	})

	t.Run("stopAgentNode missing pid", func(t *testing.T) {
		err := (&PackageUninstaller{}).stopAgentNode(&InstalledPackage{})
		if err == nil || !strings.Contains(err.Error(), "no PID found") {
			t.Fatalf("stopAgentNode error = %v", err)
		}
	})

	t.Run("loadRegistry invalid yaml", func(t *testing.T) {
		home := t.TempDir()
		if err := os.WriteFile(filepath.Join(home, "installed.yaml"), []byte("installed: ["), 0644); err != nil {
			t.Fatalf("write registry: %v", err)
		}

		_, err := (&PackageUninstaller{AgentFieldHome: home}).loadRegistry()
		if err == nil || !strings.Contains(err.Error(), "failed to parse registry") {
			t.Fatalf("loadRegistry error = %v", err)
		}
	})

	t.Run("running package force ignores stop failure", func(t *testing.T) {
		home := t.TempDir()
		pkgPath := filepath.Join(home, "packages", "demo")
		logPath := filepath.Join(home, "logs", "demo.log")
		if err := os.MkdirAll(pkgPath, 0755); err != nil {
			t.Fatalf("mkdir package: %v", err)
		}
		if err := os.MkdirAll(filepath.Dir(logPath), 0755); err != nil {
			t.Fatalf("mkdir logs: %v", err)
		}
		if err := os.WriteFile(logPath, []byte("log"), 0644); err != nil {
			t.Fatalf("write log: %v", err)
		}

		registry := &InstallationRegistry{
			Installed: map[string]InstalledPackage{
				"demo": {
					Name:   "demo",
					Path:   pkgPath,
					Status: "running",
					Runtime: RuntimeInfo{
						LogFile: logPath,
					},
				},
			},
		}
		if err := (&PackageUninstaller{AgentFieldHome: home}).saveRegistry(registry); err != nil {
			t.Fatalf("saveRegistry: %v", err)
		}

		if err := (&PackageUninstaller{AgentFieldHome: home, Force: true}).UninstallPackage("demo"); err != nil {
			t.Fatalf("UninstallPackage: %v", err)
		}
	})

	t.Run("log removal warning still succeeds", func(t *testing.T) {
		home := t.TempDir()
		pkgPath := filepath.Join(home, "packages", "demo")
		logPath := filepath.Join(home, "logs", "demo.log")
		if err := os.MkdirAll(pkgPath, 0755); err != nil {
			t.Fatalf("mkdir package: %v", err)
		}
		if err := os.MkdirAll(logPath, 0755); err != nil {
			t.Fatalf("mkdir log dir: %v", err)
		}

		registry := &InstallationRegistry{
			Installed: map[string]InstalledPackage{
				"demo": {
					Name: "demo",
					Path: pkgPath,
					Runtime: RuntimeInfo{
						LogFile: logPath,
					},
				},
			},
		}
		if err := (&PackageUninstaller{AgentFieldHome: home}).saveRegistry(registry); err != nil {
			t.Fatalf("saveRegistry: %v", err)
		}

		if err := (&PackageUninstaller{AgentFieldHome: home}).UninstallPackage("demo"); err != nil {
			t.Fatalf("UninstallPackage: %v", err)
		}
	})

	t.Run("stopAgentNode kill failure", func(t *testing.T) {
		cmd := exec.Command("sleep", "1")
		if err := cmd.Start(); err != nil {
			t.Fatalf("start sleep: %v", err)
		}
		if err := cmd.Process.Kill(); err != nil {
			t.Fatalf("kill sleep: %v", err)
		}
		_, _ = cmd.Process.Wait()

		pid := cmd.Process.Pid
		err := (&PackageUninstaller{}).stopAgentNode(&InstalledPackage{
			Runtime: RuntimeInfo{PID: &pid},
		})
		if err == nil || !strings.Contains(err.Error(), "failed to kill process") {
			t.Fatalf("stopAgentNode error = %v", err)
		}
	})
}

func TestPackageInstallerCopyPackageDestinationError(t *testing.T) {
	t.Parallel()

	src := t.TempDir()
	writeTestPackage(t, src, "name: demo\nversion: 1.0.0\n")

	parentFile := filepath.Join(t.TempDir(), "parent")
	if err := os.WriteFile(parentFile, []byte("file"), 0644); err != nil {
		t.Fatalf("write parent file: %v", err)
	}

	err := (&PackageInstaller{}).copyPackage(src, filepath.Join(parentFile, "child"))
	if err == nil || !strings.Contains(err.Error(), "failed to remove existing package") {
		t.Fatalf("copyPackage error = %v", err)
	}
}

func TestPackageInstallerCopyPackageMissingSource(t *testing.T) {
	t.Parallel()

	err := (&PackageInstaller{}).copyPackage(filepath.Join(t.TempDir(), "missing"), filepath.Join(t.TempDir(), "dest"))
	if err == nil {
		t.Fatalf("expected copyPackage to fail for missing source")
	}
}

func TestGitHubInstallerAdditionalErrorCoverage(t *testing.T) {
	t.Run("extractZip target open failure", func(t *testing.T) {
		zipPath := filepath.Join(t.TempDir(), "conflict.zip")
		file, err := os.Create(zipPath)
		if err != nil {
			t.Fatalf("create zip: %v", err)
		}

		zw := zip.NewWriter(file)
		dirHeader := &zip.FileHeader{Name: "conflict/"}
		dirHeader.SetMode(os.ModeDir | 0755)
		if _, err := zw.CreateHeader(dirHeader); err != nil {
			t.Fatalf("create dir header: %v", err)
		}
		entry, err := zw.Create("conflict")
		if err != nil {
			t.Fatalf("create conflict entry: %v", err)
		}
		if _, err := entry.Write([]byte("data")); err != nil {
			t.Fatalf("write conflict entry: %v", err)
		}
		if err := zw.Close(); err != nil {
			t.Fatalf("close zip: %v", err)
		}
		if err := file.Close(); err != nil {
			t.Fatalf("close file: %v", err)
		}

		err = (&GitHubInstaller{}).extractZip(zipPath, filepath.Join(t.TempDir(), "out"))
		if err == nil {
			t.Fatalf("expected extractZip error")
		}
	})

	t.Run("install from github invalid metadata", func(t *testing.T) {
		home := t.TempDir()

		sourceZip := filepath.Join(t.TempDir(), "bad.zip")
		createZipFile(t, sourceZip, map[string]string{
			"repo-main/agentfield-package.yaml": "name: bad-package\n",
			"repo-main/main.py":                 "print('ok')\n",
		})
		zipBytes, err := os.ReadFile(sourceZip)
		if err != nil {
			t.Fatalf("read zip: %v", err)
		}

		gi := &GitHubInstaller{
			AgentFieldHome: home,
			HTTPClient: &http.Client{
				Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
					return &http.Response{
						StatusCode: 200,
						Header:     make(http.Header),
						Body:       io.NopCloser(bytes.NewReader(zipBytes)),
					}, nil
				}),
			},
		}

		err = gi.InstallFromGitHub("acme/repo@main", true)
		if err == nil || !strings.Contains(err.Error(), "failed to parse package metadata") {
			t.Fatalf("InstallFromGitHub error = %v", err)
		}
	})

	t.Run("downloadAndExtract download failure", func(t *testing.T) {
		_, err := (&GitHubInstaller{}).downloadAndExtract(&GitHubPackageInfo{ArchiveURL: "://bad-url"})
		if err == nil || !strings.Contains(err.Error(), "failed to download archive") {
			t.Fatalf("downloadAndExtract error = %v", err)
		}
	})
}

func TestAgentNodeRunnerRunAgentNodeAdditionalCoverage(t *testing.T) {
	t.Run("package not installed", func(t *testing.T) {
		runner := &AgentNodeRunner{AgentFieldHome: t.TempDir()}
		err := runner.RunAgentNode("missing")
		if err == nil || !strings.Contains(err.Error(), "not installed") {
			t.Fatalf("RunAgentNode error = %v", err)
		}
	})

	t.Run("package already running", func(t *testing.T) {
		home := t.TempDir()
		port := 8123
		registry := &InstallationRegistry{
			Installed: map[string]InstalledPackage{
				"demo": {
					Name:   "demo",
					Status: "running",
					Runtime: RuntimeInfo{
						Port: &port,
					},
				},
			},
		}
		data, err := yaml.Marshal(registry)
		if err != nil {
			t.Fatalf("marshal registry: %v", err)
		}
		if err := os.WriteFile(filepath.Join(home, "installed.yaml"), data, 0644); err != nil {
			t.Fatalf("write registry: %v", err)
		}

		err = (&AgentNodeRunner{AgentFieldHome: home}).RunAgentNode("demo")
		if err == nil || !strings.Contains(err.Error(), "already running on port 8123") {
			t.Fatalf("RunAgentNode error = %v", err)
		}
	})
}

func TestPackageInstallerInstallPackageAdditionalCoverage(t *testing.T) {
	t.Run("validation failure", func(t *testing.T) {
		err := (&PackageInstaller{AgentFieldHome: t.TempDir()}).InstallPackage(t.TempDir(), true)
		if err == nil || !strings.Contains(err.Error(), "failed to parse package metadata") {
			t.Fatalf("InstallPackage error = %v", err)
		}
	})

	t.Run("duplicate install without force", func(t *testing.T) {
		home := t.TempDir()
		pkgDir := t.TempDir()
		writeTestPackage(t, pkgDir, "name: duplicate-demo\nversion: 1.0.0\n")

		registry := &InstallationRegistry{
			Installed: map[string]InstalledPackage{
				"duplicate-demo": {Name: "duplicate-demo"},
			},
		}
		if err := (&PackageUninstaller{AgentFieldHome: home}).saveRegistry(registry); err != nil {
			t.Fatalf("saveRegistry: %v", err)
		}

		err := (&PackageInstaller{AgentFieldHome: home}).InstallPackage(pkgDir, false)
		if err == nil || !strings.Contains(err.Error(), "already installed") {
			t.Fatalf("InstallPackage error = %v", err)
		}
	})

	t.Run("update registry failure", func(t *testing.T) {
		home := t.TempDir()
		pkgDir := t.TempDir()
		writeTestPackage(t, pkgDir, "name: registry-fail\nversion: 1.0.0\n")
		if err := os.WriteFile(filepath.Join(home, "installed.yaml"), []byte("installed: ["), 0644); err != nil {
			t.Fatalf("write registry: %v", err)
		}

		err := (&PackageInstaller{AgentFieldHome: home}).InstallPackage(pkgDir, true)
		if err == nil || !strings.Contains(err.Error(), "failed to update registry") {
			t.Fatalf("InstallPackage error = %v", err)
		}
	})
}

func TestGitHubInstallerInstallFromGitHubAdditionalCoverage(t *testing.T) {
	testClient := func(zipBytes []byte) *http.Client {
		return &http.Client{
			Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     make(http.Header),
					Body:       io.NopCloser(bytes.NewReader(zipBytes)),
				}, nil
			}),
		}
	}

	t.Run("duplicate install without force", func(t *testing.T) {
		home := t.TempDir()
		registry := &InstallationRegistry{
			Installed: map[string]InstalledPackage{
				"installed-gh": {Name: "installed-gh"},
			},
		}
		if err := (&PackageUninstaller{AgentFieldHome: home}).saveRegistry(registry); err != nil {
			t.Fatalf("saveRegistry: %v", err)
		}

		sourceZip := filepath.Join(t.TempDir(), "dup.zip")
		createZipFile(t, sourceZip, map[string]string{
			"repo-main/agentfield-package.yaml": "name: installed-gh\nversion: 1.0.0\n",
			"repo-main/main.py":                 "print('ok')\n",
		})
		zipBytes, err := os.ReadFile(sourceZip)
		if err != nil {
			t.Fatalf("read zip: %v", err)
		}

		err = (&GitHubInstaller{AgentFieldHome: home, HTTPClient: testClient(zipBytes)}).InstallFromGitHub("acme/repo@main", false)
		if err == nil || !strings.Contains(err.Error(), "already installed") {
			t.Fatalf("InstallFromGitHub error = %v", err)
		}
	})

	t.Run("update registry failure", func(t *testing.T) {
		home := t.TempDir()
		if err := os.WriteFile(filepath.Join(home, "installed.yaml"), []byte("installed: ["), 0644); err != nil {
			t.Fatalf("write registry: %v", err)
		}

		sourceZip := filepath.Join(t.TempDir(), "registry.zip")
		createZipFile(t, sourceZip, map[string]string{
			"repo-main/agentfield-package.yaml": "name: gh-registry\nversion: 1.0.0\n",
			"repo-main/main.py":                 "print('ok')\n",
		})
		zipBytes, err := os.ReadFile(sourceZip)
		if err != nil {
			t.Fatalf("read zip: %v", err)
		}

		err = (&GitHubInstaller{AgentFieldHome: home, HTTPClient: testClient(zipBytes)}).InstallFromGitHub("acme/repo@main", true)
		if err == nil || !strings.Contains(err.Error(), "failed to update registry") {
			t.Fatalf("InstallFromGitHub error = %v", err)
		}
	})

	t.Run("invalid package structure", func(t *testing.T) {
		sourceZip := filepath.Join(t.TempDir(), "invalid.zip")
		createZipFile(t, sourceZip, map[string]string{
			"repo-main/README.md": "missing package files\n",
		})
		zipBytes, err := os.ReadFile(sourceZip)
		if err != nil {
			t.Fatalf("read zip: %v", err)
		}

		err = (&GitHubInstaller{AgentFieldHome: t.TempDir(), HTTPClient: testClient(zipBytes)}).InstallFromGitHub("acme/repo@main", true)
		if err == nil || !strings.Contains(err.Error(), "invalid package structure") {
			t.Fatalf("InstallFromGitHub error = %v", err)
		}
	})

	t.Run("install dependencies failure", func(t *testing.T) {
		t.Setenv("PATH", "")

		sourceZip := filepath.Join(t.TempDir(), "deps.zip")
		createZipFile(t, sourceZip, map[string]string{
			"repo-main/agentfield-package.yaml": "name: gh-deps\nversion: 1.0.0\ndependencies:\n  python:\n    - missing-package\n",
			"repo-main/main.py":                 "print('ok')\n",
		})
		zipBytes, err := os.ReadFile(sourceZip)
		if err != nil {
			t.Fatalf("read zip: %v", err)
		}

		err = (&GitHubInstaller{AgentFieldHome: t.TempDir(), HTTPClient: testClient(zipBytes)}).InstallFromGitHub("acme/repo@main", true)
		if err == nil || !strings.Contains(err.Error(), "failed to install dependencies") {
			t.Fatalf("InstallFromGitHub error = %v", err)
		}
	})
}

func TestGitAndGitHubRegistryReadFailures(t *testing.T) {
	t.Parallel()

	fileHome := filepath.Join(t.TempDir(), "home-file")
	if err := os.WriteFile(fileHome, []byte("x"), 0644); err != nil {
		t.Fatalf("write home file: %v", err)
	}

	err := (&GitInstaller{AgentFieldHome: fileHome}).updateRegistryWithGit(
		&PackageMetadata{Name: "demo", Version: "1.0.0"},
		&GitPackageInfo{URL: "https://gitlab.com/acme/repo"},
		"/src",
		"/dst",
	)
	if err == nil || !strings.Contains(err.Error(), "failed to read registry") {
		t.Fatalf("updateRegistryWithGit error = %v", err)
	}

	err = (&GitHubInstaller{AgentFieldHome: fileHome}).updateRegistryWithGitHub(
		&PackageMetadata{Name: "demo", Version: "1.0.0"},
		&GitHubPackageInfo{Owner: "acme", Repo: "repo", Ref: "main"},
		"/src",
		"/dst",
	)
	if err == nil || !strings.Contains(err.Error(), "failed to read registry") {
		t.Fatalf("updateRegistryWithGitHub error = %v", err)
	}
}

func TestGitInstallerInstallFromGitMoreCoverage(t *testing.T) {
	t.Run("install dependencies failure", func(t *testing.T) {
		home := t.TempDir()
		repo := filepath.Join(t.TempDir(), "repo")
		writeTestPackage(t, repo, "name: git-deps\nversion: 1.0.0\ndependencies:\n  python:\n    - missing-package\n")
		setupFakeGit(t, "copy", repo, false)

		gitDir := filepath.Dir(mustLookPath(t, "git"))
		t.Setenv("PATH", strings.Join([]string{gitDir, "/usr/bin", "/bin"}, string(os.PathListSeparator)))

		err := (&GitInstaller{AgentFieldHome: home}).InstallFromGit("https://gitlab.com/acme/repo", true)
		if err == nil || !strings.Contains(err.Error(), "failed to install dependencies") {
			t.Fatalf("InstallFromGit error = %v", err)
		}
	})

	t.Run("update registry failure", func(t *testing.T) {
		home := t.TempDir()
		repo := filepath.Join(t.TempDir(), "repo")
		writeTestPackage(t, repo, "name: git-registry\nversion: 1.0.0\n")
		setupFakeGit(t, "copy", repo, false)
		if err := os.WriteFile(filepath.Join(home, "installed.yaml"), []byte("installed: ["), 0644); err != nil {
			t.Fatalf("write registry: %v", err)
		}

		err := (&GitInstaller{AgentFieldHome: home}).InstallFromGit("https://gitlab.com/acme/repo", true)
		if err == nil || !strings.Contains(err.Error(), "failed to update registry") {
			t.Fatalf("InstallFromGit error = %v", err)
		}
	})
}

func mustLookPath(t *testing.T, name string) string {
	t.Helper()
	path, err := exec.LookPath(name)
	if err != nil {
		t.Fatalf("LookPath(%s): %v", name, err)
	}
	return path
}

func TestAgentNodeRunnerStartProcessVirtualenvCoverage(t *testing.T) {
	writePython := func(t *testing.T, path string) {
		t.Helper()
		if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
			t.Fatalf("mkdir python dir: %v", err)
		}
		if err := os.WriteFile(path, []byte("#!/bin/sh\nsleep 1\n"), 0755); err != nil {
			t.Fatalf("write python: %v", err)
		}
	}

	runCase := func(t *testing.T, pythonPath string) {
		t.Helper()
		pkgPath := t.TempDir()
		if err := os.WriteFile(filepath.Join(pkgPath, "main.py"), []byte("print('ok')\n"), 0644); err != nil {
			t.Fatalf("write main.py: %v", err)
		}
		writePython(t, filepath.Join(pkgPath, pythonPath))

		logPath := filepath.Join(t.TempDir(), "runner.log")
		cmd, err := (&AgentNodeRunner{}).startAgentNodeProcess(InstalledPackage{
			Path: pkgPath,
			Runtime: RuntimeInfo{
				LogFile: logPath,
			},
		}, 9010)
		if err != nil {
			t.Fatalf("startAgentNodeProcess: %v", err)
		}
		time.Sleep(50 * time.Millisecond)
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
	}

	t.Run("venv bin python", func(t *testing.T) {
		runCase(t, filepath.Join("venv", "bin", "python"))
	})

	t.Run("venv windows python", func(t *testing.T) {
		runCase(t, filepath.Join("venv", "Scripts", "python.exe"))
	})
}
