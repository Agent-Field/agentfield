package services

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/domain"
	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

// Mock RegistryStorage for package service testing
// Note: This uses packages.InstallationRegistry, not domain.InstallationRegistry
// because package_service.go uses loadRegistryDirect which works with packages types
type mockPackageRegistryStorage struct {
	loadFunc func() (*packages.InstallationRegistry, error)
	saveFunc func(*packages.InstallationRegistry) error
	registry *packages.InstallationRegistry
}

func newMockPackageRegistryStorage() *mockPackageRegistryStorage {
	return &mockPackageRegistryStorage{
		registry: &packages.InstallationRegistry{
			Installed: make(map[string]packages.InstalledPackage),
		},
	}
}

func (m *mockPackageRegistryStorage) LoadRegistry() (*domain.InstallationRegistry, error) {
	// Convert packages.InstallationRegistry to domain.InstallationRegistry
	if m.loadFunc != nil {
		pkgReg, err := m.loadFunc()
		if err != nil {
			return nil, err
		}
		return convertToDomainRegistry(pkgReg), nil
	}
	return convertToDomainRegistry(m.registry), nil
}

func (m *mockPackageRegistryStorage) SaveRegistry(registry *domain.InstallationRegistry) error {
	// Convert domain.InstallationRegistry to packages.InstallationRegistry
	if m.saveFunc != nil {
		pkgReg := convertToPackagesRegistry(registry)
		return m.saveFunc(pkgReg)
	}
	m.registry = convertToPackagesRegistry(registry)
	return nil
}

func (m *mockPackageRegistryStorage) GetPackage(name string) (*domain.InstalledPackage, error) {
	if pkg, ok := m.registry.Installed[name]; ok {
		return convertToDomainPackage(&pkg), nil
	}
	return nil, errors.New("package not found")
}

func (m *mockPackageRegistryStorage) SavePackage(name string, pkg *domain.InstalledPackage) error {
	pkgPkg := convertToPackagesPackage(name, pkg)
	if m.registry.Installed == nil {
		m.registry.Installed = make(map[string]packages.InstalledPackage)
	}
	m.registry.Installed[name] = pkgPkg
	return nil
}

// Helper functions to convert between domain and packages types
func convertToDomainRegistry(pkgReg *packages.InstallationRegistry) *domain.InstallationRegistry {
	domainReg := &domain.InstallationRegistry{
		Installed: make(map[string]domain.InstalledPackage),
	}
	for name, pkg := range pkgReg.Installed {
		domainReg.Installed[name] = *convertToDomainPackage(&pkg)
	}
	return domainReg
}

func convertToPackagesRegistry(domainReg *domain.InstallationRegistry) *packages.InstallationRegistry {
	pkgReg := &packages.InstallationRegistry{
		Installed: make(map[string]packages.InstalledPackage),
	}
	for name, pkg := range domainReg.Installed {
		pkgReg.Installed[name] = convertToPackagesPackage(name, &pkg)
	}
	return pkgReg
}

func convertToDomainPackage(pkg *packages.InstalledPackage) *domain.InstalledPackage {
	var installedAt time.Time
	if pkg.InstalledAt != "" {
		if parsed, err := time.Parse(time.RFC3339, pkg.InstalledAt); err == nil {
			installedAt = parsed
		}
	}
	return &domain.InstalledPackage{
		Name:        pkg.Name,
		Version:     pkg.Version,
		Path:        pkg.Path,
		Environment: make(map[string]string), // packages doesn't store this
		InstalledAt: installedAt,
	}
}

func convertToPackagesPackage(name string, pkg *domain.InstalledPackage) packages.InstalledPackage {
	return packages.InstalledPackage{
		Name:        name,
		Version:     pkg.Version,
		Path:        pkg.Path,
		InstalledAt: pkg.InstalledAt.Format(time.RFC3339),
		Status:      "stopped",
		Runtime:     packages.RuntimeInfo{},
	}
}

func TestNewPackageService(t *testing.T) {
	registryStorage := newMockPackageRegistryStorage()
	fileSystem := newMockFileSystemAdapter()
	agentfieldHome := "/tmp/test-agentfield"

	service := NewPackageService(registryStorage, fileSystem, agentfieldHome)

	assert.NotNil(t, service)
	ps, ok := service.(*DefaultPackageService)
	require.True(t, ok)
	assert.Equal(t, registryStorage, ps.registryStorage)
	assert.Equal(t, fileSystem, ps.fileSystem)
	assert.Equal(t, agentfieldHome, ps.agentfieldHome)
}

func TestInstallPackage_GitURL(t *testing.T) {
	registryStorage := newMockPackageRegistryStorage()
	fileSystem := newMockFileSystemAdapter()
	agentfieldHome := t.TempDir()

	service := NewPackageService(registryStorage, fileSystem, agentfieldHome).(*DefaultPackageService)

	options := domain.InstallOptions{
		Force:   false,
		Verbose: false,
	}

	// Test with a Git URL - this will call GitInstaller which is tested separately
	err := service.InstallPackage("https://github.com/user/repo.git", options)
	// This will fail because we don't have a real Git repository, but we verify it's handled
	assert.Error(t, err)
	// The error should be about Git installation, not about local package
	assert.NotContains(t, err.Error(), "agentfield-package.yaml")
}

func TestInstallLocalPackage_Success(t *testing.T) {
	tmpDir := t.TempDir()
	sourcePath := filepath.Join(tmpDir, "source-package")
	agentfieldHome := tmpDir

	// Create source package structure
	require.NoError(t, os.MkdirAll(sourcePath, 0755))
	packageYamlPath := filepath.Join(sourcePath, "agentfield-package.yaml")
	packageYamlContent := `name: test-package
version: 1.0.0
description: Test package
main: main.py
`
	require.NoError(t, os.WriteFile(packageYamlPath, []byte(packageYamlContent), 0644))

	mainPyPath := filepath.Join(sourcePath, "main.py")
	require.NoError(t, os.WriteFile(mainPyPath, []byte("# Test package"), 0644))

	registryStorage := newMockPackageRegistryStorage()
	fileSystem := newMockFileSystemAdapter()

	service := NewPackageService(registryStorage, fileSystem, agentfieldHome).(*DefaultPackageService)

	options := domain.InstallOptions{
		Force:   false,
		Verbose: false,
	}

	// This will fail at dependency installation or other steps, but we test the validation logic
	err := service.InstallPackage(sourcePath, options)
	// The error should not be about missing agentfield-package.yaml or main.py
	if err != nil {
		assert.NotContains(t, err.Error(), "agentfield-package.yaml not found")
		assert.NotContains(t, err.Error(), "main.py not found")
	}
}

func TestInstallLocalPackage_MissingPackageYaml(t *testing.T) {
	tmpDir := t.TempDir()
	sourcePath := filepath.Join(tmpDir, "source-package")
	require.NoError(t, os.MkdirAll(sourcePath, 0755))

	registryStorage := newMockPackageRegistryStorage()
	fileSystem := newMockFileSystemAdapter()
	agentfieldHome := tmpDir

	service := NewPackageService(registryStorage, fileSystem, agentfieldHome).(*DefaultPackageService)

	options := domain.InstallOptions{
		Force:   false,
		Verbose: false,
	}

	err := service.InstallPackage(sourcePath, options)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "agentfield-package.yaml")
}

func TestInstallLocalPackage_AlreadyInstalled(t *testing.T) {
	tmpDir := t.TempDir()
	sourcePath := filepath.Join(tmpDir, "source-package")
	agentfieldHome := tmpDir

	// Create source package structure
	require.NoError(t, os.MkdirAll(sourcePath, 0755))
	packageYamlPath := filepath.Join(sourcePath, "agentfield-package.yaml")
	packageYamlContent := `name: test-package
version: 1.0.0
description: Test package
main: main.py
`
	require.NoError(t, os.WriteFile(packageYamlPath, []byte(packageYamlContent), 0644))

	mainPyPath := filepath.Join(sourcePath, "main.py")
	require.NoError(t, os.WriteFile(mainPyPath, []byte("# Test package"), 0644))

	// Create registry with package already installed
	registryPath := filepath.Join(agentfieldHome, "installed.yaml")
	registry := &packages.InstallationRegistry{
		Installed: map[string]packages.InstalledPackage{
			"test-package": {
				Name:    "test-package",
				Version: "1.0.0",
				Path:    filepath.Join(agentfieldHome, "packages", "test-package"),
				Status:  "stopped",
			},
		},
	}
	data, err := yaml.Marshal(registry)
	require.NoError(t, err)
	require.NoError(t, os.MkdirAll(filepath.Dir(registryPath), 0755))
	require.NoError(t, os.WriteFile(registryPath, data, 0644))

	registryStorage := newMockPackageRegistryStorage()
	fileSystem := newMockFileSystemAdapter()

	service := NewPackageService(registryStorage, fileSystem, agentfieldHome).(*DefaultPackageService)

	options := domain.InstallOptions{
		Force:   false,
		Verbose: false,
	}

	err = service.InstallPackage(sourcePath, options)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "already installed")
}

func TestUninstallPackage_Success(t *testing.T) {
	tmpDir := t.TempDir()
	agentfieldHome := tmpDir
	packagePath := filepath.Join(agentfieldHome, "packages", "test-package")
	require.NoError(t, os.MkdirAll(packagePath, 0755))

	// Create registry with package installed
	registryPath := filepath.Join(agentfieldHome, "installed.yaml")
	registry := &packages.InstallationRegistry{
		Installed: map[string]packages.InstalledPackage{
			"test-package": {
				Name:    "test-package",
				Version: "1.0.0",
				Path:    packagePath,
				Status:  "stopped",
				Runtime: packages.RuntimeInfo{
					LogFile: filepath.Join(agentfieldHome, "logs", "test-package.log"),
				},
			},
		},
	}
	data, err := yaml.Marshal(registry)
	require.NoError(t, err)
	require.NoError(t, os.MkdirAll(filepath.Dir(registryPath), 0755))
	require.NoError(t, os.WriteFile(registryPath, data, 0644))

	registryStorage := newMockPackageRegistryStorage()
	fileSystem := newMockFileSystemAdapter()

	service := NewPackageService(registryStorage, fileSystem, agentfieldHome).(*DefaultPackageService)

	err = service.UninstallPackage("test-package")
	require.NoError(t, err)

	// Verify package directory is removed
	_, err = os.Stat(packagePath)
	assert.True(t, os.IsNotExist(err))
}

func TestUninstallPackage_NotInstalled(t *testing.T) {
	tmpDir := t.TempDir()
	agentfieldHome := tmpDir

	registryStorage := newMockPackageRegistryStorage()
	fileSystem := newMockFileSystemAdapter()

	service := NewPackageService(registryStorage, fileSystem, agentfieldHome).(*DefaultPackageService)

	err := service.UninstallPackage("nonexistent-package")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not installed")
}

func TestUninstallPackage_Running(t *testing.T) {
	tmpDir := t.TempDir()
	agentfieldHome := tmpDir
	packagePath := filepath.Join(agentfieldHome, "packages", "test-package")
	require.NoError(t, os.MkdirAll(packagePath, 0755))

	pid := 12345
	// Create registry with package running
	registryPath := filepath.Join(agentfieldHome, "installed.yaml")
	registry := &packages.InstallationRegistry{
		Installed: map[string]packages.InstalledPackage{
			"test-package": {
				Name:    "test-package",
				Version: "1.0.0",
				Path:    packagePath,
				Status:  "running",
				Runtime: packages.RuntimeInfo{
					PID:     &pid,
					LogFile: filepath.Join(agentfieldHome, "logs", "test-package.log"),
				},
			},
		},
	}
	data, err := yaml.Marshal(registry)
	require.NoError(t, err)
	require.NoError(t, os.MkdirAll(filepath.Dir(registryPath), 0755))
	require.NoError(t, os.WriteFile(registryPath, data, 0644))

	registryStorage := newMockPackageRegistryStorage()
	fileSystem := newMockFileSystemAdapter()

	service := NewPackageService(registryStorage, fileSystem, agentfieldHome).(*DefaultPackageService)

	err = service.UninstallPackage("test-package")
	// Should fail because package is running
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "currently running")
}

func TestListInstalledPackages(t *testing.T) {
	tmpDir := t.TempDir()
	agentfieldHome := tmpDir

	// Create registry with packages
	registryPath := filepath.Join(agentfieldHome, "installed.yaml")
	installedAt := time.Now().Format(time.RFC3339)
	registry := &packages.InstallationRegistry{
		Installed: map[string]packages.InstalledPackage{
			"package-1": {
				Name:        "package-1",
				Version:     "1.0.0",
				Path:        filepath.Join(agentfieldHome, "packages", "package-1"),
				InstalledAt: installedAt,
			},
			"package-2": {
				Name:        "package-2",
				Version:     "2.0.0",
				Path:        filepath.Join(agentfieldHome, "packages", "package-2"),
				InstalledAt: installedAt,
			},
		},
	}
	data, err := yaml.Marshal(registry)
	require.NoError(t, err)
	require.NoError(t, os.MkdirAll(filepath.Dir(registryPath), 0755))
	require.NoError(t, os.WriteFile(registryPath, data, 0644))

	registryStorage := newMockPackageRegistryStorage()
	fileSystem := newMockFileSystemAdapter()

	service := NewPackageService(registryStorage, fileSystem, agentfieldHome).(*DefaultPackageService)

	packages, err := service.ListInstalledPackages()
	require.NoError(t, err)
	assert.Len(t, packages, 2)
}

func TestGetPackageInfo_Success(t *testing.T) {
	tmpDir := t.TempDir()
	agentfieldHome := tmpDir

	// Create registry with package
	registryPath := filepath.Join(agentfieldHome, "installed.yaml")
	installedAt := time.Now().Format(time.RFC3339)
	registry := &packages.InstallationRegistry{
		Installed: map[string]packages.InstalledPackage{
			"test-package": {
				Name:        "test-package",
				Version:     "1.0.0",
				Path:        filepath.Join(agentfieldHome, "packages", "test-package"),
				InstalledAt: installedAt,
			},
		},
	}
	data, err := yaml.Marshal(registry)
	require.NoError(t, err)
	require.NoError(t, os.MkdirAll(filepath.Dir(registryPath), 0755))
	require.NoError(t, os.WriteFile(registryPath, data, 0644))

	registryStorage := newMockPackageRegistryStorage()
	fileSystem := newMockFileSystemAdapter()

	service := NewPackageService(registryStorage, fileSystem, agentfieldHome).(*DefaultPackageService)

	pkg, err := service.GetPackageInfo("test-package")
	require.NoError(t, err)
	assert.Equal(t, "test-package", pkg.Name)
	assert.Equal(t, "1.0.0", pkg.Version)
}

func TestGetPackageInfo_NotFound(t *testing.T) {
	tmpDir := t.TempDir()
	agentfieldHome := tmpDir

	registryStorage := newMockPackageRegistryStorage()
	fileSystem := newMockFileSystemAdapter()

	service := NewPackageService(registryStorage, fileSystem, agentfieldHome).(*DefaultPackageService)

	_, err := service.GetPackageInfo("nonexistent-package")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not installed")
}

func TestValidatePackage_Success(t *testing.T) {
	tmpDir := t.TempDir()
	packagePath := filepath.Join(tmpDir, "test-package")
	require.NoError(t, os.MkdirAll(packagePath, 0755))

	packageYamlPath := filepath.Join(packagePath, "agentfield-package.yaml")
	require.NoError(t, os.WriteFile(packageYamlPath, []byte("name: test"), 0644))

	mainPyPath := filepath.Join(packagePath, "main.py")
	require.NoError(t, os.WriteFile(mainPyPath, []byte("# test"), 0644))

	registryStorage := newMockPackageRegistryStorage()
	fileSystem := newMockFileSystemAdapter()
	agentfieldHome := tmpDir

	service := NewPackageService(registryStorage, fileSystem, agentfieldHome).(*DefaultPackageService)

	err := service.validatePackage(packagePath)
	assert.NoError(t, err)
}

func TestValidatePackage_MissingPackageYaml(t *testing.T) {
	tmpDir := t.TempDir()
	packagePath := filepath.Join(tmpDir, "test-package")
	require.NoError(t, os.MkdirAll(packagePath, 0755))

	registryStorage := newMockPackageRegistryStorage()
	fileSystem := newMockFileSystemAdapter()
	agentfieldHome := tmpDir

	service := NewPackageService(registryStorage, fileSystem, agentfieldHome).(*DefaultPackageService)

	err := service.validatePackage(packagePath)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "agentfield-package.yaml")
}

func TestValidatePackage_MissingMainPy(t *testing.T) {
	tmpDir := t.TempDir()
	packagePath := filepath.Join(tmpDir, "test-package")
	require.NoError(t, os.MkdirAll(packagePath, 0755))

	packageYamlPath := filepath.Join(packagePath, "agentfield-package.yaml")
	require.NoError(t, os.WriteFile(packageYamlPath, []byte("name: test"), 0644))

	registryStorage := newMockPackageRegistryStorage()
	fileSystem := newMockFileSystemAdapter()
	agentfieldHome := tmpDir

	service := NewPackageService(registryStorage, fileSystem, agentfieldHome).(*DefaultPackageService)

	err := service.validatePackage(packagePath)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "main.py")
}

func TestParsePackageMetadata_Success(t *testing.T) {
	tmpDir := t.TempDir()
	packagePath := filepath.Join(tmpDir, "test-package")
	require.NoError(t, os.MkdirAll(packagePath, 0755))

	packageYamlPath := filepath.Join(packagePath, "agentfield-package.yaml")
	packageYamlContent := `name: test-package
version: 1.0.0
description: Test package
main: main.py
`
	require.NoError(t, os.WriteFile(packageYamlPath, []byte(packageYamlContent), 0644))

	registryStorage := newMockPackageRegistryStorage()
	fileSystem := newMockFileSystemAdapter()
	agentfieldHome := tmpDir

	service := NewPackageService(registryStorage, fileSystem, agentfieldHome).(*DefaultPackageService)

	metadata, err := service.parsePackageMetadata(packagePath)
	require.NoError(t, err)
	assert.Equal(t, "test-package", metadata.Name)
	assert.Equal(t, "1.0.0", metadata.Version)
	assert.Equal(t, "main.py", metadata.Main)
}

func TestParsePackageMetadata_MissingName(t *testing.T) {
	tmpDir := t.TempDir()
	packagePath := filepath.Join(tmpDir, "test-package")
	require.NoError(t, os.MkdirAll(packagePath, 0755))

	packageYamlPath := filepath.Join(packagePath, "agentfield-package.yaml")
	packageYamlContent := `version: 1.0.0
description: Test package
`
	require.NoError(t, os.WriteFile(packageYamlPath, []byte(packageYamlContent), 0644))

	registryStorage := newMockPackageRegistryStorage()
	fileSystem := newMockFileSystemAdapter()
	agentfieldHome := tmpDir

	service := NewPackageService(registryStorage, fileSystem, agentfieldHome).(*DefaultPackageService)

	_, err := service.parsePackageMetadata(packagePath)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "name is required")
}

func TestIsPackageInstalled_Installed(t *testing.T) {
	tmpDir := t.TempDir()
	agentfieldHome := tmpDir

	registryPath := filepath.Join(agentfieldHome, "installed.yaml")
	registry := &packages.InstallationRegistry{
		Installed: map[string]packages.InstalledPackage{
			"test-package": {
				Name:    "test-package",
				Version: "1.0.0",
			},
		},
	}
	data, err := yaml.Marshal(registry)
	require.NoError(t, err)
	require.NoError(t, os.MkdirAll(filepath.Dir(registryPath), 0755))
	require.NoError(t, os.WriteFile(registryPath, data, 0644))

	registryStorage := newMockPackageRegistryStorage()
	fileSystem := newMockFileSystemAdapter()

	service := NewPackageService(registryStorage, fileSystem, agentfieldHome).(*DefaultPackageService)

	installed := service.isPackageInstalled("test-package")
	assert.True(t, installed)
}

func TestIsPackageInstalled_NotInstalled(t *testing.T) {
	tmpDir := t.TempDir()
	agentfieldHome := tmpDir

	registryStorage := newMockPackageRegistryStorage()
	fileSystem := newMockFileSystemAdapter()

	service := NewPackageService(registryStorage, fileSystem, agentfieldHome).(*DefaultPackageService)

	installed := service.isPackageInstalled("nonexistent-package")
	assert.False(t, installed)
}

func TestStopAgentNode_Success(t *testing.T) {
	tmpDir := t.TempDir()
	agentfieldHome := tmpDir

	pid := 12345
	agentNode := &packages.InstalledPackage{
		Name:    "test-package",
		Version: "1.0.0",
		Runtime: packages.RuntimeInfo{
			PID: &pid,
		},
	}

	registryStorage := newMockPackageRegistryStorage()
	fileSystem := newMockFileSystemAdapter()

	service := NewPackageService(registryStorage, fileSystem, agentfieldHome).(*DefaultPackageService)

	// This will fail because we can't easily mock os.FindProcess
	// But we verify the function exists and handles the PID check
	err := service.stopAgentNode(agentNode)
	// The error should not be about missing PID
	if err != nil {
		assert.NotContains(t, err.Error(), "no PID found")
	}
}

func TestStopAgentNode_NoPID(t *testing.T) {
	tmpDir := t.TempDir()
	agentfieldHome := tmpDir

	agentNode := &packages.InstalledPackage{
		Name:    "test-package",
		Version: "1.0.0",
		Runtime: packages.RuntimeInfo{
			PID: nil,
		},
	}

	registryStorage := newMockPackageRegistryStorage()
	fileSystem := newMockFileSystemAdapter()

	service := NewPackageService(registryStorage, fileSystem, agentfieldHome).(*DefaultPackageService)

	err := service.stopAgentNode(agentNode)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no PID found")
}
