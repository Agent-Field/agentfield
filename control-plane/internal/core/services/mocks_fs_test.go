package services

// mockFileSystemAdapter is shared by test files across the package. It lives
// in its own untagged file (rather than dev_service_test.go, which is
// !windows) so that the platform-neutral tests that use it still compile
// under GOOS=windows.

import "errors"

// Mock FileSystemAdapter for testing
type mockFileSystemAdapter struct {
	readFileFunc      func(string) ([]byte, error)
	writeFileFunc     func(string, []byte) error
	existsFunc        func(string) bool
	createDirFunc     func(string) error
	listDirectoryFunc func(string) ([]string, error)
	files             map[string][]byte
	directories       map[string]bool
}

func newMockFileSystemAdapter() *mockFileSystemAdapter {
	return &mockFileSystemAdapter{
		files:       make(map[string][]byte),
		directories: make(map[string]bool),
	}
}

func (m *mockFileSystemAdapter) ReadFile(path string) ([]byte, error) {
	if m.readFileFunc != nil {
		return m.readFileFunc(path)
	}
	if data, ok := m.files[path]; ok {
		return data, nil
	}
	return nil, errors.New("file not found")
}

func (m *mockFileSystemAdapter) WriteFile(path string, data []byte) error {
	if m.writeFileFunc != nil {
		return m.writeFileFunc(path, data)
	}
	m.files[path] = data
	return nil
}

func (m *mockFileSystemAdapter) Exists(path string) bool {
	if m.existsFunc != nil {
		return m.existsFunc(path)
	}
	_, fileExists := m.files[path]
	_, dirExists := m.directories[path]
	return fileExists || dirExists
}

func (m *mockFileSystemAdapter) CreateDirectory(path string) error {
	if m.createDirFunc != nil {
		return m.createDirFunc(path)
	}
	m.directories[path] = true
	return nil
}

func (m *mockFileSystemAdapter) ListDirectory(path string) ([]string, error) {
	if m.listDirectoryFunc != nil {
		return m.listDirectoryFunc(path)
	}
	return []string{}, nil
}
