//go:build windows

package services

// DevService is a stub for Windows builds.
type DevService struct{}

// NewDevService returns a stub DevService on Windows.
func NewDevService(processManager interface{}, portManager interface{}, fileSystem interface{}) *DevService {
	return &DevService{}
}
