// haxen/internal/infrastructure/process/port_manager.go
package process

import (
	"fmt"
	"net"

	"github.com/your-org/haxen/control-plane/internal/core/interfaces"
)

// DefaultPortManager provides a default implementation for managing network ports.
// It keeps track of reserved ports in memory.
type DefaultPortManager struct {
	reservedPorts map[int]bool
}

// NewPortManager creates a new instance of DefaultPortManager.
// It initializes the map for tracking reserved ports.
func NewPortManager() interfaces.PortManager {
	return &DefaultPortManager{
		reservedPorts: make(map[int]bool),
	}
}

// FindFreePort searches for an available port within a specific range (startPort to startPort+100).
// It checks both system availability and internal reservations.
// Returns the first free port found, or an error if no port is available in the range.
func (pm *DefaultPortManager) FindFreePort(startPort int) (int, error) {
	// The search range is typically small, e.g., 100 ports.
	// This can be made configurable if needed.
	for port := startPort; port <= startPort+100; port++ {
		if pm.IsPortAvailable(port) && !pm.reservedPorts[port] {
			return port, nil
		}
	}
	return 0, fmt.Errorf("no free port available in range %d-%d", startPort, startPort+100)
}

// IsPortAvailable checks if a specific port can be listened on.
// Returns true if the port is available (can be opened for listening), false otherwise.
func (pm *DefaultPortManager) IsPortAvailable(port int) bool {
	address := fmt.Sprintf(":%d", port)
	listener, err := net.Listen("tcp", address)
	if err != nil {
		return false // Port is likely in use or not permissible to use
	}
	// If listening was successful, close the listener immediately as we only wanted to check.
	listener.Close()
	return true
}

// ReservePort marks a port as reserved within this PortManager instance.
// It first checks if the port is system-available before reserving.
// Returns an error if the port is not available or cannot be reserved.
func (pm *DefaultPortManager) ReservePort(port int) error {
	if !pm.IsPortAvailable(port) {
		return fmt.Errorf("port %d is not available at the system level", port)
	}
	if pm.reservedPorts[port] {
		return fmt.Errorf("port %d is already reserved by this manager", port)
	}
	pm.reservedPorts[port] = true
	return nil
}

// ReleasePort removes a port from the internal reservation list.
// This makes the port available for future reservations by this manager.
// It does not affect the system-level availability of the port.
func (pm *DefaultPortManager) ReleasePort(port int) error {
	if _, ok := pm.reservedPorts[port]; !ok {
		return fmt.Errorf("port %d was not reserved by this manager", port)
	}
	delete(pm.reservedPorts, port)
	return nil
}
