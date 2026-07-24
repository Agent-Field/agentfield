// agentfield/internal/infrastructure/process/port_manager.go
package process

import (
	"fmt"
	"net"
	"sync"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/interfaces"
)

// DefaultPortManager provides a default implementation for managing network ports.
// It keeps track of reserved ports in memory.
type DefaultPortManager struct {
	mu            sync.Mutex
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
	pm.mu.Lock()
	defer pm.mu.Unlock()
	for port := startPort; port <= startPort+100; port++ {
		if !pm.reservedPorts[port] && pm.IsPortAvailable(port) {
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
	// A successful bind is not proof on Windows: without SO_EXCLUSIVEADDRUSE a
	// probe bind can succeed while another process is actively listening on
	// the same port (observed with uvicorn agent nodes). If something accepts
	// a connection, the port is taken.
	if dial, err := net.DialTimeout("tcp", fmt.Sprintf("127.0.0.1:%d", port), 250*time.Millisecond); err == nil {
		dial.Close()
		return false
	}
	return true
}

// ReservePort marks a port as reserved within this PortManager instance.
// It first checks if the port is system-available before reserving.
// Returns an error if the port is not available or cannot be reserved.
func (pm *DefaultPortManager) ReservePort(port int) error {
	pm.mu.Lock()
	defer pm.mu.Unlock()
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
	pm.mu.Lock()
	defer pm.mu.Unlock()
	if _, ok := pm.reservedPorts[port]; !ok {
		return fmt.Errorf("port %d was not reserved by this manager", port)
	}
	delete(pm.reservedPorts, port)
	return nil
}
