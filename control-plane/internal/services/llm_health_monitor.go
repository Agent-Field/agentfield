package services

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
)

// CircuitState represents the state of a circuit breaker.
type CircuitState string

const (
	CircuitClosed   CircuitState = "closed"    // Healthy — requests pass through
	CircuitOpen     CircuitState = "open"      // Unhealthy — requests fail fast
	CircuitHalfOpen CircuitState = "half_open" // Testing recovery — limited probes allowed
)

// LLMEndpointStatus represents the health status of a single LLM endpoint.
type LLMEndpointStatus struct {
	Name                string       `json:"name"`
	URL                 string       `json:"url"`
	CircuitState        CircuitState `json:"circuit_state"`
	Healthy             bool         `json:"healthy"`
	LastChecked         time.Time    `json:"last_checked"`
	LastSuccess         time.Time    `json:"last_success,omitempty"`
	LastError           string       `json:"last_error,omitempty"`
	ConsecutiveFailures int          `json:"consecutive_failures"`
	TotalChecks         int64        `json:"total_checks"`
	TotalFailures       int64        `json:"total_failures"`
	// Circuit breaker internals
	circuitOpenedAt   time.Time
	halfOpenSuccesses int
}

// LLMHealthMonitor monitors LLM backend health using circuit breaker pattern.
type LLMHealthMonitor struct {
	config     config.LLMHealthConfig
	httpClient *http.Client
	endpoints  map[string]*LLMEndpointStatus
	mu         sync.RWMutex
	stopCh     chan struct{}
	stopOnce   sync.Once
	uiService  *UIService
}

// NewLLMHealthMonitor creates a new LLM health monitor.
func NewLLMHealthMonitor(cfg config.LLMHealthConfig, uiService *UIService) *LLMHealthMonitor {
	if cfg.CheckInterval <= 0 {
		cfg.CheckInterval = 15 * time.Second
	}
	if cfg.CheckTimeout <= 0 {
		cfg.CheckTimeout = 5 * time.Second
	}
	if cfg.FailureThreshold <= 0 {
		cfg.FailureThreshold = 3
	}
	if cfg.RecoveryTimeout <= 0 {
		cfg.RecoveryTimeout = 30 * time.Second
	}
	if cfg.HalfOpenMaxProbes <= 0 {
		cfg.HalfOpenMaxProbes = 2
	}

	endpoints := make(map[string]*LLMEndpointStatus)
	for _, ep := range cfg.Endpoints {
		name := normalizeLLMEndpointName(ep.Name)
		endpoints[name] = &LLMEndpointStatus{
			Name:         name,
			URL:          ep.URL,
			CircuitState: CircuitClosed,
			Healthy:      true, // Assume healthy until proven otherwise
		}
	}

	return &LLMHealthMonitor{
		config:     cfg,
		httpClient: &http.Client{Timeout: cfg.CheckTimeout},
		endpoints:  endpoints,
		stopCh:     make(chan struct{}),
		uiService:  uiService,
	}
}

// Enabled reports whether monitoring is active.
func (m *LLMHealthMonitor) Enabled() bool {
	return m != nil && m.config.Enabled
}

// EndpointCount returns the number of configured endpoints.
func (m *LLMHealthMonitor) EndpointCount() int {
	if m == nil {
		return 0
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.endpoints)
}

// Start begins the health monitoring loop.
func (m *LLMHealthMonitor) Start() {
	if !m.config.Enabled || len(m.config.Endpoints) == 0 {
		logger.Logger.Info().Msg("LLM health monitor disabled or no endpoints configured")
		return
	}

	logger.Logger.Info().
		Int("endpoint_count", len(m.config.Endpoints)).
		Dur("check_interval", m.config.CheckInterval).
		Int("failure_threshold", m.config.FailureThreshold).
		Msg("Starting LLM health monitor")

	ticker := time.NewTicker(m.config.CheckInterval)
	defer ticker.Stop()

	// Initial check
	m.checkAllEndpoints()

	for {
		select {
		case <-m.stopCh:
			logger.Logger.Info().Msg("LLM health monitor stopped")
			return
		case <-ticker.C:
			m.checkAllEndpoints()
		}
	}
}

// Stop halts the monitoring loop.
func (m *LLMHealthMonitor) Stop() {
	m.stopOnce.Do(func() {
		close(m.stopCh)
	})
}

// IsEndpointHealthy checks if a specific LLM endpoint is available.
// Returns true if healthy or if no endpoints are configured (fail-open).
func (m *LLMHealthMonitor) IsEndpointHealthy(name string) bool {
	if !m.config.Enabled {
		return true // Fail-open when monitoring is disabled
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	ep, ok := m.endpoints[normalizeLLMEndpointName(name)]
	if !ok {
		return true // Unknown endpoint — fail-open
	}

	return ep.CircuitState != CircuitOpen
}

// IsAnyEndpointHealthy returns true if at least one configured LLM endpoint is healthy.
// Returns true if no endpoints are configured (fail-open).
func (m *LLMHealthMonitor) IsAnyEndpointHealthy() bool {
	if !m.config.Enabled || len(m.endpoints) == 0 {
		return true
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	for _, ep := range m.endpoints {
		if ep.CircuitState != CircuitOpen {
			return true
		}
	}
	return false
}

// GetAllStatuses returns a snapshot of all endpoint statuses.
func (m *LLMHealthMonitor) GetAllStatuses() []LLMEndpointStatus {
	m.mu.RLock()
	defer m.mu.RUnlock()

	statuses := make([]LLMEndpointStatus, 0, len(m.endpoints))
	for _, ep := range m.endpoints {
		statuses = append(statuses, *ep)
	}
	return statuses
}

// GetStatus returns the status of a specific endpoint.
func (m *LLMHealthMonitor) GetStatus(name string) (*LLMEndpointStatus, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	ep, ok := m.endpoints[normalizeLLMEndpointName(name)]
	if !ok {
		return nil, false
	}
	copied := *ep
	return &copied, true
}

func (m *LLMHealthMonitor) checkAllEndpoints() {
	for _, epCfg := range m.config.Endpoints {
		m.checkEndpoint(epCfg)
	}
}

func (m *LLMHealthMonitor) checkEndpoint(epCfg config.LLMEndpoint) {
	m.mu.Lock()
	ep, ok := m.endpoints[normalizeLLMEndpointName(epCfg.Name)]
	if !ok {
		m.mu.Unlock()
		return
	}

	// If circuit is open, check if recovery timeout has elapsed
	if ep.CircuitState == CircuitOpen {
		if time.Since(ep.circuitOpenedAt) >= m.config.RecoveryTimeout {
			ep.CircuitState = CircuitHalfOpen
			ep.halfOpenSuccesses = 0
			logger.Logger.Info().
				Str("endpoint", ep.Name).
				Msg("LLM circuit breaker transitioning to half-open")
		} else {
			// Still in open state, skip the actual check
			ep.TotalChecks++
			ep.LastChecked = time.Now()
			m.mu.Unlock()
			return
		}
	}
	m.mu.Unlock()

	// Perform the actual health check outside the lock
	healthy, errMsg := m.probeEndpoint(epCfg)

	m.mu.Lock()
	defer m.mu.Unlock()

	ep.TotalChecks++
	ep.LastChecked = time.Now()
	previousState := ep.CircuitState

	if healthy {
		m.handleSuccess(ep)
	} else {
		ep.LastError = errMsg
		m.handleFailure(ep)
	}

	// Log state transitions
	if ep.CircuitState != previousState {
		logger.Logger.Warn().
			Str("endpoint", ep.Name).
			Str("from", string(previousState)).
			Str("to", string(ep.CircuitState)).
			Str("last_error", ep.LastError).
			Msg("LLM circuit breaker state changed")
	}
}

func normalizeLLMEndpointName(name string) string {
	return strings.TrimSpace(strings.ToLower(name))
}

func (m *LLMHealthMonitor) handleSuccess(ep *LLMEndpointStatus) {
	ep.LastSuccess = time.Now()
	ep.ConsecutiveFailures = 0

	switch ep.CircuitState {
	case CircuitHalfOpen:
		ep.halfOpenSuccesses++
		if ep.halfOpenSuccesses >= m.config.HalfOpenMaxProbes {
			ep.CircuitState = CircuitClosed
			ep.Healthy = true
			logger.Logger.Info().
				Str("endpoint", ep.Name).
				Msg("LLM circuit breaker closed — endpoint recovered")
		}
	case CircuitClosed:
		ep.Healthy = true
	}
}

func (m *LLMHealthMonitor) handleFailure(ep *LLMEndpointStatus) {
	ep.ConsecutiveFailures++
	ep.TotalFailures++

	switch ep.CircuitState {
	case CircuitClosed:
		if ep.ConsecutiveFailures >= m.config.FailureThreshold {
			ep.CircuitState = CircuitOpen
			ep.Healthy = false
			ep.circuitOpenedAt = time.Now()
			logger.Logger.Error().
				Str("endpoint", ep.Name).
				Int("consecutive_failures", ep.ConsecutiveFailures).
				Str("last_error", ep.LastError).
				Msg("LLM circuit breaker opened — endpoint marked unhealthy")
		}
	case CircuitHalfOpen:
		// Any failure in half-open immediately re-opens the circuit
		ep.CircuitState = CircuitOpen
		ep.Healthy = false
		ep.circuitOpenedAt = time.Now()
		ep.halfOpenSuccesses = 0
		logger.Logger.Warn().
			Str("endpoint", ep.Name).
			Msg("LLM circuit breaker re-opened from half-open — recovery failed")
	}
}

func (m *LLMHealthMonitor) probeEndpoint(epCfg config.LLMEndpoint) (bool, string) {
	method := epCfg.Method
	if method == "" {
		method = http.MethodGet
	}

	ctx, cancel := context.WithTimeout(context.Background(), m.config.CheckTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, method, epCfg.URL, nil)
	if err != nil {
		return false, fmt.Sprintf("failed to create request: %v", err)
	}

	if epCfg.Header != "" {
		req.Header.Set("Authorization", epCfg.Header)
	}

	resp, err := m.httpClient.Do(req)
	if err != nil {
		return false, fmt.Sprintf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return true, ""
	}

	return false, fmt.Sprintf("unhealthy status code: %d", resp.StatusCode)
}
