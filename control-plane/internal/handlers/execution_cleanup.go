package handlers

import (
	"context"
	"sync"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
)

// defaultInitialCleanupDelay is how long cleanupLoop waits before its first
// cleanup pass, giving the server time to finish starting up before the
// cleanup service touches the database.
const defaultInitialCleanupDelay = 30 * time.Second

// ExecutionCleanupService manages the background cleanup of old executions
type ExecutionCleanupService struct {
	storage   storage.StorageProvider
	payloads  services.PayloadStore
	config    config.ExecutionCleanupConfig
	stopChan  chan struct{}
	wg        sync.WaitGroup
	isRunning bool
	mu        sync.RWMutex

	// initialDelay is the wait before the first cleanup pass. It is always
	// defaultInitialCleanupDelay in production; tests shorten it so the
	// first-pass branch of cleanupLoop is reachable without a 30s sleep.
	initialDelay time.Duration

	// Metrics
	totalCleaned    int64
	lastCleanupTime time.Time
	lastCleanupErr  error
}

// NewExecutionCleanupService creates a new execution cleanup service
func NewExecutionCleanupService(storage storage.StorageProvider, config config.ExecutionCleanupConfig, payloads ...services.PayloadStore) *ExecutionCleanupService {
	service := &ExecutionCleanupService{
		storage:      storage,
		config:       config,
		stopChan:     make(chan struct{}),
		initialDelay: defaultInitialCleanupDelay,
	}
	if len(payloads) > 0 {
		service.payloads = payloads[0]
	}
	return service
}

// Start begins the background cleanup process
func (ecs *ExecutionCleanupService) Start(ctx context.Context) error {
	ecs.mu.Lock()
	defer ecs.mu.Unlock()

	if ecs.isRunning {
		return nil // Already running
	}

	if !ecs.config.Enabled {
		logger.Logger.Warn().Msg("Execution cleanup is disabled")
		return nil
	}
	if ecs.config.CleanupInterval <= 0 {
		logger.Logger.Warn().Dur("configured_interval", ecs.config.CleanupInterval).Dur("default_interval", config.DefaultExecutionCleanupInterval).Msg("invalid execution cleanup interval; using default")
		ecs.config.CleanupInterval = config.DefaultExecutionCleanupInterval
	}

	logger.Logger.Info().
		Bool("enabled", ecs.config.Enabled).
		Dur("retention_period", ecs.config.RetentionPeriod).
		Dur("cleanup_interval", ecs.config.CleanupInterval).
		Dur("stale_timeout", ecs.config.StaleExecutionTimeout).
		Int("batch_size", ecs.config.BatchSize).
		Msg("execution cleanup configuration")

	ecs.isRunning = true
	ecs.wg.Add(1)

	go ecs.cleanupLoop(ctx)

	return nil
}

// Stop stops the background cleanup process
func (ecs *ExecutionCleanupService) Stop() error {
	ecs.mu.Lock()
	if !ecs.isRunning {
		ecs.mu.Unlock()
		return nil // Already stopped
	}

	logger.Logger.Debug().Msg("Stopping execution cleanup service")

	close(ecs.stopChan)
	ecs.isRunning = false
	ecs.mu.Unlock()

	ecs.wg.Wait()

	logger.Logger.Debug().Msg("Execution cleanup service stopped")
	return nil
}

// GetMetrics returns cleanup metrics
func (ecs *ExecutionCleanupService) GetMetrics() (totalCleaned int64, lastCleanupTime time.Time, lastError error) {
	ecs.mu.RLock()
	defer ecs.mu.RUnlock()

	return ecs.totalCleaned, ecs.lastCleanupTime, ecs.lastCleanupErr
}

// cleanupLoop runs the periodic cleanup process
func (ecs *ExecutionCleanupService) cleanupLoop(ctx context.Context) {
	defer ecs.wg.Done()

	ticker := time.NewTicker(ecs.config.CleanupInterval)
	defer ticker.Stop()

	// Run initial cleanup after a short delay
	initialDelay := time.NewTimer(ecs.initialDelay)
	defer initialDelay.Stop()

	for {
		select {
		case <-ctx.Done():
			logger.Logger.Debug().Msg("Execution cleanup loop stopped due to context cancellation")
			return
		case <-ecs.stopChan:
			logger.Logger.Debug().Msg("Execution cleanup loop stopped")
			return
		case <-initialDelay.C:
			ecs.performCleanup(ctx)
		case <-ticker.C:
			ecs.performCleanup(ctx)
		}
	}
}

// performCleanup executes the actual cleanup operation
func (ecs *ExecutionCleanupService) performCleanup(ctx context.Context) {
	startTime := time.Now()

	logger.Logger.Debug().
		Dur("retention_period", ecs.config.RetentionPeriod).
		Int("batch_size", ecs.config.BatchSize).
		Msg("Starting execution cleanup")

	// Create a context with timeout for the cleanup operation
	cleanupCtx, cancel := context.WithTimeout(ctx, 5*time.Minute)
	defer cancel()
	ecs.sweepOrphans(cleanupCtx)

	// Perform cleanup in batches until no more executions to clean
	totalCleaned := 0
	payloadURIListErrors := 0
	effectiveRetention := storage.EffectiveExecutionRetention(ecs.config.RetentionPeriod, ecs.config.PreserveRecentDuration)
	if ecs.config.StaleExecutionTimeout > 0 {
		// Retry eligible workflow executions before marking anything as timed out
		if ecs.config.MaxRetries > 0 {
			retriedIDs, err := ecs.storage.RetryStaleWorkflowExecutions(
				cleanupCtx, ecs.config.StaleExecutionTimeout,
				ecs.config.MaxRetries, ecs.config.BatchSize,
			)
			if err != nil {
				logger.Logger.Error().Err(err).Msg("failed to retry stale workflow executions")
			} else if len(retriedIDs) > 0 {
				logger.Logger.Info().
					Int("retried", len(retriedIDs)).
					Int("max_retries", ecs.config.MaxRetries).
					Dur("stale_timeout", ecs.config.StaleExecutionTimeout).
					Msg("retried stale workflow executions")
			}
		}

		timedOut, err := ecs.storage.MarkStaleExecutions(cleanupCtx, ecs.config.StaleExecutionTimeout, ecs.config.BatchSize)
		if err != nil {
			logger.Logger.Error().Err(err).Msg("failed to mark stale executions as timed out")
		} else if timedOut > 0 {
			logger.Logger.Debug().
				Int("timed_out", timedOut).
				Dur("stale_timeout", ecs.config.StaleExecutionTimeout).
				Msg("marked stale executions as timed out")
		}

		wfTimedOut, err := ecs.storage.MarkStaleWorkflowExecutions(cleanupCtx, ecs.config.StaleExecutionTimeout, ecs.config.BatchSize)
		if err != nil {
			logger.Logger.Error().Err(err).Msg("failed to mark stale workflow executions as timed out")
		} else if wfTimedOut > 0 {
			logger.Logger.Debug().
				Int("timed_out", wfTimedOut).
				Dur("stale_timeout", ecs.config.StaleExecutionTimeout).
				Msg("marked stale workflow executions as timed out")
		}
	}

	for {
		var payloadURIs []string
		if ecs.payloads != nil && effectiveRetention > 0 {
			if source, ok := ecs.storage.(interface {
				ListExpiredExecutionPayloadURIs(context.Context, time.Duration, int) ([]string, error)
			}); ok {
				var err error
				payloadURIs, err = source.ListExpiredExecutionPayloadURIs(cleanupCtx, effectiveRetention, ecs.config.BatchSize)
				if err != nil {
					payloadURIListErrors++
					logger.Logger.Warn().Err(err).Msg("failed to list expired execution payload URIs")
				}
			}
		}
		cleaned, err := ecs.storage.CleanupOldExecutions(cleanupCtx, effectiveRetention, ecs.config.BatchSize)
		if err != nil {
			ecs.mu.Lock()
			ecs.lastCleanupErr = err
			ecs.lastCleanupTime = time.Now()
			ecs.mu.Unlock()

			logger.Logger.Error().
				Err(err).
				Int("total_cleaned_before_error", totalCleaned).
				Msg("Failed to cleanup old executions")
			return
		}

		totalCleaned += cleaned
		for _, uri := range payloadURIs {
			if err := ecs.payloads.Remove(cleanupCtx, uri); err != nil {
				logger.Logger.Warn().Err(err).Str("uri", uri).Msg("failed to remove retained execution payload")
			}
		}

		// If we cleaned fewer than the batch size, we're done
		if cleaned < ecs.config.BatchSize {
			break
		}

		// Check if context is cancelled between batches
		if cleanupCtx.Err() != nil {
			logger.Logger.Warn().
				Err(cleanupCtx.Err()).
				Int("total_cleaned", totalCleaned).
				Msg("Execution cleanup cancelled")
			return
		}

		// Small delay between batches to avoid overwhelming the database
		time.Sleep(100 * time.Millisecond)
	}

	duration := time.Since(startTime)

	// Update metrics
	ecs.mu.Lock()
	ecs.totalCleaned += int64(totalCleaned)
	ecs.lastCleanupTime = time.Now()
	ecs.lastCleanupErr = nil
	ecs.mu.Unlock()

	if totalCleaned > 0 {
		logger.Logger.Debug().
			Int("cleaned_count", totalCleaned).
			Int("payload_uri_list_errors", payloadURIListErrors).
			Dur("duration", duration).
			Msg("Execution cleanup completed")
	} else {
		logger.Logger.Debug().
			Int("payload_uri_list_errors", payloadURIListErrors).
			Dur("duration", duration).
			Msg("Execution cleanup completed - no executions to clean")
	}
}

func (ecs *ExecutionCleanupService) sweepOrphans(ctx context.Context) {
	if ecs.payloads == nil {
		return
	}
	sweeper, ok := ecs.payloads.(interface {
		Sweep(context.Context, map[string]struct{}, time.Duration, int) (int, int, error)
	})
	if !ok {
		return
	}
	source, ok := ecs.storage.(interface {
		ListPayloadURIs(context.Context) (map[string]struct{}, error)
	})
	if !ok {
		return
	}
	references, err := source.ListPayloadURIs(ctx)
	if err != nil {
		logger.Logger.Warn().Err(err).Msg("failed to list payload references for orphan sweep")
		return
	}
	inspected, removed, err := sweeper.Sweep(ctx, references, ecs.config.PayloadOrphanGrace, 10000)
	if err != nil {
		logger.Logger.Warn().Err(err).Msg("payload orphan sweep failed")
		return
	}
	logger.Logger.Info().Int("inspected", inspected).Int("removed", removed).Msg("payload orphan sweep completed")
}

// ForceCleanup performs an immediate cleanup operation (useful for testing or manual triggers)
func (ecs *ExecutionCleanupService) ForceCleanup(ctx context.Context) (int, error) {
	logger.Logger.Debug().Msg("Force cleanup requested")

	cleanupCtx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()

	totalCleaned := 0
	effectiveRetention := storage.EffectiveExecutionRetention(ecs.config.RetentionPeriod, ecs.config.PreserveRecentDuration)
	for {
		cleaned, err := ecs.storage.CleanupOldExecutions(cleanupCtx, effectiveRetention, ecs.config.BatchSize)
		if err != nil {
			return totalCleaned, err
		}

		totalCleaned += cleaned

		// If we cleaned fewer than the batch size, we're done
		if cleaned < ecs.config.BatchSize {
			break
		}

		// Check if context is cancelled between batches
		if cleanupCtx.Err() != nil {
			return totalCleaned, cleanupCtx.Err()
		}
	}

	// Update metrics
	ecs.mu.Lock()
	ecs.totalCleaned += int64(totalCleaned)
	ecs.lastCleanupTime = time.Now()
	ecs.lastCleanupErr = nil
	ecs.mu.Unlock()

	logger.Logger.Debug().
		Int("cleaned_count", totalCleaned).
		Msg("Force cleanup completed")

	return totalCleaned, nil
}
