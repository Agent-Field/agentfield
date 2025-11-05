package handlers

import (
	"context"
	"github.com/your-org/agentfield/control-plane/internal/config"
	"github.com/your-org/agentfield/control-plane/internal/logger"
	"github.com/your-org/agentfield/control-plane/internal/storage"
	"sync"
	"time"
)

// ExecutionCleanupService manages the background cleanup of old executions
type ExecutionCleanupService struct {
	storage   storage.StorageProvider
	config    config.ExecutionCleanupConfig
	stopChan  chan struct{}
	wg        sync.WaitGroup
	isRunning bool
	mu        sync.RWMutex

	// Metrics
	totalCleaned    int64
	lastCleanupTime time.Time
	lastCleanupErr  error
}

// NewExecutionCleanupService creates a new execution cleanup service
func NewExecutionCleanupService(storage storage.StorageProvider, config config.ExecutionCleanupConfig) *ExecutionCleanupService {
	return &ExecutionCleanupService{
		storage:  storage,
		config:   config,
		stopChan: make(chan struct{}),
	}
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

	logger.Logger.Debug().
		Dur("retention_period", ecs.config.RetentionPeriod).
		Dur("cleanup_interval", ecs.config.CleanupInterval).
		Int("batch_size", ecs.config.BatchSize).
		Msg("Starting execution cleanup service")

	ecs.isRunning = true
	ecs.wg.Add(1)

	go ecs.cleanupLoop(ctx)

	return nil
}

// Stop stops the background cleanup process
func (ecs *ExecutionCleanupService) Stop() error {
	ecs.mu.Lock()
	defer ecs.mu.Unlock()

	if !ecs.isRunning {
		return nil // Already stopped
	}

	logger.Logger.Debug().Msg("Stopping execution cleanup service")

	close(ecs.stopChan)
	ecs.wg.Wait()
	ecs.isRunning = false

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
	initialDelay := time.NewTimer(30 * time.Second)
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

	// Perform cleanup in batches until no more executions to clean
	totalCleaned := 0
	if ecs.config.StaleExecutionTimeout > 0 {
		timedOut, err := ecs.storage.MarkStaleExecutions(cleanupCtx, ecs.config.StaleExecutionTimeout, ecs.config.BatchSize)
		if err != nil {
			logger.Logger.Error().Err(err).Msg("failed to mark stale executions as timed out")
		} else if timedOut > 0 {
			logger.Logger.Debug().
				Int("timed_out", timedOut).
				Dur("stale_timeout", ecs.config.StaleExecutionTimeout).
				Msg("marked stale executions as timed out")
		}
	}

	for {
		cleaned, err := ecs.storage.CleanupOldExecutions(cleanupCtx, ecs.config.RetentionPeriod, ecs.config.BatchSize)
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
			Dur("duration", duration).
			Msg("Execution cleanup completed")
	} else {
		logger.Logger.Debug().
			Dur("duration", duration).
			Msg("Execution cleanup completed - no executions to clean")
	}
}

// ForceCleanup performs an immediate cleanup operation (useful for testing or manual triggers)
func (ecs *ExecutionCleanupService) ForceCleanup(ctx context.Context) (int, error) {
	logger.Logger.Debug().Msg("Force cleanup requested")

	cleanupCtx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()

	totalCleaned := 0
	for {
		cleaned, err := ecs.storage.CleanupOldExecutions(cleanupCtx, ecs.config.RetentionPeriod, ecs.config.BatchSize)
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
