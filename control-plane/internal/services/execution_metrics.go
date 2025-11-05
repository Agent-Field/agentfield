package services

import (
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	queueDepthGauge = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "agentfield_gateway_queue_depth",
		Help: "Number of workflow steps currently queued or in-flight for execution.",
	})

	workerInflightGauge = promauto.NewGaugeVec(prometheus.GaugeOpts{ //nolint:unused // Reserved for future use
		Name: "agentfield_worker_inflight",
		Help: "Number of active worker executions grouped by agent node.",
	}, []string{"agent"})

	stepDurationHistogram = promauto.NewHistogramVec(prometheus.HistogramOpts{ //nolint:unused // Reserved for future use
		Name:    "agentfield_step_duration_seconds",
		Help:    "Duration of workflow step executions split by terminal status.",
		Buckets: prometheus.DefBuckets,
	}, []string{"status"})

	stepRetriesCounter = promauto.NewCounterVec(prometheus.CounterOpts{ //nolint:unused // Reserved for future use
		Name: "agentfield_step_retries_total",
		Help: "Total number of workflow step retry attempts grouped by agent node.",
	}, []string{"agent"})

	waiterInflightGauge = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "agentfield_waiters_inflight",
		Help: "Number of synchronous waiter channels currently registered.",
	})

	backpressureCounter = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "agentfield_gateway_backpressure_total",
		Help: "Count of backpressure events emitted by the execution gateway grouped by reason.",
	}, []string{"reason"})
)

func recordQueueDepth(depth int64) {
	if depth < 0 {
		depth = 0
	}
	queueDepthGauge.Set(float64(depth))
}

func recordWaiterCount(count int) {
	if count < 0 {
		count = 0
	}
	waiterInflightGauge.Set(float64(count))
}

//nolint:unused // Reserved for future use
func recordWorkerAcquire(agent string) {
	workerInflightGauge.WithLabelValues(normalizeAgentLabel(agent)).Inc()
}

//nolint:unused // Reserved for future use
func recordWorkerRelease(agent string) {
	workerInflightGauge.WithLabelValues(normalizeAgentLabel(agent)).Dec()
}

//nolint:unused // Reserved for future use
func observeStepDuration(status string, duration time.Duration) {
	normalized := types.NormalizeExecutionStatus(status)
	stepDurationHistogram.WithLabelValues(normalized).Observe(duration.Seconds())
}

//nolint:unused // Reserved for future use
func incrementStepRetry(agent string) {
	stepRetriesCounter.WithLabelValues(normalizeAgentLabel(agent)).Inc()
}

func incrementBackpressure(reason string) {
	if reason == "" {
		reason = "unknown"
	}
	backpressureCounter.WithLabelValues(strings.ToLower(reason)).Inc()
}

// RecordGatewayBackpressure increments the counter for external callers (e.g. HTTP handlers).
func RecordGatewayBackpressure(reason string) {
	incrementBackpressure(reason)
}

func normalizeAgentLabel(agent string) string {
	agent = strings.TrimSpace(agent)
	if agent == "" {
		return "unknown"
	}
	return agent
}
