package services

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"
)

func TestRecordQueueDepthClampsNegative(t *testing.T) {
	recordQueueDepth(-10)
	require.Equal(t, float64(0), testutil.ToFloat64(queueDepthGauge))
}

func TestRecordWaiterCountClampsNegative(t *testing.T) {
	recordWaiterCount(-5)
	require.Equal(t, float64(0), testutil.ToFloat64(waiterInflightGauge))
}

func TestNormalizeAgentLabel(t *testing.T) {
	require.Equal(t, "worker", normalizeAgentLabel(" worker "))
	require.Equal(t, "unknown", normalizeAgentLabel(""))
}

func TestRecordGatewayBackpressure(t *testing.T) {
	RecordGatewayBackpressure("Queue_Full")
	require.Equal(t, float64(1), testutil.ToFloat64(backpressureCounter.WithLabelValues("queue_full")))
}
