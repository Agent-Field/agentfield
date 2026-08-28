package observability

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestInitTracer_ExportsToFullHTTPURL(t *testing.T) {
	var requests atomic.Int32
	collector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1/traces", r.URL.Path)
		requests.Add(1)
		w.Header().Set("Content-Type", "application/x-protobuf")
		w.WriteHeader(http.StatusOK)
	}))
	defer collector.Close()

	tracer, shutdown, err := InitTracer(context.Background(), TracerConfig{
		Enabled: true, Exporter: "otlp-http", Endpoint: collector.URL,
	})
	require.NoError(t, err)
	_, span := tracer.StartExecutionSpan(context.Background(), "exec-1", "run-1", "agent-1")
	span.End()
	require.NoError(t, tracer.provider.ForceFlush(context.Background()))
	require.NoError(t, shutdown(context.Background()))
	require.Positive(t, requests.Load())
}

func TestInitTracer_EndpointAndExporterValidation(t *testing.T) {
	_, _, err := InitTracer(context.Background(), TracerConfig{Enabled: true, Endpoint: "http://[bad"})
	require.ErrorContains(t, err, "invalid OTLP endpoint")

	tracer, shutdown, err := InitTracer(context.Background(), TracerConfig{
		Enabled: true, Exporter: "otlp-grpc", Endpoint: "http://127.0.0.1:4317",
	})
	require.NoError(t, err)
	require.NotNil(t, tracer)
	require.NoError(t, shutdown(context.Background()))
}

func TestResolveTraceEndpoint_DefaultsByExporter(t *testing.T) {
	assert.Equal(t, "localhost:4318", resolveTraceEndpoint("otlp-http", ""))
	assert.Equal(t, "localhost:4317", resolveTraceEndpoint("otlp-grpc", ""))
}

func TestInitTracer_Disabled(t *testing.T) {
	cfg := TracerConfig{Enabled: false}
	tracer, shutdown, err := InitTracer(context.Background(), cfg)

	require.NoError(t, err)
	assert.Nil(t, tracer)
	assert.NotNil(t, shutdown)

	// Shutdown should be a no-op when disabled
	err = shutdown(context.Background())
	assert.NoError(t, err)
}

func TestTracer_Enabled(t *testing.T) {
	var tracer *Tracer
	assert.False(t, tracer.Enabled(), "nil tracer should not be enabled")

	tracer = &Tracer{}
	assert.False(t, tracer.Enabled(), "tracer with nil inner tracer should not be enabled")
}

func TestTracerConfig_Defaults(t *testing.T) {
	cfg := TracerConfig{
		Enabled: true,
		// All other fields left empty to test defaults
	}

	// InitTracer will fail to connect since there's no OTLP endpoint,
	// but it should not error on creation (connection is lazy).
	tracer, shutdown, err := InitTracer(context.Background(), cfg)
	require.NoError(t, err)
	require.NotNil(t, tracer)
	require.NotNil(t, shutdown)

	assert.True(t, tracer.Enabled())

	// Clean up
	err = shutdown(context.Background())
	assert.NoError(t, err)
}

func TestStartExecutionSpan(t *testing.T) {
	cfg := TracerConfig{
		Enabled:     true,
		Endpoint:    "localhost:4318",
		ServiceName: "test-agentfield",
		Insecure:    true,
	}

	tracer, shutdown, err := InitTracer(context.Background(), cfg)
	require.NoError(t, err)
	require.NotNil(t, tracer)
	defer shutdown(context.Background()) //nolint:errcheck

	ctx, span := tracer.StartExecutionSpan(context.Background(), "exec-123", "wf-456", "node-789")
	assert.NotNil(t, ctx)
	assert.NotNil(t, span)
	assert.True(t, span.SpanContext().IsValid())
	assert.True(t, span.SpanContext().HasTraceID())
	assert.True(t, span.SpanContext().HasSpanID())
	span.End()
}

func TestStartStepSpan(t *testing.T) {
	cfg := TracerConfig{
		Enabled:     true,
		Endpoint:    "localhost:4318",
		ServiceName: "test-agentfield",
		Insecure:    true,
	}

	tracer, shutdown, err := InitTracer(context.Background(), cfg)
	require.NoError(t, err)
	require.NotNil(t, tracer)
	defer shutdown(context.Background()) //nolint:errcheck

	// Create a parent execution span first
	ctx, parentSpan := tracer.StartExecutionSpan(context.Background(), "exec-123", "wf-456", "node-789")
	defer parentSpan.End()

	// Create a child step span
	_, childSpan := tracer.StartStepSpan(ctx, "reasoner", "reasoner-abc", "node-789")
	assert.NotNil(t, childSpan)
	assert.True(t, childSpan.SpanContext().IsValid())

	// Verify parent-child relationship via trace ID
	assert.Equal(t, parentSpan.SpanContext().TraceID(), childSpan.SpanContext().TraceID(),
		"child span should share the same trace ID as parent")
	assert.NotEqual(t, parentSpan.SpanContext().SpanID(), childSpan.SpanContext().SpanID(),
		"child span should have a different span ID")

	childSpan.End()
}
