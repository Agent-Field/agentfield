package observability

import (
	"context"
	"errors"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

func newRecordedTracer(t *testing.T) (*Tracer, *tracetest.SpanRecorder) {
	t.Helper()

	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))

	t.Cleanup(func() {
		_ = provider.Shutdown(context.Background())
	})

	return &Tracer{
		tracer:   provider.Tracer(instrumentationName),
		provider: provider,
	}, recorder
}

func attrsByKey(attrs []attribute.KeyValue) map[string]string {
	out := make(map[string]string, len(attrs))
	for _, attr := range attrs {
		out[string(attr.Key)] = attr.Value.AsString()
	}
	return out
}

func findEndedSpanByName(t *testing.T, spans []sdktrace.ReadOnlySpan, name string) sdktrace.ReadOnlySpan {
	t.Helper()

	for _, span := range spans {
		if span.Name() == name {
			return span
		}
	}

	t.Fatalf("span %q not found", name)
	return nil
}

func TestTracerRecordError(t *testing.T) {
	tracer, recorder := newRecordedTracer(t)

	_, span := tracer.StartExecutionSpan(context.Background(), "exec-1", "wf-1", "node-1")
	tracer.RecordError(span, errors.New("boom"))
	span.End()

	ended := recorder.Ended()
	require.Len(t, ended, 1)

	recorded := ended[0]
	assert.Equal(t, 2, int(recorded.Status().Code))
	assert.Empty(t, recorded.Status().Description)
	require.Len(t, recorded.Events(), 1)
	assert.Equal(t, "exception", recorded.Events()[0].Name)
}

func TestExecutionTracerHandleReasonerEvent(t *testing.T) {
	t.Run("heartbeat is ignored", func(t *testing.T) {
		tracer, recorder := newRecordedTracer(t)
		et := NewExecutionTracer(tracer)
		et.ctx = context.Background()

		et.handleReasonerEvent(events.ReasonerEvent{
			Type:      events.Heartbeat,
			Timestamp: events.ReasonerEvent{}.Timestamp,
		})

		assert.Empty(t, recorder.Ended())
	})

	t.Run("creates standalone span without parent context", func(t *testing.T) {
		tracer, recorder := newRecordedTracer(t)
		et := NewExecutionTracer(tracer)
		et.ctx = context.Background()

		et.handleReasonerEvent(events.ReasonerEvent{
			Type:       events.ReasonerUpdated,
			ReasonerID: "reasoner-1",
			NodeID:     "node-1",
			Status:     "online",
		})

		ended := recorder.Ended()
		require.Len(t, ended, 1)

		span := ended[0]
		assert.Equal(t, "step.reasoner", span.Name())
		assert.False(t, span.Parent().IsValid())
		assert.Equal(t, map[string]string{
			"agentfield.step.type":        "reasoner",
			"agentfield.step.id":          "reasoner-1",
			"agentfield.agent.node_id":    "node-1",
			"agentfield.reasoner.status":  "online",
			"agentfield.event.type":       string(events.ReasonerUpdated),
		}, attrsByKey(span.Attributes()))
	})

	t.Run("uses stored parent context when available", func(t *testing.T) {
		tracer, recorder := newRecordedTracer(t)
		et := NewExecutionTracer(tracer)
		et.ctx = context.Background()

		parentCtx, parentSpan := tracer.StartExecutionSpan(context.Background(), "exec-2", "wf-2", "node-2")
		et.ctxs["node-2"] = parentCtx

		et.handleReasonerEvent(events.ReasonerEvent{
			Type:       events.ReasonerOnline,
			ReasonerID: "reasoner-2",
			NodeID:     "node-2",
			Status:     "online",
		})
		parentSpan.End()

		child := findEndedSpanByName(t, recorder.Ended(), "step.reasoner")
		assert.Equal(t, parentSpan.SpanContext().TraceID(), child.SpanContext().TraceID())
		assert.Equal(t, parentSpan.SpanContext().SpanID(), child.Parent().SpanID())
	})
}

func TestExecutionTracerHandleExecutionEvent_TableDriven(t *testing.T) {
	t.Run("non-terminal execution events are added to tracked span", func(t *testing.T) {
		tracer, recorder := newRecordedTracer(t)
		et := NewExecutionTracer(tracer)
		et.ctx = context.Background()

		et.handleExecutionEvent(events.ExecutionEvent{
			Type:        events.ExecutionCreated,
			ExecutionID: "exec-3",
			WorkflowID:  "wf-3",
			AgentNodeID: "node-3",
			Status:      "created",
		})

		cases := []struct {
			eventType events.ExecutionEventType
			status    string
		}{
			{eventType: events.ExecutionUpdated, status: "running"},
			{eventType: events.ExecutionWaiting, status: "waiting"},
			{eventType: events.ExecutionPaused, status: "paused"},
			{eventType: events.ExecutionResumed, status: "resumed"},
		}

		for _, tc := range cases {
			et.handleExecutionEvent(events.ExecutionEvent{
				Type:        tc.eventType,
				ExecutionID: "exec-3",
				Status:      tc.status,
			})
		}

		et.handleExecutionEvent(events.ExecutionEvent{
			Type:        events.ExecutionCompleted,
			ExecutionID: "exec-3",
			Status:      "succeeded",
		})

		execution := findEndedSpanByName(t, recorder.Ended(), "execution")
		require.Len(t, execution.Events(), len(cases))

		for i, tc := range cases {
			assert.Equal(t, string(tc.eventType), execution.Events()[i].Name)
			require.Len(t, execution.Events()[i].Attributes, 1)
			assert.Equal(t, "agentfield.execution.status", string(execution.Events()[i].Attributes[0].Key))
			assert.Equal(t, tc.status, execution.Events()[i].Attributes[0].Value.AsString())
		}

		assert.Equal(t, "succeeded", attrsByKey(execution.Attributes())["agentfield.execution.final_status"])
	})

	t.Run("cancelled execution marks the span as error", func(t *testing.T) {
		tracer, recorder := newRecordedTracer(t)
		et := NewExecutionTracer(tracer)
		et.ctx = context.Background()

		et.handleExecutionEvent(events.ExecutionEvent{
			Type:        events.ExecutionStarted,
			ExecutionID: "exec-4",
			WorkflowID:  "wf-4",
			AgentNodeID: "node-4",
			Status:      "running",
		})

		et.handleExecutionEvent(events.ExecutionEvent{
			Type:        events.ExecutionCancelledEvent,
			ExecutionID: "exec-4",
			Status:      "cancelled",
		})

		execution := findEndedSpanByName(t, recorder.Ended(), "execution")
		assert.Equal(t, 2, int(execution.Status().Code))
		assert.Empty(t, execution.Status().Description)
		assert.Equal(t, "cancelled", attrsByKey(execution.Attributes())["agentfield.execution.final_status"])
	})

	t.Run("missing tracked span is a no-op", func(t *testing.T) {
		tracer, recorder := newRecordedTracer(t)
		et := NewExecutionTracer(tracer)
		et.ctx = context.Background()

		et.addExecutionEvent(events.ExecutionEvent{
			Type:        events.ExecutionUpdated,
			ExecutionID: "missing",
			Status:      "running",
		})
		et.endExecution(events.ExecutionEvent{
			Type:        events.ExecutionFailed,
			ExecutionID: "missing",
			Status:      "failed",
		}, true)

		assert.Empty(t, recorder.Ended())
	})
}
