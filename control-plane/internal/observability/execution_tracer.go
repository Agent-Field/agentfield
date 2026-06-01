package observability

import (
	"context"
	"fmt"
	"sync"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

// ExecutionTracer subscribes to the execution and reasoner event buses
// and translates events into OpenTelemetry spans.
type ExecutionTracer struct {
	tracer *Tracer

	mu    sync.Mutex
	spans map[string]trace.Span // keyed by execution_id
	ctxs  map[string]context.Context

	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// NewExecutionTracer creates a new execution tracer that bridges events to OTel spans.
func NewExecutionTracer(tracer *Tracer) *ExecutionTracer {
	return &ExecutionTracer{
		tracer: tracer,
		spans:  make(map[string]trace.Span),
		ctxs:   make(map[string]context.Context),
	}
}

// Start subscribes to event buses and begins translating events to spans.
func (et *ExecutionTracer) Start(ctx context.Context) {
	et.ctx, et.cancel = context.WithCancel(ctx)

	et.wg.Add(2)
	go et.subscribeExecutionEvents()
	go et.subscribeReasonerEvents()

	logger.Logger.Info().Msg("execution tracer started")
}

// Stop unsubscribes from event buses and ends any open spans.
func (et *ExecutionTracer) Stop() {
	if et.cancel != nil {
		et.cancel()
	}
	et.wg.Wait()

	// End any remaining open spans
	et.mu.Lock()
	for id, span := range et.spans {
		span.SetAttributes(attribute.String("agentfield.execution.end_reason", "tracer_shutdown"))
		span.End()
		delete(et.spans, id)
		delete(et.ctxs, id)
	}
	et.mu.Unlock()

	logger.Logger.Info().Msg("execution tracer stopped")
}

func (et *ExecutionTracer) subscribeExecutionEvents() {
	defer et.wg.Done()

	subscriberID := "otel-execution-tracer"
	ch := events.GlobalExecutionEventBus.Subscribe(subscriberID)
	defer events.GlobalExecutionEventBus.Unsubscribe(subscriberID)

	for {
		select {
		case <-et.ctx.Done():
			return
		case event, ok := <-ch:
			if !ok {
				return
			}
			et.handleExecutionEvent(event)
		}
	}
}

func (et *ExecutionTracer) subscribeReasonerEvents() {
	defer et.wg.Done()

	subscriberID := "otel-reasoner-tracer"
	ch := events.GlobalReasonerEventBus.Subscribe(subscriberID)
	defer events.GlobalReasonerEventBus.Unsubscribe(subscriberID)

	for {
		select {
		case <-et.ctx.Done():
			return
		case event, ok := <-ch:
			if !ok {
				return
			}
			et.handleReasonerEvent(event)
		}
	}
}

func (et *ExecutionTracer) handleExecutionEvent(event events.ExecutionEvent) {
	switch event.Type {
	case events.ExecutionCreated, events.ExecutionStarted:
		et.startExecution(event)
	case events.ExecutionCompleted:
		et.endExecution(event, false)
	case events.ExecutionFailed, events.ExecutionCancelledEvent:
		et.endExecution(event, true)
	case events.ExecutionUpdated, events.ExecutionWaiting, events.ExecutionPaused, events.ExecutionResumed:
		et.addExecutionEvent(event)
	}
}

func (et *ExecutionTracer) handleReasonerEvent(event events.ReasonerEvent) {
	if event.Type == events.Heartbeat {
		return
	}

	et.mu.Lock()
	parentCtx, exists := et.ctxs[event.NodeID]
	et.mu.Unlock()

	if !exists {
		// No parent execution span; create a standalone reasoner span
		parentCtx = et.ctx
	}

	_, span := et.tracer.StartStepSpan(parentCtx, "reasoner", event.ReasonerID, event.NodeID)
	span.SetAttributes(
		attribute.String("agentfield.reasoner.status", event.Status),
		attribute.String("agentfield.event.type", string(event.Type)),
	)

	// Reasoner events are point-in-time; end the span immediately.
	span.End()
}

func (et *ExecutionTracer) startExecution(event events.ExecutionEvent) {
	et.mu.Lock()
	defer et.mu.Unlock()

	if _, exists := et.spans[event.ExecutionID]; exists {
		return // already tracking
	}

	ctx, span := et.tracer.StartExecutionSpan(et.ctx, event.ExecutionID, event.WorkflowID, event.AgentNodeID)
	span.SetAttributes(
		attribute.String("agentfield.execution.status", event.Status),
	)

	et.spans[event.ExecutionID] = span
	et.ctxs[event.ExecutionID] = ctx

	logger.Logger.Debug().
		Str("execution_id", event.ExecutionID).
		Msg("started OTel execution span")
}

func (et *ExecutionTracer) endExecution(event events.ExecutionEvent, isError bool) {
	et.mu.Lock()
	span, exists := et.spans[event.ExecutionID]
	if exists {
		delete(et.spans, event.ExecutionID)
		delete(et.ctxs, event.ExecutionID)
	}
	et.mu.Unlock()

	if !exists {
		return
	}

	span.SetAttributes(
		attribute.String("agentfield.execution.final_status", event.Status),
	)

	if isError {
		span.SetStatus(2, fmt.Sprintf("execution %s: %s", event.Type, event.Status))
	}

	span.End()

	logger.Logger.Debug().
		Str("execution_id", event.ExecutionID).
		Bool("error", isError).
		Msg("ended OTel execution span")
}

func (et *ExecutionTracer) addExecutionEvent(event events.ExecutionEvent) {
	et.mu.Lock()
	span, exists := et.spans[event.ExecutionID]
	et.mu.Unlock()

	if !exists {
		return
	}

	span.AddEvent(string(event.Type),
		trace.WithAttributes(
			attribute.String("agentfield.execution.status", event.Status),
		),
	)
}
