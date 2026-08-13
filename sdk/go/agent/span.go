package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/sdk/go/types"
)

const (
	// spanEventQueueSize bounds the async queue used for non-terminal span
	// events. When the queue is full, "running" events are dropped — the
	// control plane creates the execution node from the terminal event
	// alone (upsert-on-event), so a dropped start event costs at most the
	// node's live "running" visibility, never the node itself.
	spanEventQueueSize = 256

	// spanPayloadMaxBytes caps the JSON size of input/result payloads
	// attached to span events. The events ingestion path stores payloads
	// verbatim with no blob offloading, so large research artifacts must
	// be truncated agent-side.
	spanPayloadMaxBytes = 16 * 1024

	// spanPayloadPreviewBytes is how much of an oversized payload is kept.
	spanPayloadPreviewBytes = 4 * 1024
)

// Span runs fn as a traced child execution of the execution context carried
// by ctx, emitting workflow events to the control plane so the call appears
// as a node in the run's DAG — the in-process equivalent of CallLocal for
// functions that are not registered reasoners.
//
// The child context is injected into the ctx passed to fn, so nested Span
// and CallLocal calls, Note, and structured execution logs all inherit
// correct lineage, to arbitrary depth. Concurrent Spans branched from the
// same ctx become siblings under the same parent.
//
// The terminal event ("succeeded"/"failed") is sent synchronously so a
// completed span is never left dangling in "running" state; the start event
// is sent asynchronously and may be shed under load. If fn panics, a
// "failed" terminal event is emitted and the panic is re-raised.
//
// name is used as the reasoner id on the trace node. input and the returned
// result are attached to the events after size capping; pass a digest rather
// than full artifacts for large payloads.
//
// A nil receiver runs fn untraced, so test code exercising wrapped functions
// with a nil *Agent keeps working.
func (a *Agent) Span(ctx context.Context, name string, input map[string]any, fn func(ctx context.Context) (any, error)) (result any, err error) {
	if a == nil {
		return fn(ctx)
	}
	parent := executionContextFrom(ctx)
	child := a.buildChildContext(parent, name)
	ctx = contextWithExecution(ctx, child)

	tracedInput := truncateTraceInput(input)
	a.enqueueSpanStart(child, tracedInput)

	start := time.Now()
	defer func() {
		durationMS := time.Since(start).Milliseconds()
		if r := recover(); r != nil {
			panicErr := fmt.Errorf("panic in span %q: %v", name, r)
			a.emitWorkflowEvent(child, "failed", tracedInput, nil, panicErr, durationMS)
			panic(r)
		}
		if err != nil {
			a.emitWorkflowEvent(child, "failed", tracedInput, nil, err, durationMS)
			return
		}
		a.emitWorkflowEvent(child, "succeeded", tracedInput, truncateTracePayload(result), nil, durationMS)
	}()

	result, err = fn(ctx)
	return result, err
}

// enqueueSpanStart queues the non-terminal "running" event for async
// delivery, dropping it if the queue is full.
func (a *Agent) enqueueSpanStart(execCtx ExecutionContext, input map[string]any) {
	if strings.TrimSpace(a.cfg.AgentFieldURL) == "" {
		return
	}

	a.spanEventOnce.Do(func() {
		a.spanEventCh = make(chan types.WorkflowExecutionEvent, spanEventQueueSize)
		go func() {
			for event := range a.spanEventCh {
				if sendErr := a.sendWorkflowEvent(event); sendErr != nil {
					a.logger.Printf("span start event send failed: %v", sendErr)
				}
			}
		}()
	})

	event := types.WorkflowExecutionEvent{
		ExecutionID: execCtx.ExecutionID,
		WorkflowID:  execCtx.WorkflowID,
		RunID:       execCtx.RunID,
		ReasonerID:  execCtx.ReasonerName,
		Type:        execCtx.ReasonerName,
		AgentNodeID: a.cfg.NodeID,
		Status:      "running",
	}
	if execCtx.ParentExecutionID != "" {
		event.ParentExecutionID = &execCtx.ParentExecutionID
	}
	if execCtx.ParentWorkflowID != "" {
		event.ParentWorkflowID = &execCtx.ParentWorkflowID
	}
	if input != nil {
		event.InputData = input
	}

	select {
	case a.spanEventCh <- event:
	default:
		// Queue full: shed the start event. The terminal event will still
		// create the node control-plane-side.
	}
}

// truncateTracePayload caps an arbitrary payload's serialized size for
// attachment to a workflow event. Oversized payloads are replaced by a
// marker object carrying a preview and the original size.
func truncateTracePayload(v any) any {
	if v == nil {
		return nil
	}
	raw, err := json.Marshal(v)
	if err != nil {
		return map[string]any{"_unserializable": fmt.Sprintf("%T", v)}
	}
	if len(raw) <= spanPayloadMaxBytes {
		return v
	}
	return map[string]any{
		"_truncated":  true,
		"total_bytes": len(raw),
		"preview":     string(raw[:spanPayloadPreviewBytes]),
	}
}

// truncateTraceInput is truncateTracePayload specialized to the event
// InputData field, which must remain a map. Unlike results (marshaled
// synchronously before the caller resumes), the input is also serialized
// later by the async start-event goroutine, so it must be a deep copy —
// returning the caller's map would race with fn mutating it.
func truncateTraceInput(input map[string]any) map[string]any {
	if input == nil {
		return nil
	}
	raw, err := json.Marshal(input)
	if err != nil {
		return map[string]any{"_unserializable": true}
	}
	if len(raw) > spanPayloadMaxBytes {
		return map[string]any{
			"_truncated":  true,
			"total_bytes": len(raw),
			"preview":     string(raw[:spanPayloadPreviewBytes]),
		}
	}
	var copied map[string]any
	if err := json.Unmarshal(raw, &copied); err != nil {
		return map[string]any{"_unserializable": true}
	}
	return copied
}
