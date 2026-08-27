package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/ard"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/utils"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
)

func (c *executionController) tryHandleExternalARDCall(ctx *gin.Context) bool {
	if c.readARDConfig == nil {
		return false
	}
	targetParam := strings.TrimSpace(ctx.Param("target"))
	if !strings.HasPrefix(targetParam, "external.") {
		return false
	}

	var req ExecuteRequest
	if err := ctx.ShouldBindJSON(&req); err != nil {
		RespondBadRequest(ctx, "invalid request body: "+err.Error())
		return true
	}
	if req.Input == nil {
		req.Input = map[string]interface{}{}
	}

	reader, ok := c.store.(ard.StateReader)
	if !ok {
		RespondInternalError(ctx, "external ARD invocation state is unavailable")
		return true
	}
	state, err := ard.LoadStateReadOnly(ctx.Request.Context(), reader)
	if err != nil {
		RespondInternalError(ctx, err.Error())
		return true
	}
	effective := ard.Effective(c.readARDConfig(), state)
	if !effective.ExternalInvocationEnabled {
		RespondError(ctx, http.StatusForbidden, "external ARD invocation is disabled by config")
		return true
	}

	entry, binding, ok := externalARDBindingForTarget(state, targetParam)
	if !ok {
		RespondNotFound(ctx, "external ARD target is not imported and callable")
		return true
	}
	if err := validateExternalARDOperation(req, binding); err != nil {
		writeExecutionError(ctx, err)
		return true
	}

	headers := readExecutionHeaders(ctx)
	runID := headers.runID
	if runID == "" {
		runID = utils.GenerateRunID()
	}
	executionID := utils.GenerateExecutionID()
	start := time.Now().UTC()
	clientPayload := map[string]interface{}{"input": req.Input}
	if len(req.Context) > 0 {
		clientPayload["context"] = req.Context
	}
	storedPayload, err := json.Marshal(clientPayload)
	if err != nil {
		writeExecutionError(ctx, fmt.Errorf("encode execution payload: %w", err))
		return true
	}
	externalReasonerID := strings.TrimPrefix(targetParam, "external.")
	exec := &types.Execution{
		ExecutionID:       executionID,
		RunID:             runID,
		ParentExecutionID: headers.parentExecutionID,
		AgentNodeID:       "external",
		ReasonerID:        externalReasonerID,
		NodeID:            "external",
		Status:            types.ExecutionStatusRunning,
		InputPayload:      json.RawMessage(storedPayload),
		InputURI:          c.savePayload(ctx.Request.Context(), storedPayload),
		StartedAt:         start,
		CreatedAt:         start,
		UpdatedAt:         start,
	}
	if headers.sessionID != nil {
		exec.SessionID = headers.sessionID
	}
	if headers.actorID != nil {
		exec.ActorID = headers.actorID
	}
	if err := c.store.CreateExecutionRecord(ctx.Request.Context(), exec); err != nil {
		writeExecutionError(ctx, fmt.Errorf("create external ARD execution record: %w", err))
		return true
	}

	result, err := c.callExternalARD(ctx.Request.Context(), req, entry, binding, runID, executionID)
	if err != nil {
		_ = c.finishExternalARDExecution(ctx.Request.Context(), executionID, types.ExecutionStatusFailed, time.Since(start), nil, err)
		writeExecutionError(ctx, err)
		return true
	}
	resultBytes, err := json.Marshal(result)
	if err != nil {
		_ = c.finishExternalARDExecution(ctx.Request.Context(), executionID, types.ExecutionStatusFailed, time.Since(start), nil, err)
		writeExecutionError(ctx, fmt.Errorf("encode external ARD result: %w", err))
		return true
	}
	if err := c.finishExternalARDExecution(ctx.Request.Context(), executionID, types.ExecutionStatusSucceeded, time.Since(start), resultBytes, nil); err != nil {
		writeExecutionError(ctx, err)
		return true
	}

	ctx.Header("X-Execution-ID", executionID)
	ctx.Header("X-Run-ID", runID)
	ctx.JSON(http.StatusOK, ExecuteResponse{
		ExecutionID: executionID,
		RunID:       runID,
		Status:      types.ExecutionStatusSucceeded,
		Result:      result,
		DurationMS:  time.Since(start).Milliseconds(),
		FinishedAt:  time.Now().UTC().Format(time.RFC3339),
	})
	return true
}

func (c *executionController) callExternalARD(ctx context.Context, req ExecuteRequest, entry *ard.ExternalEntry, binding *ard.ExternalBinding, runID string, executionID string) (interface{}, error) {
	endpoint := strings.TrimSpace(entry.URL)
	if endpoint == "" {
		return nil, &callError{statusCode: http.StatusNotImplemented, message: "external ARD entry has no URL-backed adapter"}
	}
	parsed, err := url.Parse(endpoint)
	if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.User != nil {
		return nil, &callError{statusCode: http.StatusBadGateway, message: "external ARD entry URL is invalid"}
	}
	if err := services.ValidateWebhookURL(endpoint); err != nil {
		return nil, &callError{statusCode: http.StatusBadGateway, message: "external ARD entry URL rejected: " + err.Error()}
	}

	payload := map[string]interface{}{"input": req.Input}
	if len(req.Context) > 0 {
		payload["context"] = req.Context
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("encode external ARD request: %w", err)
	}

	timeout := time.Duration(binding.TimeoutMS) * time.Millisecond
	if timeout <= 0 {
		timeout = 30 * time.Second
	}
	client := services.NewSSRFSafeClient(timeout)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, parsed.String(), bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create external ARD request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("X-Run-ID", runID)
	httpReq.Header.Set("X-Execution-ID", executionID)
	httpReq.Header.Set("X-AgentField-ARD-Target", binding.LocalTarget)
	httpReq.Header.Set("X-AgentField-ARD-Adapter", binding.Adapter)

	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("external ARD call failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read external ARD response: %w", err)
	}
	if resp.StatusCode >= http.StatusBadRequest {
		return nil, &callError{
			statusCode: resp.StatusCode,
			message:    fmt.Sprintf("external ARD error (%d): %s", resp.StatusCode, truncateForLog(respBody)),
			body:       respBody,
		}
	}
	decoded := decodeJSON(respBody)
	if envelope, ok := decoded.(map[string]interface{}); ok {
		if result, ok := envelope["result"]; ok {
			return result, nil
		}
	}
	return decoded, nil
}

func (c *executionController) finishExternalARDExecution(ctx context.Context, executionID string, status string, elapsed time.Duration, result []byte, callErr error) error {
	resultURI := c.savePayload(ctx, result)
	_, err := c.store.UpdateExecutionRecord(ctx, executionID, func(current *types.Execution) (*types.Execution, error) {
		if current == nil {
			return nil, fmt.Errorf("execution %s not found", executionID)
		}
		now := time.Now().UTC()
		current.Status = status
		current.CompletedAt = pointerTime(now)
		duration := elapsed.Milliseconds()
		current.DurationMS = &duration
		current.UpdatedAt = now
		if len(result) > 0 {
			current.ResultPayload = json.RawMessage(result)
			current.ResultURI = resultURI
		}
		if callErr != nil {
			errMsg := callErr.Error()
			current.ErrorMessage = &errMsg
			category := string(classifyExecutionError(callErr))
			current.StatusReason = &category
		} else {
			current.ErrorMessage = nil
			current.StatusReason = nil
		}
		return current, nil
	})
	if err != nil {
		return fmt.Errorf("update external ARD execution record: %w", err)
	}
	return nil
}

func externalARDBindingForTarget(state ard.State, target string) (*ard.ExternalEntry, *ard.ExternalBinding, bool) {
	for entryID, binding := range state.Bindings {
		if !binding.Callable || strings.TrimSpace(binding.LocalTarget) != target {
			continue
		}
		for i := range state.Imports {
			if state.Imports[i].ID == entryID || state.Imports[i].ID == binding.ExternalEntryID {
				return &state.Imports[i], &binding, true
			}
		}
	}
	return nil, nil, false
}

func validateExternalARDOperation(req ExecuteRequest, binding *ard.ExternalBinding) error {
	if len(binding.AllowedOperations) == 0 {
		return nil
	}
	contextOperation := stringValue(req.Context["operation"])
	inputOperation := stringValue(req.Input["operation"])
	if contextOperation != "" && inputOperation != "" && !strings.EqualFold(contextOperation, inputOperation) {
		return &callError{statusCode: http.StatusBadRequest, message: "external ARD operation is ambiguous between input.operation and context.operation"}
	}
	operation := firstNonEmpty(contextOperation, inputOperation)
	if operation == "" {
		return &callError{statusCode: http.StatusForbidden, message: "external ARD operation is required by binding policy"}
	}
	for _, allowed := range binding.AllowedOperations {
		if strings.EqualFold(strings.TrimSpace(allowed), operation) {
			return nil
		}
	}
	return &callError{statusCode: http.StatusForbidden, message: fmt.Sprintf("external ARD operation %q is not allowed by binding policy", operation)}
}

func stringValue(value interface{}) string {
	switch typed := value.(type) {
	case nil:
		return ""
	case string:
		return strings.TrimSpace(typed)
	default:
		return strings.TrimSpace(fmt.Sprint(typed))
	}
}
