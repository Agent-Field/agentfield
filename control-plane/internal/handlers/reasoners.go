package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time" // Added for time.Now()

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/internal/utils" // Added for ID generation
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"      // Added for new types

	"github.com/gin-gonic/gin"
)

// ExecuteReasonerRequest represents a request to execute a reasoner
type ExecuteReasonerRequest struct {
	Input   map[string]interface{} `json:"input" binding:"required"`
	Context map[string]interface{} `json:"context,omitempty"`
}

func persistWorkflowExecution(ctx context.Context, storageProvider storage.StorageProvider, execution *types.WorkflowExecution) {
	if err := storageProvider.StoreWorkflowExecution(ctx, execution); err != nil {
		logger.Logger.Error().
			Err(err).
			Str("execution_id", execution.ExecutionID).
			Msg("failed to persist workflow execution state")
	}
}

// ExecuteReasonerResponse represents the response from executing a reasoner
type ExecuteReasonerResponse struct {
	Result    interface{} `json:"result"`
	NodeID    string      `json:"node_id"`
	Duration  int64       `json:"duration_ms"`
	Timestamp string      `json:"timestamp"`
}

// ExecuteReasonerHandler handles execution of reasoners with full tracking
func ExecuteReasonerHandler(storageProvider storage.StorageProvider) gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx := c.Request.Context()
		startTime := time.Now()

		// Generate AgentField Request ID
		agentfieldRequestID := utils.GenerateAgentFieldRequestID()

		// Extract headers
		workflowID := c.GetHeader("X-Workflow-ID")
		sessionID := c.GetHeader("X-Session-ID")
		actorID := c.GetHeader("X-Actor-ID")
		parentWorkflowID := c.GetHeader("X-Parent-Workflow-ID")
		parentExecutionID := c.GetHeader("X-Parent-Execution-ID")
		rootWorkflowID := c.GetHeader("X-Root-Workflow-ID")
		workflowName := c.GetHeader("X-Workflow-Name")
		workflowTagsHeader := c.GetHeader("X-Workflow-Tags")
		callerDID := c.GetHeader("X-Caller-DID")
		targetDID := c.GetHeader("X-Target-DID")
		agentNodeDID := c.GetHeader("X-Agent-Node-DID")

		// Generate Workflow ID if not provided
		if workflowID == "" {
			workflowID = utils.GenerateWorkflowID()
		}

		// Validate Workflow ID
		if !utils.ValidateWorkflowID(workflowID) {
			c.JSON(http.StatusBadRequest, gin.H{"error": "invalid workflow_id format"})
			return
		}

		// Generate Execution ID
		executionID := utils.GenerateExecutionID()

		// Parse reasoner ID from URL
		reasonerID := c.Param("reasoner_id")
		if reasonerID == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "reasoner_id is required"})
			return
		}

		// Split node_id and reasoner_name
		parts := strings.Split(reasonerID, ".")
		if len(parts) != 2 {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": "reasoner_id must be in format 'node_id.reasoner_name'",
			})
			return
		}

		nodeID := parts[0]
		reasonerName := parts[1]

		// Parse request body
		var req ExecuteReasonerRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		// Find the agent node
		targetNode, err := storageProvider.GetAgent(ctx, nodeID)
		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{
				"error": fmt.Sprintf("node '%s' not found", nodeID),
			})
			return
		}

		// Check if reasoner exists on the node
		reasonerExists := false
		for _, reasoner := range targetNode.Reasoners {
			if reasoner.ID == reasonerName {
				reasonerExists = true
				break
			}
		}

		if !reasonerExists {
			c.JSON(http.StatusNotFound, gin.H{
				"error": fmt.Sprintf("reasoner '%s' not found on node '%s'", reasonerName, nodeID),
			})
			return
		}

		// Create workflow execution record
		workflowExecution := &types.WorkflowExecution{
			WorkflowID:          workflowID,
			ExecutionID:         executionID,
			AgentFieldRequestID: agentfieldRequestID,
			AgentNodeID:         nodeID,
			ReasonerID:          reasonerName,
			Status:              "running",
			StartedAt:           startTime,
			CreatedAt:           startTime,
			UpdatedAt:           startTime,
		}

		// Set optional fields
		if sessionID != "" {
			workflowExecution.SessionID = &sessionID
		}
		if actorID != "" {
			workflowExecution.ActorID = &actorID
		}
		if parentWorkflowID != "" {
			workflowExecution.ParentWorkflowID = &parentWorkflowID
		}
		if parentExecutionID != "" {
			workflowExecution.ParentExecutionID = &parentExecutionID
		}
		if rootWorkflowID != "" {
			workflowExecution.RootWorkflowID = &rootWorkflowID
		}
		if workflowName != "" {
			workflowExecution.WorkflowName = &workflowName
		}

		// Parse workflow tags
		if workflowTagsHeader != "" {
			tags := strings.Split(workflowTagsHeader, ",")
			for i, tag := range tags {
				tags[i] = strings.TrimSpace(tag)
			}
			workflowExecution.WorkflowTags = tags
		}

		// Store input data
		inputJSON, err := json.Marshal(req.Input)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to marshal input"})
			return
		}
		workflowExecution.InputData = inputJSON
		workflowExecution.InputSize = len(inputJSON)

		// Prepare request to agent node with workflow context propagation
		agentURL := fmt.Sprintf("%s/reasoners/%s", targetNode.BaseURL, reasonerName)
		agentBody := inputJSON

		if targetNode.DeploymentType == "serverless" {
			target := &parsedTarget{
				NodeID:     nodeID,
				TargetName: reasonerName,
				TargetType: "reasoner",
			}
			var parentPtr, sessionPtr, actorPtr *string
			if parentExecutionID != "" {
				parentPtr = &parentExecutionID
			}
			if sessionID != "" {
				sessionPtr = &sessionID
			}
			if actorID != "" {
				actorPtr = &actorID
			}
			headers := executionHeaders{
				runID:             workflowID,
				parentExecutionID: parentPtr,
				sessionID:         sessionPtr,
				actorID:           actorPtr,
			}
			now := time.Now().UTC()
			exec := &types.Execution{
				ExecutionID:       executionID,
				RunID:             workflowID,
				ParentExecutionID: parentPtr,
				AgentNodeID:       nodeID,
				ReasonerID:        reasonerName,
				NodeID:            nodeID,
				Status:            types.ExecutionStatusRunning,
				StartedAt:         now,
				CreatedAt:         now,
				UpdatedAt:         now,
			}
			agentURL = buildAgentURL(targetNode, target)

			serverlessPayload, err := json.Marshal(buildServerlessPayload(target, exec, headers, req.Input))
			if err != nil {
				endTime := time.Now()
				workflowExecution.Status = types.ExecutionStatusFailed
				errorMsg := fmt.Sprintf("failed to encode serverless payload: %v", err)
				workflowExecution.ErrorMessage = &errorMsg
				workflowExecution.CompletedAt = &endTime
				duration := endTime.Sub(startTime).Milliseconds()
				workflowExecution.DurationMS = &duration
				workflowExecution.UpdatedAt = endTime
				persistWorkflowExecution(ctx, storageProvider, workflowExecution)

				c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to encode serverless payload"})
				return
			}
			agentBody = serverlessPayload
		}

		agentReq, err := http.NewRequestWithContext(ctx, http.MethodPost, agentURL, bytes.NewBuffer(agentBody))
		if err != nil {
			workflowExecution.Status = types.ExecutionStatusFailed
			errorMessage := fmt.Sprintf("failed to create agent request: %v", err)
			workflowExecution.ErrorMessage = &errorMessage
			endTime := time.Now()
			workflowExecution.CompletedAt = &endTime
			duration := endTime.Sub(startTime).Milliseconds()
			workflowExecution.DurationMS = &duration
			workflowExecution.UpdatedAt = endTime
			persistWorkflowExecution(ctx, storageProvider, workflowExecution)
			c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to create agent request"})
			return
		}

		agentReq.Header.Set("Content-Type", "application/json")
		agentReq.Header.Set("X-Workflow-ID", workflowID)
		agentReq.Header.Set("X-Execution-ID", executionID)
		agentReq.Header.Set("X-AgentField-Request-ID", agentfieldRequestID)
		if targetNode.DeploymentType == "serverless" {
			agentReq.Header.Set("X-Run-ID", workflowID)
		}
		if parentWorkflowID != "" {
			agentReq.Header.Set("X-Parent-Workflow-ID", parentWorkflowID)
		}
		if parentExecutionID != "" {
			agentReq.Header.Set("X-Parent-Execution-ID", parentExecutionID)
		}
		if rootWorkflowID != "" {
			agentReq.Header.Set("X-Root-Workflow-ID", rootWorkflowID)
		}
		if sessionID != "" {
			agentReq.Header.Set("X-Session-ID", sessionID)
		}
		if actorID != "" {
			agentReq.Header.Set("X-Actor-ID", actorID)
		}
		if workflowName != "" {
			agentReq.Header.Set("X-Workflow-Name", workflowName)
		}
		if workflowTagsHeader != "" {
			agentReq.Header.Set("X-Workflow-Tags", workflowTagsHeader)
		}
		if callerDID != "" {
			agentReq.Header.Set("X-Caller-DID", callerDID)
		}
		if targetDID != "" {
			agentReq.Header.Set("X-Target-DID", targetDID)
		}
		if agentNodeDID != "" {
			agentReq.Header.Set("X-Agent-Node-DID", agentNodeDID)
		}

		// Make HTTP request to agent node
		resp, err := http.DefaultClient.Do(agentReq)
		if err != nil {
			// Update execution with error
			endTime := time.Now()
			workflowExecution.Status = types.ExecutionStatusFailed
			errorMessage := err.Error()
			workflowExecution.ErrorMessage = &errorMessage
			workflowExecution.CompletedAt = &endTime
			duration := endTime.Sub(startTime).Milliseconds()
			workflowExecution.DurationMS = &duration
			workflowExecution.UpdatedAt = endTime

			// Store execution record
			persistWorkflowExecution(ctx, storageProvider, workflowExecution)

			c.JSON(http.StatusServiceUnavailable, gin.H{
				"error": fmt.Sprintf("failed to call agent node: %v", err),
			})
			return
		}
		defer resp.Body.Close()

		// Read response from agent node
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			// Update execution with error
			endTime := time.Now()
			workflowExecution.Status = types.ExecutionStatusFailed
			errorMsg := "failed to read agent response"
			workflowExecution.ErrorMessage = &errorMsg
			workflowExecution.CompletedAt = &endTime
			duration := endTime.Sub(startTime).Milliseconds()
			workflowExecution.DurationMS = &duration
			workflowExecution.UpdatedAt = endTime

			// Store execution record
			persistWorkflowExecution(ctx, storageProvider, workflowExecution)

			c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to read agent response"})
			return
		}

		// Parse agent response
		var result interface{}
		if err := json.Unmarshal(body, &result); err != nil {
			logger.Logger.Error().
				Err(err).
				Str("agent", nodeID).
				Str("agent_url", agentURL).
				Msgf("failed to decode agent response: %s", truncateForLog(body))
			// Update execution with error
			endTime := time.Now()
			workflowExecution.Status = types.ExecutionStatusFailed
			errorMsg := "failed to parse agent response"
			workflowExecution.ErrorMessage = &errorMsg
			workflowExecution.CompletedAt = &endTime
			duration := endTime.Sub(startTime).Milliseconds()
			workflowExecution.DurationMS = &duration
			workflowExecution.UpdatedAt = endTime

			// Store execution record
			persistWorkflowExecution(ctx, storageProvider, workflowExecution)

			c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to parse agent response"})
			return
		}

		// Update execution with success
		endTime := time.Now()
		workflowExecution.Status = types.ExecutionStatusSucceeded
		workflowExecution.OutputData = body
		workflowExecution.OutputSize = len(body)
		workflowExecution.CompletedAt = &endTime
		duration := endTime.Sub(startTime).Milliseconds()
		workflowExecution.DurationMS = &duration
		workflowExecution.UpdatedAt = endTime

		// Store execution record
		// Store execution record
		persistWorkflowExecution(ctx, storageProvider, workflowExecution)

		// Set response headers
		c.Header("X-Workflow-ID", workflowID)
		c.Header("X-Execution-ID", executionID)
		c.Header("X-AgentField-Request-ID", agentfieldRequestID)
		c.Header("X-Agent-Node-ID", nodeID)
		c.Header("X-Duration-MS", fmt.Sprintf("%d", duration))

		// Return successful response
		c.JSON(http.StatusOK, ExecuteReasonerResponse{
			Result:    result,
			NodeID:    nodeID,
			Duration:  duration,
			Timestamp: endTime.Format(time.RFC3339),
		})
	}
}

// ExecuteSkillHandler handles execution of skills via AgentField server
func ExecuteSkillHandler(storageProvider storage.StorageProvider) gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx := c.Request.Context()
		startTime := time.Now()

		// Generate AgentField Request ID
		agentfieldRequestID := utils.GenerateAgentFieldRequestID()

		// Extract headers
		workflowID := c.GetHeader("X-Workflow-ID")
		sessionID := c.GetHeader("X-Session-ID")
		actorID := c.GetHeader("X-Actor-ID")
		parentWorkflowID := c.GetHeader("X-Parent-Workflow-ID")
		parentExecutionID := c.GetHeader("X-Parent-Execution-ID")
		rootWorkflowID := c.GetHeader("X-Root-Workflow-ID")
		workflowName := c.GetHeader("X-Workflow-Name")
		workflowTagsHeader := c.GetHeader("X-Workflow-Tags")
		callerDID := c.GetHeader("X-Caller-DID")
		targetDID := c.GetHeader("X-Target-DID")
		agentNodeDID := c.GetHeader("X-Agent-Node-DID")

		// Generate Workflow ID if not provided
		if workflowID == "" {
			workflowID = utils.GenerateWorkflowID()
		}

		// Validate Workflow ID
		if !utils.ValidateWorkflowID(workflowID) {
			c.JSON(http.StatusBadRequest, gin.H{"error": "invalid workflow_id format"})
			return
		}

		// Generate Execution ID
		executionID := utils.GenerateExecutionID()

		// Parse skill ID from URL: node_id.skill_name
		skillID := c.Param("skill_id")
		if skillID == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "skill_id is required"})
			return
		}

		// Split node_id and skill_name
		parts := strings.Split(skillID, ".")
		if len(parts) != 2 {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": "skill_id must be in format 'node_id.skill_name'",
			})
			return
		}

		nodeID := parts[0]
		skillName := parts[1]

		// Parse request body
		var req ExecuteReasonerRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		// Find the agent node
		targetNode, err := storageProvider.GetAgent(ctx, nodeID)
		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{
				"error": fmt.Sprintf("node '%s' not found", nodeID),
			})
			return
		}

		// Check if skill exists on the node
		skillExists := false
		for _, skill := range targetNode.Skills {
			if skill.ID == skillName {
				skillExists = true
				break
			}
		}

		if !skillExists {
			c.JSON(http.StatusNotFound, gin.H{
				"error": fmt.Sprintf("skill '%s' not found on node '%s'", skillName, nodeID),
			})
			return
		}

		// Create workflow execution record
		workflowExecution := &types.WorkflowExecution{
			WorkflowID:          workflowID,
			ExecutionID:         executionID,
			AgentFieldRequestID: agentfieldRequestID,
			AgentNodeID:         nodeID,
			ReasonerID:          skillName, // For skills, ReasonerID will store skillName
			Status:              "running",
			StartedAt:           startTime,
			CreatedAt:           startTime,
			UpdatedAt:           startTime,
		}

		// Set optional fields
		if sessionID != "" {
			workflowExecution.SessionID = &sessionID
		}
		if actorID != "" {
			workflowExecution.ActorID = &actorID
		}
		if parentWorkflowID != "" {
			workflowExecution.ParentWorkflowID = &parentWorkflowID
		}
		if parentExecutionID != "" {
			workflowExecution.ParentExecutionID = &parentExecutionID
		}
		if rootWorkflowID != "" {
			workflowExecution.RootWorkflowID = &rootWorkflowID
		}
		if workflowName != "" {
			workflowExecution.WorkflowName = &workflowName
		}

		// Parse workflow tags
		if workflowTagsHeader != "" {
			tags := strings.Split(workflowTagsHeader, ",")
			for i, tag := range tags {
				tags[i] = strings.TrimSpace(tag)
			}
			workflowExecution.WorkflowTags = tags
		}

		// Store input data
		inputJSON, err := json.Marshal(req.Input)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to marshal input"})
			return
		}
		workflowExecution.InputData = inputJSON
		workflowExecution.InputSize = len(inputJSON)

		// Prepare request to agent node with workflow context propagation
		agentURL := fmt.Sprintf("%s/skills/%s", targetNode.BaseURL, skillName)
		agentReq, err := http.NewRequestWithContext(ctx, http.MethodPost, agentURL, bytes.NewBuffer(inputJSON))
		if err != nil {
			workflowExecution.Status = types.ExecutionStatusFailed
			errorMessage := fmt.Sprintf("failed to create agent request: %v", err)
			workflowExecution.ErrorMessage = &errorMessage
			endTime := time.Now()
			workflowExecution.CompletedAt = &endTime
			duration := endTime.Sub(startTime).Milliseconds()
			workflowExecution.DurationMS = &duration
			workflowExecution.UpdatedAt = endTime
			persistWorkflowExecution(ctx, storageProvider, workflowExecution)
			c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to create agent request"})
			return
		}

		agentReq.Header.Set("Content-Type", "application/json")
		agentReq.Header.Set("X-Workflow-ID", workflowID)
		agentReq.Header.Set("X-Execution-ID", executionID)
		agentReq.Header.Set("X-AgentField-Request-ID", agentfieldRequestID)
		if parentWorkflowID != "" {
			agentReq.Header.Set("X-Parent-Workflow-ID", parentWorkflowID)
		}
		if parentExecutionID != "" {
			agentReq.Header.Set("X-Parent-Execution-ID", parentExecutionID)
		}
		if rootWorkflowID != "" {
			agentReq.Header.Set("X-Root-Workflow-ID", rootWorkflowID)
		}
		if sessionID != "" {
			agentReq.Header.Set("X-Session-ID", sessionID)
		}
		if actorID != "" {
			agentReq.Header.Set("X-Actor-ID", actorID)
		}
		if workflowName != "" {
			agentReq.Header.Set("X-Workflow-Name", workflowName)
		}
		if workflowTagsHeader != "" {
			agentReq.Header.Set("X-Workflow-Tags", workflowTagsHeader)
		}
		if callerDID != "" {
			agentReq.Header.Set("X-Caller-DID", callerDID)
		}
		if targetDID != "" {
			agentReq.Header.Set("X-Target-DID", targetDID)
		}
		if agentNodeDID != "" {
			agentReq.Header.Set("X-Agent-Node-DID", agentNodeDID)
		}

		// Make HTTP request to agent node
		resp, err := http.DefaultClient.Do(agentReq)
		if err != nil {
			// Update execution with error
			endTime := time.Now()
			workflowExecution.Status = types.ExecutionStatusFailed
			errorMessage := err.Error()
			workflowExecution.ErrorMessage = &errorMessage
			workflowExecution.CompletedAt = &endTime
			duration := endTime.Sub(startTime).Milliseconds()
			workflowExecution.DurationMS = &duration
			workflowExecution.UpdatedAt = endTime

			// Store execution record
			persistWorkflowExecution(ctx, storageProvider, workflowExecution)

			c.JSON(http.StatusServiceUnavailable, gin.H{
				"error": fmt.Sprintf("failed to call agent node: %v", err),
			})
			return
		}
		defer resp.Body.Close()

		// Read response from agent node
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			// Update execution with error
			endTime := time.Now()
			workflowExecution.Status = types.ExecutionStatusFailed
			errorMsg := "failed to read agent response"
			workflowExecution.ErrorMessage = &errorMsg
			workflowExecution.CompletedAt = &endTime
			duration := endTime.Sub(startTime).Milliseconds()
			workflowExecution.DurationMS = &duration
			workflowExecution.UpdatedAt = endTime

			// Store execution record
			persistWorkflowExecution(ctx, storageProvider, workflowExecution)

			c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to read agent response"})
			return
		}

		// Parse agent response
		var result interface{}
		if err := json.Unmarshal(body, &result); err != nil {
			// Update execution with error
			endTime := time.Now()
			workflowExecution.Status = types.ExecutionStatusFailed
			errorMsg := "failed to parse agent response"
			workflowExecution.ErrorMessage = &errorMsg
			workflowExecution.CompletedAt = &endTime
			duration := endTime.Sub(startTime).Milliseconds()
			workflowExecution.DurationMS = &duration
			workflowExecution.UpdatedAt = endTime

			// Store execution record
			persistWorkflowExecution(ctx, storageProvider, workflowExecution)

			c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to parse agent response"})
			return
		}

		// Update execution with success
		endTime := time.Now()
		workflowExecution.Status = types.ExecutionStatusSucceeded
		workflowExecution.OutputData = body
		workflowExecution.OutputSize = len(body)
		workflowExecution.CompletedAt = &endTime
		duration := endTime.Sub(startTime).Milliseconds()
		workflowExecution.DurationMS = &duration
		workflowExecution.UpdatedAt = endTime

		// Store execution record
		persistWorkflowExecution(ctx, storageProvider, workflowExecution)

		// Set response headers
		c.Header("X-Workflow-ID", workflowID)
		c.Header("X-Execution-ID", executionID)
		c.Header("X-AgentField-Request-ID", agentfieldRequestID)
		c.Header("X-Agent-Node-ID", nodeID)
		c.Header("X-Duration-MS", fmt.Sprintf("%d", duration))

		// Return successful response
		c.JSON(http.StatusOK, ExecuteReasonerResponse{
			Result:    result,
			NodeID:    nodeID,
			Duration:  duration,
			Timestamp: endTime.Format(time.RFC3339),
		})
	}
}
