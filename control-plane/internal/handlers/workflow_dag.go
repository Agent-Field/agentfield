package handlers

import (
	"context"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/your-org/haxen/control-plane/internal/storage"
	"github.com/your-org/haxen/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
)

type executionRecordProvider interface {
	QueryExecutionRecords(ctx context.Context, filter types.ExecutionFilter) ([]*types.Execution, error)
	GetExecutionRecord(ctx context.Context, executionID string) (*types.Execution, error)
}

type executionGraphService struct {
	store executionRecordProvider
}

func newExecutionGraphService(storageProvider storage.StorageProvider) *executionGraphService {
	return &executionGraphService{store: storageProvider}
}

type WorkflowDAGNode struct {
	WorkflowID        string                `json:"workflow_id"`
	ExecutionID       string                `json:"execution_id"`
	AgentNodeID       string                `json:"agent_node_id"`
	ReasonerID        string                `json:"reasoner_id"`
	Status            string                `json:"status"`
	StartedAt         string                `json:"started_at"`
	CompletedAt       *string               `json:"completed_at,omitempty"`
	DurationMS        *int64                `json:"duration_ms,omitempty"`
	ParentExecutionID *string               `json:"parent_execution_id,omitempty"`
	WorkflowDepth     int                   `json:"workflow_depth"`
	Children          []WorkflowDAGNode     `json:"children"`
	Notes             []types.ExecutionNote `json:"notes"`
	NotesCount        int                   `json:"notes_count"`
	LatestNote        *types.ExecutionNote  `json:"latest_note,omitempty"`
}

type WorkflowDAGResponse struct {
	RootWorkflowID string            `json:"root_workflow_id"`
	WorkflowStatus string            `json:"workflow_status"`
	WorkflowName   string            `json:"workflow_name"`
	SessionID      *string           `json:"session_id,omitempty"`
	ActorID        *string           `json:"actor_id,omitempty"`
	TotalNodes     int               `json:"total_nodes"`
	MaxDepth       int               `json:"max_depth"`
	DAG            WorkflowDAGNode   `json:"dag"`
	Timeline       []WorkflowDAGNode `json:"timeline"`
}

type SessionWorkflowsResponse struct {
	SessionID      string            `json:"session_id"`
	ActorID        *string           `json:"actor_id,omitempty"`
	TotalWorkflows int               `json:"total_workflows"`
	RootWorkflows  []WorkflowDAGNode `json:"root_workflows"`
	AllWorkflows   []WorkflowDAGNode `json:"all_workflows"`
}

func GetWorkflowDAGHandler(storageProvider storage.StorageProvider) gin.HandlerFunc {
	svc := newExecutionGraphService(storageProvider)
	return svc.handleGetWorkflowDAG
}

func (s *executionGraphService) handleGetWorkflowDAG(c *gin.Context) {
	ctx := c.Request.Context()
	runID := strings.TrimSpace(c.Param("workflowId"))
	if runID == "" {
		runID = strings.TrimSpace(c.Param("workflow_id"))
	}
	if runID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "workflowId or workflow_id is required"})
		return
	}

	executions, err := s.loadRunExecutions(ctx, runID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("failed to load workflow: %v", err)})
		return
	}
	if len(executions) == 0 {
		c.JSON(http.StatusNotFound, gin.H{"error": "workflow not found"})
		return
	}

	dag, timeline, workflowStatus, workflowName, sessionID, actorID, maxDepth := buildExecutionDAG(executions)

	response := WorkflowDAGResponse{
		RootWorkflowID: runID,
		WorkflowStatus: workflowStatus,
		WorkflowName:   workflowName,
		SessionID:      sessionID,
		ActorID:        actorID,
		TotalNodes:     len(executions),
		MaxDepth:       maxDepth,
		DAG:            dag,
		Timeline:       timeline,
	}

	c.JSON(http.StatusOK, response)
}

func GetWorkflowChildrenHandler(storageProvider storage.StorageProvider) gin.HandlerFunc {
	svc := newExecutionGraphService(storageProvider)
	return svc.handleGetWorkflowChildren
}

func (s *executionGraphService) handleGetWorkflowChildren(c *gin.Context) {
	ctx := c.Request.Context()
	parent := strings.TrimSpace(c.Param("workflow_id"))
	if parent == "" {
		parent = strings.TrimSpace(c.Param("execution_id"))
	}
	if parent == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "execution_id is required"})
		return
	}

	filter := types.ExecutionFilter{
		ParentExecutionID: &parent,
		SortBy:            "started_at",
		SortDescending:    false,
	}
	executions, err := s.store.QueryExecutionRecords(ctx, filter)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("failed to query executions: %v", err)})
		return
	}

	children := make([]WorkflowDAGNode, 0, len(executions))
	for _, exec := range executions {
		node := executionToDAGNode(exec, 0)
		node.Children = nil
		children = append(children, node)
	}

	c.JSON(http.StatusOK, gin.H{
		"execution_id": parent,
		"children":     children,
		"count":        len(children),
	})
}

func GetSessionWorkflowsHandler(storageProvider storage.StorageProvider) gin.HandlerFunc {
	svc := newExecutionGraphService(storageProvider)
	return svc.handleGetSessionWorkflows
}

func (s *executionGraphService) handleGetSessionWorkflows(c *gin.Context) {
	ctx := c.Request.Context()
	sessionID := strings.TrimSpace(c.Param("session_id"))
	if sessionID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "session_id is required"})
		return
	}

	filter := types.ExecutionFilter{
		SessionID:      &sessionID,
		SortBy:         "started_at",
		SortDescending: false,
	}
	executions, err := s.store.QueryExecutionRecords(ctx, filter)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("failed to query executions: %v", err)})
		return
	}

	if len(executions) == 0 {
		c.JSON(http.StatusOK, SessionWorkflowsResponse{
			SessionID:      sessionID,
			ActorID:        nil,
			TotalWorkflows: 0,
			RootWorkflows:  []WorkflowDAGNode{},
			AllWorkflows:   []WorkflowDAGNode{},
		})
		return
	}

	grouped := types.GroupExecutionsByRun(executions)
	rootNodes := make([]WorkflowDAGNode, 0, len(grouped))
	allNodes := make([]WorkflowDAGNode, 0, len(grouped))
	var actorID *string

	for runID, execs := range grouped {
		dag, _, _, _, sessionPtr, actorPtr, _ := buildExecutionDAG(execs)
		dag.WorkflowID = runID
		if actorPtr != nil && actorID == nil {
			actorID = actorPtr
		}
		if sessionPtr != nil && *sessionPtr != "" {
			sessionID = *sessionPtr
		}
		dag.WorkflowDepth = 0
		rootNodes = append(rootNodes, dag)
		allNodes = append(allNodes, dag)
	}

	response := SessionWorkflowsResponse{
		SessionID:      sessionID,
		ActorID:        actorID,
		TotalWorkflows: len(rootNodes),
		RootWorkflows:  rootNodes,
		AllWorkflows:   allNodes,
	}

	c.JSON(http.StatusOK, response)
}

func (s *executionGraphService) loadRunExecutions(ctx context.Context, runID string) ([]*types.Execution, error) {
	filter := types.ExecutionFilter{
		RunID:          &runID,
		SortBy:         "started_at",
		SortDescending: false,
	}
	return s.store.QueryExecutionRecords(ctx, filter)
}

func buildExecutionDAG(executions []*types.Execution) (WorkflowDAGNode, []WorkflowDAGNode, string, string, *string, *string, int) {
	execMap := make(map[string]*types.Execution, len(executions))
	childrenMap := make(map[string][]*types.Execution)
	var rootExec *types.Execution

	for _, exec := range executions {
		if exec == nil {
			continue
		}
		execMap[exec.ExecutionID] = exec
		if exec.ParentExecutionID != nil && *exec.ParentExecutionID != "" {
			parent := *exec.ParentExecutionID
			childrenMap[parent] = append(childrenMap[parent], exec)
		} else if rootExec == nil {
			rootExec = exec
		}
	}

	if rootExec == nil && len(executions) > 0 {
		rootExec = executions[0]
	}

	var maxDepth int
	var buildNode func(exec *types.Execution, depth int) WorkflowDAGNode

	buildNode = func(exec *types.Execution, depth int) WorkflowDAGNode {
		if exec == nil {
			return WorkflowDAGNode{}
		}

		node := executionToDAGNode(exec, depth)
		if depth > maxDepth {
			maxDepth = depth
		}

		children := childrenMap[exec.ExecutionID]
		if len(children) > 0 {
			node.Children = make([]WorkflowDAGNode, 0, len(children))
			for _, child := range children {
				node.Children = append(node.Children, buildNode(child, depth+1))
			}
		}

		return node
	}

	dag := buildNode(rootExec, 0)

	timeline := make([]WorkflowDAGNode, 0, len(executions))
	sort.Slice(executions, func(i, j int) bool {
		return executions[i].StartedAt.Before(executions[j].StartedAt)
	})
	for _, exec := range executions {
		node := executionToDAGNode(exec, 0)
		node.Children = nil
		timeline = append(timeline, node)
	}

	status := deriveOverallStatus(executions)
	workflowName := ""
	if rootExec != nil && rootExec.ReasonerID != "" {
		workflowName = rootExec.ReasonerID
	}

	var sessionID, actorID *string
	if rootExec != nil {
		sessionID = rootExec.SessionID
		actorID = rootExec.ActorID
	}

	return dag, timeline, status, workflowName, sessionID, actorID, maxDepth
}

// BuildWorkflowDAG exposes the DAG construction logic for other packages (UI handlers).
func BuildWorkflowDAG(executions []*types.Execution) (WorkflowDAGNode, []WorkflowDAGNode, string, string, *string, *string, int) {
	return buildExecutionDAG(executions)
}

func executionToDAGNode(exec *types.Execution, depth int) WorkflowDAGNode {
	started := exec.StartedAt.Format(time.RFC3339)
	var completed *string
	if exec.CompletedAt != nil {
		formatted := exec.CompletedAt.Format(time.RFC3339)
		completed = &formatted
	}

	return WorkflowDAGNode{
		WorkflowID:        exec.RunID,
		ExecutionID:       exec.ExecutionID,
		AgentNodeID:       exec.AgentNodeID,
		ReasonerID:        exec.ReasonerID,
		Status:            types.NormalizeExecutionStatus(exec.Status),
		StartedAt:         started,
		CompletedAt:       completed,
		DurationMS:        exec.DurationMS,
		ParentExecutionID: exec.ParentExecutionID,
		WorkflowDepth:     depth,
		Notes:             []types.ExecutionNote{},
		NotesCount:        0,
	}
}

func deriveOverallStatus(executions []*types.Execution) string {
	hasRunning := false
	for _, exec := range executions {
		status := types.NormalizeExecutionStatus(exec.Status)
		switch status {
		case string(types.ExecutionStatusFailed):
			return string(types.ExecutionStatusFailed)
		case string(types.ExecutionStatusRunning), string(types.ExecutionStatusPending), string(types.ExecutionStatusQueued):
			hasRunning = true
		}
	}
	if hasRunning {
		return string(types.ExecutionStatusRunning)
	}
	return string(types.ExecutionStatusSucceeded)
}
