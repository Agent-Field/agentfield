package storage

import (
	"context"
	"database/sql"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/stretchr/testify/require"
)

func TestLocalStorageCleanupWorkflowDeletesExecutionRecordsWithoutFTS(t *testing.T) {
	ctx := context.Background()

	db, err := sql.Open("sqlite3", ":memory:")
	require.NoError(t, err)
	t.Cleanup(func() {
		_ = db.Close()
	})

	schema := []string{
		`CREATE TABLE executions (execution_id TEXT PRIMARY KEY, run_id TEXT NOT NULL)`,
		`CREATE TABLE execution_webhooks (execution_id TEXT PRIMARY KEY)`,
		`CREATE TABLE execution_webhook_events (id INTEGER PRIMARY KEY AUTOINCREMENT, execution_id TEXT NOT NULL)`,
		`CREATE TABLE workflow_runs (run_id TEXT PRIMARY KEY, root_workflow_id TEXT)`,
		`CREATE TABLE workflow_executions (execution_id TEXT PRIMARY KEY, workflow_id TEXT, root_workflow_id TEXT, run_id TEXT)`,
		`CREATE TABLE workflow_execution_events (event_id INTEGER PRIMARY KEY AUTOINCREMENT, workflow_id TEXT, run_id TEXT)`,
		`CREATE TABLE execution_vcs (vc_id TEXT PRIMARY KEY, workflow_id TEXT)`,
		`CREATE TABLE workflow_vcs (workflow_vc_id TEXT PRIMARY KEY, workflow_id TEXT)`,
		`CREATE TABLE workflows (workflow_id TEXT PRIMARY KEY)`,
	}
	for _, stmt := range schema {
		_, err := db.Exec(stmt)
		require.NoError(t, err)
	}

	const runID = "run_cleanup_exec_only"
	_, err = db.Exec(`INSERT INTO executions (execution_id, run_id) VALUES (?, ?)`, "exec_cleanup_1", runID)
	require.NoError(t, err)
	_, err = db.Exec(`INSERT INTO execution_webhooks (execution_id) VALUES (?)`, "exec_cleanup_1")
	require.NoError(t, err)
	_, err = db.Exec(`INSERT INTO execution_webhook_events (execution_id) VALUES (?)`, "exec_cleanup_1")
	require.NoError(t, err)

	ls := &LocalStorage{db: newSQLDatabase(db, "local")}

	result, err := ls.CleanupWorkflow(ctx, runID, false)
	require.NoError(t, err)
	require.True(t, result.Success)
	require.Greater(t, result.DeletedRecords["executions"], 0)
	require.Greater(t, result.DeletedRecords["execution_webhooks"], 0)
	require.Greater(t, result.DeletedRecords["execution_webhook_events"], 0)

	var count int
	require.NoError(t, db.QueryRow(`SELECT COUNT(*) FROM executions WHERE run_id = ?`, runID).Scan(&count))
	require.Equal(t, 0, count)
}

func TestLocalStorageCleanupWorkflowByRunID(t *testing.T) {
	ctx := context.Background()
	tempDir := t.TempDir()

	cfg := StorageConfig{
		Mode: "local",
		Local: LocalStorageConfig{
			DatabasePath: filepath.Join(tempDir, "agentfield.db"),
			KVStorePath:  filepath.Join(tempDir, "agentfield.bolt"),
		},
	}

	ls := NewLocalStorage(LocalStorageConfig{})
	if err := ls.Initialize(ctx, cfg); err != nil {
		if strings.Contains(err.Error(), "fts5") {
			t.Skip("sqlite3 compiled without FTS5; skipping cleanup test")
		}
		t.Fatalf("initialize local storage: %v", err)
	}
	t.Cleanup(func() {
		_ = ls.Close(ctx)
	})

	runID := "run_cleanup_test"
	workflowID := "wf_cleanup_test"
	now := time.Now().UTC()

	run := &types.WorkflowRun{
		RunID:          runID,
		RootWorkflowID: workflowID,
		Status:         string(types.ExecutionStatusRunning),
		TotalSteps:     1,
		CreatedAt:      now,
		UpdatedAt:      now,
	}
	if err := ls.StoreWorkflowRun(ctx, run); err != nil {
		t.Fatalf("store workflow run: %v", err)
	}

	workflow := &types.Workflow{
		WorkflowID:    workflowID,
		WorkflowName:  nil,
		WorkflowTags:  []string{},
		WorkflowDepth: 0,
		Status:        string(types.ExecutionStatusRunning),
		StartedAt:     now,
		CreatedAt:     now,
		UpdatedAt:     now,
	}
	if err := ls.CreateOrUpdateWorkflow(ctx, workflow); err != nil {
		t.Fatalf("store workflow definition: %v", err)
	}

	exec := &types.WorkflowExecution{
		WorkflowID:          workflowID,
		ExecutionID:         "exec_cleanup_test",
		AgentFieldRequestID: "req_cleanup_test",
		RunID:               &runID,
		AgentNodeID:         "agent_cleanup",
		ReasonerID:          "reasoner.cleanup",
		InputData:           json.RawMessage("{}"),
		OutputData:          json.RawMessage("{}"),
		InputSize:           0,
		OutputSize:          0,
		Status:              string(types.ExecutionStatusRunning),
		StartedAt:           now,
		CreatedAt:           now,
		UpdatedAt:           now,
		WorkflowDepth:       0,
		WorkflowTags:        []string{},
	}
	if err := ls.StoreWorkflowExecution(ctx, exec); err != nil {
		t.Fatalf("store workflow execution: %v", err)
	}

	executionRecord := &types.Execution{
		ExecutionID: "exec_record_cleanup_test",
		RunID:       runID,
		AgentNodeID: "agent_cleanup",
		ReasonerID:  "reasoner.cleanup",
		NodeID:      "node.cleanup",
		Status:      string(types.ExecutionStatusRunning),
		StartedAt:   now,
		CreatedAt:   now,
		UpdatedAt:   now,
	}
	if err := ls.CreateExecutionRecord(ctx, executionRecord); err != nil {
		t.Fatalf("store execution record: %v", err)
	}

	filterRunID := runID
	summariesBeforeCleanup, _, err := ls.QueryRunSummaries(ctx, types.ExecutionFilter{
		RunID: &filterRunID,
		Limit: 10,
	})
	if err != nil {
		t.Fatalf("query run summaries before cleanup: %v", err)
	}
	if len(summariesBeforeCleanup) == 0 {
		t.Fatalf("expected run summaries before cleanup")
	}

	step := &types.WorkflowStep{
		StepID:    "step_cleanup",
		RunID:     runID,
		Status:    string(types.ExecutionStatusPending),
		Attempt:   0,
		Priority:  0,
		NotBefore: now,
		Metadata:  json.RawMessage("{}"),
		CreatedAt: now,
		UpdatedAt: now,
	}
	if err := ls.StoreWorkflowStep(ctx, step); err != nil {
		t.Fatalf("store workflow step: %v", err)
	}

	event := &types.WorkflowRunEvent{
		RunID:            runID,
		Sequence:         1,
		PreviousSequence: 0,
		EventType:        "test",
		Payload:          json.RawMessage("{}"),
		EmittedAt:        now,
	}
	if err := ls.StoreWorkflowRunEvent(ctx, event); err != nil {
		t.Fatalf("store workflow run event: %v", err)
	}

	result, err := ls.CleanupWorkflow(ctx, runID, false)
	if err != nil {
		t.Fatalf("cleanup workflow by run id: %v", err)
	}

	if !result.Success {
		t.Fatalf("expected cleanup success, got %#v", result)
	}

	if result.WorkflowID != workflowID {
		t.Fatalf("expected workflow id %q, got %q", workflowID, result.WorkflowID)
	}

	if result.DeletedRecords["workflow_runs"] == 0 {
		t.Fatalf("expected workflow_runs to be deleted, got %#v", result.DeletedRecords)
	}
	if result.DeletedRecords["workflow_executions"] == 0 {
		t.Fatalf("expected workflow_executions to be deleted, got %#v", result.DeletedRecords)
	}
	if result.DeletedRecords["executions"] == 0 {
		t.Fatalf("expected executions to be deleted, got %#v", result.DeletedRecords)
	}

	// Run should be gone
	fetchedRun, err := ls.GetWorkflowRun(ctx, runID)
	if err != nil {
		t.Fatalf("get workflow run after cleanup: %v", err)
	}
	if fetchedRun != nil {
		t.Fatalf("expected workflow run to be deleted")
	}

	// Workflow definition should also be removed
	if _, err := ls.GetWorkflow(ctx, workflowID); err == nil {
		t.Fatalf("expected workflow definition to be deleted")
	}

	executionAfterCleanup, err := ls.GetExecutionRecord(ctx, executionRecord.ExecutionID)
	if err != nil {
		t.Fatalf("get execution record after cleanup: %v", err)
	}
	if executionAfterCleanup != nil {
		t.Fatalf("expected execution record to be deleted")
	}

	summariesAfterCleanup, _, err := ls.QueryRunSummaries(ctx, types.ExecutionFilter{
		RunID: &filterRunID,
		Limit: 10,
	})
	if err != nil {
		t.Fatalf("query run summaries after cleanup: %v", err)
	}
	if len(summariesAfterCleanup) != 0 {
		t.Fatalf("expected run summaries to be deleted, got %d", len(summariesAfterCleanup))
	}
}

func TestLocalStorageCleanupOldExecutions(t *testing.T) {
	ctx := context.Background()
	tempDir := t.TempDir()

	cfg := StorageConfig{
		Mode: "local",
		Local: LocalStorageConfig{
			DatabasePath: filepath.Join(tempDir, "agentfield.db"),
			KVStorePath:  filepath.Join(tempDir, "agentfield.bolt"),
		},
	}

	ls := NewLocalStorage(LocalStorageConfig{})
	if err := ls.Initialize(ctx, cfg); err != nil {
		if strings.Contains(err.Error(), "fts5") {
			t.Skip("sqlite3 compiled without FTS5; skipping old execution cleanup test")
		}
		t.Fatalf("initialize local storage: %v", err)
	}
	t.Cleanup(func() {
		_ = ls.Close(ctx)
	})

	const workflowID = "wf_cleanup_window"
	oldCompleted := time.Now().Add(-2 * time.Hour).UTC()
	recentCompleted := time.Now().Add(-15 * time.Minute).UTC()

	insertExecution := func(executionID string, completedAt time.Time) {
		exec := &types.WorkflowExecution{
			WorkflowID:          workflowID,
			ExecutionID:         executionID,
			AgentFieldRequestID: executionID + "_req",
			AgentNodeID:         "agent",
			ReasonerID:          "reasoner",
			Status:              "completed",
			StartedAt:           completedAt,
			CreatedAt:           completedAt,
			UpdatedAt:           completedAt,
			WorkflowDepth:       0,
			WorkflowTags:        []string{},
		}
		exec.CompletedAt = &completedAt
		require.NoError(t, ls.StoreWorkflowExecution(ctx, exec))
	}

	insertExecution("old-exec", oldCompleted)
	insertExecution("recent-exec", recentCompleted)

	deleted, err := ls.CleanupOldExecutions(ctx, time.Hour, 10)
	require.NoError(t, err)
	require.Equal(t, 1, deleted)

	stillThere, err := ls.GetWorkflowExecution(ctx, "recent-exec")
	require.NoError(t, err)
	require.NotNil(t, stillThere)

	removed, err := ls.GetWorkflowExecution(ctx, "old-exec")
	require.NoError(t, err)
	require.Nil(t, removed)
}

func TestCleanupOldExecutionsRetentionContract(t *testing.T) {
	ctx := context.Background()
	ls := NewLocalStorage(LocalStorageConfig{})
	cfg := StorageConfig{Mode: "local", Local: LocalStorageConfig{DatabasePath: filepath.Join(t.TempDir(), "retention.db"), KVStorePath: filepath.Join(t.TempDir(), "retention.bolt")}}
	require.NoError(t, ls.Initialize(ctx, cfg))
	t.Cleanup(func() { _ = ls.Close(ctx) })
	old := time.Now().UTC().Add(-96 * time.Hour)
	terminal := []string{"succeeded", "failed", "cancelled", "timeout", "completed", "revoked"}
	active := []string{"running", "pending", "queued", "waiting", "paused"}
	all := append(append([]string{}, terminal...), active...)
	for _, status := range all {
		id := "record-" + status
		input, result := "payload://input-"+status, "payload://result-"+status
		exec := &types.Execution{ExecutionID: id, RunID: "run-" + status, AgentNodeID: "agent", ReasonerID: "reasoner", NodeID: "node", Status: status, StartedAt: old, CompletedAt: &old, CreatedAt: old, UpdatedAt: old, InputURI: &input, ResultURI: &result}
		require.NoError(t, ls.CreateExecutionRecord(ctx, exec))
		wf := &types.WorkflowExecution{WorkflowID: "wf-" + status, ExecutionID: "workflow-" + status, AgentFieldRequestID: "request-" + status, AgentNodeID: "agent", ReasonerID: "reasoner", Status: status, StartedAt: old, CompletedAt: &old, CreatedAt: old, UpdatedAt: old, WorkflowTags: []string{}}
		require.NoError(t, ls.StoreWorkflowExecution(ctx, wf))
	}

	deleted, err := ls.CleanupOldExecutions(ctx, 0, 100)
	require.NoError(t, err)
	require.Zero(t, deleted)
	for _, table := range []string{"executions", "workflow_executions", "workflow_runs", "workflow_steps"} {
		var count int
		require.NoError(t, ls.db.QueryRowContext(ctx, "SELECT COUNT(*) FROM "+table).Scan(&count))
		if table == "executions" || table == "workflow_executions" {
			require.Equal(t, len(all), count)
		}
	}
	preserved, err := ls.CleanupOldExecutions(ctx, EffectiveExecutionRetention(72*time.Hour, 120*time.Hour), 100)
	require.NoError(t, err)
	require.Zero(t, preserved, "preserve-recent window must outlast the shorter retention period")

	uris, err := ls.ListExpiredExecutionPayloadURIs(ctx, 72*time.Hour, 100)
	require.NoError(t, err)
	require.Len(t, uris, len(terminal)*2)
	deleted, err = ls.CleanupOldExecutions(ctx, 72*time.Hour, 100)
	require.NoError(t, err)
	require.Equal(t, len(terminal)*2, deleted)
	for _, status := range terminal {
		var count int
		require.NoError(t, ls.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM executions WHERE execution_id = ?`, "record-"+status).Scan(&count))
		require.Zero(t, count)
		require.NoError(t, ls.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM workflow_executions WHERE execution_id = ?`, "workflow-"+status).Scan(&count))
		require.Zero(t, count)
	}
	for _, status := range active {
		var count int
		require.NoError(t, ls.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM executions WHERE execution_id = ?`, "record-"+status).Scan(&count))
		require.Equal(t, 1, count)
		require.NoError(t, ls.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM workflow_executions WHERE execution_id = ?`, "workflow-"+status).Scan(&count))
		require.Equal(t, 1, count)
	}
}

func TestPayloadURIListingsHandleEmptyNullAndErrors(t *testing.T) {
	ctx := context.Background()
	rawDB, err := sql.Open("sqlite3", filepath.Join(t.TempDir(), "payload-uris.db"))
	require.NoError(t, err)
	_, err = rawDB.Exec(`CREATE TABLE executions (
		execution_id TEXT PRIMARY KEY, run_id TEXT, agent_node_id TEXT, reasoner_id TEXT,
		node_id TEXT, status TEXT, started_at TIMESTAMP, completed_at TIMESTAMP,
		created_at TIMESTAMP, updated_at TIMESTAMP, input_uri TEXT, result_uri TEXT
	)`)
	require.NoError(t, err)
	ls := &LocalStorage{db: newSQLDatabase(rawDB, "local"), mode: "local"}
	old := time.Now().UTC().Add(-48 * time.Hour)
	rows := []struct {
		id     string
		input  interface{}
		result interface{}
	}{
		{"nulls", nil, nil},
		{"empty", "", ""},
		{"values", "payload://input", "payload://result"},
	}
	for _, row := range rows {
		_, err := ls.db.ExecContext(ctx, `INSERT INTO executions
			(execution_id, run_id, agent_node_id, reasoner_id, node_id, status, started_at, completed_at, created_at, updated_at, input_uri, result_uri)
			VALUES (?, ?, 'agent', 'reasoner', 'node', 'completed', ?, ?, ?, ?, ?, ?)`, row.id, "run-"+row.id, old, old, old, old, row.input, row.result)
		require.NoError(t, err)
	}

	uris, err := ls.ListExpiredExecutionPayloadURIs(ctx, 0, 10)
	require.NoError(t, err)
	require.Nil(t, uris)
	uris, err = ls.ListExpiredExecutionPayloadURIs(ctx, time.Hour, 10)
	require.NoError(t, err)
	require.ElementsMatch(t, []string{"payload://input", "payload://result"}, uris)
	refs, err := ls.ListPayloadURIs(ctx)
	require.NoError(t, err)
	require.Equal(t, map[string]struct{}{"payload://input": {}, "payload://result": {}}, refs)

	require.NoError(t, rawDB.Close())
	_, err = ls.ListExpiredExecutionPayloadURIs(ctx, time.Hour, 10)
	require.ErrorContains(t, err, "query expired execution payload URIs")
	_, err = ls.ListPayloadURIs(ctx)
	require.ErrorContains(t, err, "query payload URIs")
}

func TestCleanupOldExecutionsBatchSizeGuard(t *testing.T) {
	ls, ctx := setupLocalStorage(t)
	deleted, err := ls.CleanupOldExecutions(ctx, time.Hour, 0)
	require.NoError(t, err)
	require.Zero(t, deleted)
}
