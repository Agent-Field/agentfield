package storage

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

// Execution VC operations
func (ls *LocalStorage) StoreExecutionVC(ctx context.Context, vcID, executionID, workflowID, sessionID, issuerDID, targetDID, callerDID, inputHash, outputHash, status string, vcDocument []byte, signature string, storageURI string, documentSizeBytes int64) error {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during store execution VC: %w", err)
	}

	query := `
		INSERT INTO execution_vcs (
			vc_id, execution_id, workflow_id, session_id, issuer_did, target_did,
			caller_did, vc_document, signature, storage_uri, document_size_bytes,
			input_hash, output_hash, status, created_at
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(vc_id) DO UPDATE SET
			status = excluded.status,
			vc_document = excluded.vc_document,
			signature = excluded.signature,
			storage_uri = excluded.storage_uri,
			document_size_bytes = excluded.document_size_bytes;`

	_, err := ls.db.ExecContext(ctx, query, vcID, executionID, workflowID, sessionID, issuerDID, targetDID,
		callerDID, vcDocument, signature, storageURI, documentSizeBytes, inputHash, outputHash, status, time.Now())
	if err != nil {
		return fmt.Errorf("failed to store execution VC: %w", err)
	}
	return nil
}

// StoreExecutionVCRecord persists an ExecutionVC including the kind
// discriminator, parent_vc_id chain pointer, and trigger-event metadata.
// Used by all new VC writers — the older scalar StoreExecutionVC stays for
// backward compatibility with existing call sites.
func (ls *LocalStorage) StoreExecutionVCRecord(ctx context.Context, vc *types.ExecutionVC) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during store execution VC: %w", err)
	}
	if vc == nil {
		return fmt.Errorf("execution VC is nil")
	}
	kind := vc.Kind
	if kind == "" {
		kind = types.ExecutionVCKindExecution
	}
	created := vc.CreatedAt
	if created.IsZero() {
		created = time.Now()
	}

	query := `
		INSERT INTO execution_vcs (
			vc_id, execution_id, workflow_id, session_id, issuer_did, target_did,
			caller_did, vc_document, signature, storage_uri, document_size_bytes,
			input_hash, output_hash, status, parent_vc_id, kind,
			trigger_id, source_name, event_type, event_id, created_at
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(vc_id) DO UPDATE SET
			status = excluded.status,
			vc_document = excluded.vc_document,
			signature = excluded.signature,
			storage_uri = excluded.storage_uri,
			document_size_bytes = excluded.document_size_bytes,
			parent_vc_id = excluded.parent_vc_id,
			kind = excluded.kind,
			trigger_id = excluded.trigger_id,
			source_name = excluded.source_name,
			event_type = excluded.event_type,
			event_id = excluded.event_id;`

	_, err := ls.db.ExecContext(ctx, query,
		vc.VCID, vc.ExecutionID, vc.WorkflowID, vc.SessionID, vc.IssuerDID, vc.TargetDID,
		vc.CallerDID, []byte(vc.VCDocument), vc.Signature, vc.StorageURI, vc.DocumentSize,
		vc.InputHash, vc.OutputHash, vc.Status, vc.ParentVCID, kind,
		vc.TriggerID, vc.SourceName, vc.EventType, vc.EventID, created,
	)
	if err != nil {
		return fmt.Errorf("failed to store execution VC record: %w", err)
	}
	return nil
}

func (ls *LocalStorage) GetExecutionVC(ctx context.Context, vcID string) (*types.ExecutionVCInfo, error) {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during get execution VC: %w", err)
	}

	query := `
		SELECT vc_id, execution_id, workflow_id, session_id, issuer_did, target_did,
			   caller_did, input_hash, output_hash, status, created_at, storage_uri, document_size_bytes,
			   parent_vc_id, kind, trigger_id, source_name, event_type, event_id
		FROM execution_vcs WHERE vc_id = ?`

	row := ls.db.QueryRowContext(ctx, query, vcID)
	info := &types.ExecutionVCInfo{}

	err := row.Scan(&info.VCID, &info.ExecutionID, &info.WorkflowID, &info.SessionID,
		&info.IssuerDID, &info.TargetDID, &info.CallerDID, &info.InputHash,
		&info.OutputHash, &info.Status, &info.CreatedAt, &info.StorageURI, &info.DocumentSize,
		&info.ParentVCID, &info.Kind, &info.TriggerID, &info.SourceName, &info.EventType, &info.EventID)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("execution VC %s not found", vcID)
		}
		return nil, fmt.Errorf("failed to get execution VC: %w", err)
	}
	return info, nil
}

func buildExecutionVCFilterClauses(filters types.VCFilters) (string, []interface{}) {
	var (
		conditions []string
		args       []interface{}
	)

	if filters.ExecutionID != nil {
		conditions = append(conditions, "evc.execution_id = ?")
		args = append(args, *filters.ExecutionID)
	}
	if filters.WorkflowID != nil {
		conditions = append(conditions, "evc.workflow_id = ?")
		args = append(args, *filters.WorkflowID)
	}
	if filters.SessionID != nil {
		conditions = append(conditions, "evc.session_id = ?")
		args = append(args, *filters.SessionID)
	}
	if filters.IssuerDID != nil {
		conditions = append(conditions, "evc.issuer_did = ?")
		args = append(args, *filters.IssuerDID)
	}
	if filters.TargetDID != nil {
		conditions = append(conditions, "evc.target_did = ?")
		args = append(args, *filters.TargetDID)
	}
	if filters.CallerDID != nil {
		conditions = append(conditions, "evc.caller_did = ?")
		args = append(args, *filters.CallerDID)
	}
	if filters.AgentNodeID != nil {
		conditions = append(conditions, "COALESCE(we.agent_node_id, '') = ?")
		args = append(args, *filters.AgentNodeID)
	}
	if filters.Status != nil {
		conditions = append(conditions, "evc.status = ?")
		args = append(args, *filters.Status)
	}
	if filters.CreatedAfter != nil {
		conditions = append(conditions, "evc.created_at >= ?")
		args = append(args, filters.CreatedAfter.UTC())
	}
	if filters.CreatedBefore != nil {
		conditions = append(conditions, "evc.created_at <= ?")
		args = append(args, filters.CreatedBefore.UTC())
	}

	if filters.Search != nil {
		if trimmed := strings.TrimSpace(*filters.Search); trimmed != "" {
			search := "%" + strings.ToLower(trimmed) + "%"
			conditions = append(conditions, "("+
				"LOWER(evc.execution_id) LIKE ? OR "+
				"LOWER(evc.workflow_id) LIKE ? OR "+
				"LOWER(evc.issuer_did) LIKE ? OR "+
				"LOWER(evc.target_did) LIKE ? OR "+
				"LOWER(evc.caller_did) LIKE ? OR "+
				"LOWER(evc.session_id) LIKE ? OR "+
				"LOWER(COALESCE(we.agent_node_id, '')) LIKE ? OR "+
				"LOWER(COALESCE(we.workflow_name, '')) LIKE ?"+
				")")
			for i := 0; i < 8; i++ {
				args = append(args, search)
			}
		}
	}

	return strings.Join(conditions, " AND "), args
}

func (ls *LocalStorage) ListExecutionVCs(ctx context.Context, filters types.VCFilters) ([]*types.ExecutionVCInfo, error) {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during list execution VCs: %w", err)
	}

	query := `
		SELECT evc.vc_id, evc.execution_id, evc.workflow_id, evc.session_id,
		       evc.issuer_did, evc.target_did, evc.caller_did, evc.input_hash,
		       evc.output_hash, evc.status, evc.created_at, evc.storage_uri,
		       evc.document_size_bytes, we.agent_node_id, we.workflow_name,
		       evc.parent_vc_id, evc.kind, evc.trigger_id, evc.source_name, evc.event_type, evc.event_id
		FROM execution_vcs evc
		LEFT JOIN workflow_executions we ON we.execution_id = evc.execution_id`

	whereClause, args := buildExecutionVCFilterClauses(filters)
	if whereClause != "" {
		query += " WHERE " + whereClause
	}

	query += " ORDER BY evc.created_at DESC"

	if filters.Limit > 0 {
		query += fmt.Sprintf(" LIMIT %d", filters.Limit)
	}

	if filters.Offset > 0 {
		query += fmt.Sprintf(" OFFSET %d", filters.Offset)
	}

	rows, err := ls.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to list execution VCs: %w", err)
	}
	defer rows.Close()

	var infos []*types.ExecutionVCInfo
	for rows.Next() {
		// Check context cancellation during iteration
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("context cancelled during execution VC list iteration: %w", err)
		}

		info := &types.ExecutionVCInfo{}
		err := rows.Scan(&info.VCID, &info.ExecutionID, &info.WorkflowID, &info.SessionID,
			&info.IssuerDID, &info.TargetDID, &info.CallerDID, &info.InputHash,
			&info.OutputHash, &info.Status, &info.CreatedAt, &info.StorageURI,
			&info.DocumentSize, &info.AgentNodeID, &info.WorkflowName,
			&info.ParentVCID, &info.Kind, &info.TriggerID, &info.SourceName, &info.EventType, &info.EventID)
		if err != nil {
			return nil, fmt.Errorf("failed to scan execution VC: %w", err)
		}
		infos = append(infos, info)
	}
	return infos, nil
}

func (ls *LocalStorage) ListWorkflowVCStatusSummaries(ctx context.Context, workflowIDs []string) ([]*types.WorkflowVCStatusAggregation, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during workflow VC status summary query: %w", err)
	}

	if len(workflowIDs) == 0 {
		return []*types.WorkflowVCStatusAggregation{}, nil
	}

	placeholders := make([]string, len(workflowIDs))
	for i := range workflowIDs {
		placeholders[i] = "?"
	}

	query := fmt.Sprintf(`
		SELECT workflow_id,
		       COUNT(*) AS vc_count,
		       SUM(CASE WHEN status = ? THEN 1 ELSE 0 END) AS verified_count,
		       SUM(CASE WHEN status = ? OR status = ? THEN 1 ELSE 0 END) AS failed_count,
		       MAX(created_at) AS last_created_at
		FROM execution_vcs
		WHERE workflow_id IN (%s)
		GROUP BY workflow_id
	`, strings.Join(placeholders, ","))

	args := []interface{}{
		string(types.ExecutionStatusSucceeded),
		string(types.ExecutionStatusFailed),
		string(types.ExecutionStatusTimeout),
	}
	for _, id := range workflowIDs {
		args = append(args, id)
	}

	rows, err := ls.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to query workflow VC status summaries: %w", err)
	}
	defer rows.Close()

	var summaries []*types.WorkflowVCStatusAggregation
	for rows.Next() {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("context cancelled during workflow VC status iteration: %w", err)
		}

		var lastCreated sql.NullTime
		summary := &types.WorkflowVCStatusAggregation{}
		if err := rows.Scan(
			&summary.WorkflowID,
			&summary.VCCount,
			&summary.VerifiedCount,
			&summary.FailedCount,
			&lastCreated,
		); err != nil {
			return nil, fmt.Errorf("failed to scan workflow VC status summary: %w", err)
		}

		if lastCreated.Valid {
			summary.LastCreatedAt = &lastCreated.Time
		}

		summaries = append(summaries, summary)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("workflow VC status summary rows error: %w", err)
	}

	return summaries, nil
}

func (ls *LocalStorage) CountExecutionVCs(ctx context.Context, filters types.VCFilters) (int, error) {
	if err := ctx.Err(); err != nil {
		return 0, fmt.Errorf("context cancelled during count execution VCs: %w", err)
	}

	query := `
		SELECT COUNT(*)
		FROM execution_vcs evc
		LEFT JOIN workflow_executions we ON we.execution_id = evc.execution_id`

	whereClause, args := buildExecutionVCFilterClauses(filters)
	if whereClause != "" {
		query += " WHERE " + whereClause
	}

	var total int
	if err := ls.db.QueryRowContext(ctx, query, args...).Scan(&total); err != nil {
		return 0, fmt.Errorf("failed to count execution VCs: %w", err)
	}
	return total, nil
}

// Workflow VC operations
func (ls *LocalStorage) StoreWorkflowVC(ctx context.Context, workflowVCID, workflowID, sessionID string, componentVCIDs []string, status string, startTime, endTime *time.Time, totalSteps, completedSteps int, storageURI string, documentSizeBytes int64) error {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during store workflow VC: %w", err)
	}

	componentVCIDsJSON, err := json.Marshal(componentVCIDs)
	if err != nil {
		return fmt.Errorf("failed to marshal component VC IDs: %w", err)
	}

	query := `
		INSERT INTO workflow_vcs (
			workflow_vc_id, workflow_id, session_id, component_vc_ids, status,
			start_time, end_time, total_steps, completed_steps, storage_uri, document_size_bytes
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(workflow_vc_id) DO UPDATE SET
			component_vc_ids = excluded.component_vc_ids,
			status = excluded.status,
			end_time = excluded.end_time,
			completed_steps = excluded.completed_steps,
			storage_uri = excluded.storage_uri,
			document_size_bytes = excluded.document_size_bytes;`

	_, err = ls.db.ExecContext(ctx, query, workflowVCID, workflowID, sessionID, componentVCIDsJSON, status,
		startTime, endTime, totalSteps, completedSteps, storageURI, documentSizeBytes)
	if err != nil {
		return fmt.Errorf("failed to store workflow VC: %w", err)
	}
	return nil
}

func (ls *LocalStorage) GetWorkflowVC(ctx context.Context, workflowVCID string) (*types.WorkflowVCInfo, error) {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during get workflow VC: %w", err)
	}

	query := `
		SELECT workflow_vc_id, workflow_id, session_id, component_vc_ids, status,
			   start_time, end_time, total_steps, completed_steps, storage_uri, document_size_bytes
		FROM workflow_vcs WHERE workflow_vc_id = ?`

	row := ls.db.QueryRowContext(ctx, query, workflowVCID)
	info := &types.WorkflowVCInfo{}
	var componentVCIDsJSON []byte

	err := row.Scan(&info.WorkflowVCID, &info.WorkflowID, &info.SessionID, &componentVCIDsJSON,
		&info.Status, &info.StartTime, &info.EndTime, &info.TotalSteps, &info.CompletedSteps, &info.StorageURI, &info.DocumentSize)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("workflow VC %s not found", workflowVCID)
		}
		return nil, fmt.Errorf("failed to get workflow VC: %w", err)
	}

	if len(componentVCIDsJSON) > 0 {
		if err := json.Unmarshal(componentVCIDsJSON, &info.ComponentVCIDs); err != nil {
			return nil, fmt.Errorf("failed to unmarshal component VC IDs: %w", err)
		}
	}

	return info, nil
}

func (ls *LocalStorage) ListWorkflowVCs(ctx context.Context, workflowID string) ([]*types.WorkflowVCInfo, error) {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during list workflow VCs: %w", err)
	}

	var query string
	var args []interface{}

	if workflowID == "" {
		// Get all workflow VCs
		query = `
			SELECT workflow_vc_id, workflow_id, session_id, component_vc_ids, status,
				   start_time, end_time, total_steps, completed_steps, storage_uri, document_size_bytes
			FROM workflow_vcs ORDER BY start_time DESC`
	} else {
		// Get workflow VCs for specific workflow
		query = `
			SELECT workflow_vc_id, workflow_id, session_id, component_vc_ids, status,
				   start_time, end_time, total_steps, completed_steps, storage_uri, document_size_bytes
			FROM workflow_vcs WHERE workflow_id = ? ORDER BY start_time DESC`
		args = append(args, workflowID)
	}

	rows, err := ls.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to list workflow VCs: %w", err)
	}
	defer rows.Close()

	var infos []*types.WorkflowVCInfo
	for rows.Next() {
		// Check context cancellation during iteration
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("context cancelled during workflow VC list iteration: %w", err)
		}

		info := &types.WorkflowVCInfo{}
		var componentVCIDsJSON []byte

		err := rows.Scan(&info.WorkflowVCID, &info.WorkflowID, &info.SessionID, &componentVCIDsJSON,
			&info.Status, &info.StartTime, &info.EndTime, &info.TotalSteps, &info.CompletedSteps, &info.StorageURI, &info.DocumentSize)
		if err != nil {
			return nil, fmt.Errorf("failed to scan workflow VC: %w", err)
		}

		if len(componentVCIDsJSON) > 0 {
			if err := json.Unmarshal(componentVCIDsJSON, &info.ComponentVCIDs); err != nil {
				return nil, fmt.Errorf("failed to unmarshal component VC IDs: %w", err)
			}
		}

		infos = append(infos, info)
	}
	return infos, nil
}

// GetFullExecutionVC retrieves the full execution VC including the VC document and signature
func (ls *LocalStorage) GetFullExecutionVC(vcID string) (json.RawMessage, string, error) {
	query := `
		SELECT vc_document, signature
		FROM execution_vcs WHERE vc_id = ?`

	row := ls.db.QueryRow(query, vcID)

	var vcDocument string
	var signature string

	err := row.Scan(&vcDocument, &signature)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, "", fmt.Errorf("execution VC %s not found", vcID)
		}
		return nil, "", fmt.Errorf("failed to get full execution VC: %w", err)
	}

	return json.RawMessage(vcDocument), signature, nil
}
