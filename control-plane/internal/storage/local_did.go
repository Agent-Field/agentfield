package storage

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"hash/fnv"
	"strconv"
	"strings"
	"time"

	"github.com/jackc/pgx/v5/pgconn"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

// ErrDIDAllocationConflict means a derived DID already belongs to another
// identity. Callers may re-read issued identities and allocate a new path.
var ErrDIDAllocationConflict = errors.New("DID allocation conflict")

// ErrDIDNodeAlreadyRegistered means another active identity owns the node or component name.
var ErrDIDNodeAlreadyRegistered = errors.New("DID node or component already registered")

// ErrDIDDocumentNotFound means no DID document exists for the requested DID.
// It lets callers distinguish a genuine missing record from a transient
// storage failure, so a momentary DB error is not reported as "not found".
var ErrDIDDocumentNotFound = errors.New("DID document not found")

const didPostgresLockTimeout = "15s"

func isSQLiteContention(err error) bool {
	message := err.Error()
	return strings.Contains(message, "database is locked") ||
		strings.Contains(message, "database table is locked") ||
		strings.Contains(message, "database schema is locked") ||
		strings.Contains(message, "SQLITE_BUSY") ||
		strings.Contains(message, "SQLITE_LOCKED")
}

// IsTransientDIDTransactionError identifies contention that requires a new transaction.
func IsTransientDIDTransactionError(err error) bool {
	var pgErr *pgconn.PgError
	if errors.As(err, &pgErr) {
		return pgErr.Code == "40001" || pgErr.Code == "40P01" || pgErr.Code == "55P03"
	}
	return err != nil && isSQLiteContention(err)
}

func (ls *LocalStorage) StoreAgentFieldServerDID(ctx context.Context, agentfieldServerID, rootDID string, masterSeed []byte, createdAt, lastKeyRotation time.Time) error {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during store af server DID: %w", err)
	}

	// Validate input parameters
	if agentfieldServerID == "" {
		return &ValidationError{
			Field:   "agentfield_server_id",
			Value:   agentfieldServerID,
			Reason:  "af server ID cannot be empty",
			Context: "StoreAgentFieldServerDID",
		}
	}
	if rootDID == "" {
		return &ValidationError{
			Field:   "root_did",
			Value:   rootDID,
			Reason:  "root DID cannot be empty",
			Context: "StoreAgentFieldServerDID",
		}
	}
	if len(masterSeed) == 0 {
		return &ValidationError{
			Field:   "master_seed",
			Value:   "<encrypted>",
			Reason:  "master seed cannot be empty",
			Context: "StoreAgentFieldServerDID",
		}
	}

	// Use transaction for data consistency
	tx, err := ls.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer func() {
		if err != nil {
			rollbackTx(tx, "StoreAgentFieldServerDID")
		}
	}()

	// Execute with retry logic
	err = ls.retryOnConstraintFailure(ctx, func() error {
		// Never replace the key material of an existing registry with a
		// different one. A row with the same root DID holds the same identity, so
		// only its seed encoding (for example after enabling encryption at rest)
		// and rotation time are refreshed. Callers read the row back to learn
		// which identity won when two control planes create it at once.
		query := `
                        INSERT INTO did_registry (agentfield_server_id, root_did, master_seed_encrypted, created_at, last_key_rotation, total_dids)
                        VALUES (?, ?, ?, ?, ?, 0)
                        ON CONFLICT (agentfield_server_id) DO UPDATE SET
                                master_seed_encrypted = excluded.master_seed_encrypted,
                                last_key_rotation = excluded.last_key_rotation
                        WHERE did_registry.root_did = excluded.root_did
                `
		_, execErr := tx.ExecContext(ctx, query, agentfieldServerID, rootDID, masterSeed, createdAt, lastKeyRotation)
		if execErr != nil {
			return fmt.Errorf("failed to store af server DID: %w", execErr)
		}
		return nil
	}, 3) // Retry up to 3 times for transient errors

	if err != nil {
		return err
	}

	// Commit transaction
	if err = tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	logger.Logger.Debug().Msgf("Successfully stored af server DID: agentfield_server_id=%s, root_did=%s", agentfieldServerID, rootDID)
	return nil
}

// StoreAgentDIDWithComponents stores an agent DID along with its component DIDs in a single transaction
func (ls *LocalStorage) StoreAgentDIDWithComponents(ctx context.Context, agentID, agentDID, agentfieldServerDID, publicKeyJWK string, derivationIndex int, components []ComponentDIDRequest) error {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during store agent DID with components: %w", err)
	}

	// Pre-storage validation
	if err := ls.validateAgentFieldServerExists(ctx, agentfieldServerDID); err != nil {
		return fmt.Errorf("pre-storage validation failed: %w", err)
	}

	// Use transaction for data consistency across all operations
	tx, err := ls.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer func() {
		if err != nil {
			rollbackTx(tx, "StoreAgentDIDWithComponents")
		}
	}()

	// Store agent DID first
	err = ls.retryOnConstraintFailure(ctx, func() error {
		query := `
			INSERT INTO agent_dids (
				agent_node_id, did, agentfield_server_id, public_key_jwk, derivation_path, registered_at, status
			) VALUES (?, ?, ?, ?, ?, ?, ?)`

		derivationPath := fmt.Sprintf("m/44'/0'/0'/%d", derivationIndex)
		_, execErr := tx.ExecContext(ctx, query, agentID, agentDID, agentfieldServerDID, publicKeyJWK, derivationPath, time.Now(), "active")
		if execErr != nil {
			if strings.Contains(execErr.Error(), "UNIQUE constraint failed") || strings.Contains(execErr.Error(), "agent_dids") {
				return &DuplicateDIDError{
					DID:  fmt.Sprintf("agent:%s@%s", agentID, agentfieldServerDID),
					Type: "agent",
				}
			}
			if strings.Contains(execErr.Error(), "FOREIGN KEY constraint failed") {
				return &ForeignKeyConstraintError{
					Table:           "agent_dids",
					Column:          "agentfield_server_id",
					ReferencedTable: "did_registry",
					ReferencedValue: agentfieldServerDID,
					Operation:       "INSERT",
				}
			}
			return fmt.Errorf("failed to store agent DID: %w", execErr)
		}
		return nil
	}, 3)

	if err != nil {
		var dupErr *DuplicateDIDError
		if errors.As(err, &dupErr) {
			return dupErr
		}
		return fmt.Errorf("failed to store agent DID: %w", err)
	}

	// Store component DIDs
	for i, component := range components {
		err = ls.retryOnConstraintFailure(ctx, func() error {
			query := `
				INSERT INTO component_dids (
					did, agent_did, component_type, function_name, public_key_jwk, derivation_path
				) VALUES (?, ?, ?, ?, ?, ?)`

			derivationPath := fmt.Sprintf("m/44'/0'/0'/%d", component.DerivationIndex)
			_, execErr := tx.ExecContext(ctx, query, component.ComponentDID, agentDID, component.ComponentType, component.ComponentName, component.PublicKeyJWK, derivationPath)
			if execErr != nil {
				if strings.Contains(execErr.Error(), "UNIQUE constraint failed") || strings.Contains(execErr.Error(), "component_dids") {
					return &DuplicateDIDError{
						DID:  fmt.Sprintf("component:%s/%s@%s", component.ComponentType, component.ComponentName, agentDID),
						Type: "component",
					}
				}
				if strings.Contains(execErr.Error(), "FOREIGN KEY constraint failed") {
					return &ForeignKeyConstraintError{
						Table:           "component_dids",
						Column:          "agent_did",
						ReferencedTable: "agent_dids",
						ReferencedValue: agentDID,
						Operation:       "INSERT",
					}
				}
				return fmt.Errorf("failed to store component DID %d: %w", i, execErr)
			}
			return nil
		}, 3)

		if err != nil {
			var dupErr *DuplicateDIDError
			if errors.As(err, &dupErr) {
				return dupErr
			}
			return fmt.Errorf("failed to store component DID %d (%s): %w", i, component.ComponentName, err)
		}
	}

	// Commit transaction
	if err = tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	logger.Logger.Debug().Msgf("Successfully stored agent DID with %d components: agent_id=%s, did=%s", len(components), agentID, agentDID)
	return nil
}

// UpsertAgentDIDWithComponents persists an agent DID and its components in one
// transaction. Derivation paths and public keys are stored as given so the
// identity can be re-derived after a restart. Status and X25519 epoch cannot
// move backwards. A row is never moved to a different owner or component name.
func (ls *LocalStorage) UpsertAgentDIDWithComponents(ctx context.Context, record AgentDIDRecord) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during upsert agent DID: %w", err)
	}

	required := map[string]string{
		"agent_node_id":        record.AgentNodeID,
		"did":                  record.DID,
		"agentfield_server_id": record.AgentFieldServerID,
		"public_key_jwk":       record.PublicKeyJWK,
		"derivation_path":      record.DerivationPath,
	}
	for field, value := range required {
		if strings.TrimSpace(value) == "" {
			return &ValidationError{Field: field, Value: value, Reason: field + " cannot be empty", Context: "UpsertAgentDIDWithComponents"}
		}
	}
	for _, component := range record.Components {
		if component.ComponentDID == "" || component.ComponentName == "" || component.PublicKeyJWK == "" || component.DerivationPath == "" {
			return &ValidationError{
				Field:   "component",
				Value:   component.ComponentName,
				Reason:  "component DID, name, public key and derivation path are required",
				Context: "UpsertAgentDIDWithComponents",
			}
		}
		if component.ComponentType != "reasoner" && component.ComponentType != "skill" {
			return &ValidationError{Field: "component_type", Value: component.ComponentType, Reason: "component type must be 'reasoner' or 'skill'", Context: "UpsertAgentDIDWithComponents"}
		}
	}
	status := record.Status
	if status == "" {
		status = string(types.AgentDIDStatusActive)
	}

	if err := ls.validateAgentFieldServerExists(ctx, record.AgentFieldServerID); err != nil {
		return fmt.Errorf("pre-storage validation failed: %w", err)
	}

	tx, err := ls.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	// Every early return, including a typed conflict, releases the transaction.
	defer rollbackTx(tx, "UpsertAgentDIDWithComponents")
	if err := ls.lockDIDServerTransaction(ctx, tx, record.AgentFieldServerID); err != nil {
		return err
	}
	var storedDID string
	err = tx.QueryRowContext(ctx, "SELECT did FROM agent_dids WHERE did = ?", record.DID).Scan(&storedDID)
	if err != nil && !errors.Is(err, sql.ErrNoRows) {
		return fmt.Errorf("failed to check agent DID: %w", err)
	}
	if errors.Is(err, sql.ErrNoRows) {
		err = tx.QueryRowContext(ctx, `SELECT did FROM agent_dids WHERE agentfield_server_id = ? AND agent_node_id = ? AND status <> 'revoked' LIMIT 1`, record.AgentFieldServerID, record.AgentNodeID).Scan(&storedDID)
		if err == nil {
			return fmt.Errorf("agent node %s already has active DID %s: %w", record.AgentNodeID, storedDID, ErrDIDNodeAlreadyRegistered)
		}
		if !errors.Is(err, sql.ErrNoRows) {
			return fmt.Errorf("failed to check active agent DID: %w", err)
		}
	}
	err = nil

	err = func() error {
		result, execErr := tx.ExecContext(ctx, `
			INSERT INTO agent_dids (
				agent_node_id, did, agentfield_server_id, public_key_jwk, derivation_path, x25519_epoch, registered_at, status
			) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
			ON CONFLICT (did) DO UPDATE SET
				public_key_jwk = excluded.public_key_jwk,
				derivation_path = excluded.derivation_path,
				x25519_epoch = CASE WHEN agent_dids.x25519_epoch > excluded.x25519_epoch THEN agent_dids.x25519_epoch ELSE excluded.x25519_epoch END,
				status = CASE WHEN agent_dids.status = 'revoked' THEN 'revoked' ELSE excluded.status END,
				updated_at = CURRENT_TIMESTAMP
			WHERE agent_dids.agentfield_server_id = excluded.agentfield_server_id
			  AND agent_dids.agent_node_id = excluded.agent_node_id`,
			record.AgentNodeID, record.DID, record.AgentFieldServerID, record.PublicKeyJWK,
			record.DerivationPath, record.X25519Epoch, time.Now(), status)
		if execErr != nil {
			return fmt.Errorf("failed to upsert agent DID: %w", execErr)
		}
		if affected, affErr := result.RowsAffected(); affErr == nil && affected == 0 {
			return fmt.Errorf("agent DID %s is already stored for a different server or agent node: %w", record.DID, ErrDIDAllocationConflict)
		}
		return nil
	}()
	if err != nil {
		return err
	}

	for _, component := range record.Components {
		component := component
		err = tx.QueryRowContext(ctx, "SELECT did FROM component_dids WHERE did = ?", component.ComponentDID).Scan(&storedDID)
		if err != nil && !errors.Is(err, sql.ErrNoRows) {
			return fmt.Errorf("failed to check component DID: %w", err)
		}
		if errors.Is(err, sql.ErrNoRows) {
			err = tx.QueryRowContext(ctx, `SELECT did FROM component_dids WHERE agent_did = ? AND component_type = ? AND function_name = ? AND COALESCE(status, 'active') = 'active' LIMIT 1`, record.DID, component.ComponentType, component.ComponentName).Scan(&storedDID)
			if err == nil {
				return fmt.Errorf("component %s already has active DID %s: %w", component.ComponentName, storedDID, ErrDIDNodeAlreadyRegistered)
			}
			if !errors.Is(err, sql.ErrNoRows) {
				return fmt.Errorf("failed to check active component DID: %w", err)
			}
		}
		err = nil
		err = func() error {
			result, execErr := tx.ExecContext(ctx, `
				INSERT INTO component_dids (
					did, agent_did, component_type, function_name, public_key_jwk, derivation_path, status
				) VALUES (?, ?, ?, ?, ?, ?, ?)
				ON CONFLICT (did) DO UPDATE SET
					public_key_jwk = excluded.public_key_jwk,
					derivation_path = excluded.derivation_path,
					status = CASE WHEN component_dids.status = 'retired' THEN 'retired' ELSE excluded.status END,
					updated_at = CURRENT_TIMESTAMP
				WHERE component_dids.agent_did = excluded.agent_did
				  AND component_dids.component_type = excluded.component_type
				  AND component_dids.function_name = excluded.function_name`,
				component.ComponentDID, record.DID, component.ComponentType, component.ComponentName,
				component.PublicKeyJWK, component.DerivationPath, ComponentDIDStatusActive)
			if execErr != nil {
				return fmt.Errorf("failed to upsert component DID %s: %w", component.ComponentName, execErr)
			}
			if affected, affErr := result.RowsAffected(); affErr == nil && affected == 0 {
				return fmt.Errorf("component DID %s is already stored for a different agent or component: %w", component.ComponentDID, ErrDIDAllocationConflict)
			}
			return nil
		}()
		if err != nil {
			return err
		}
	}

	if err = tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}
	return nil
}

func (ls *LocalStorage) GetAgentFieldServerDID(ctx context.Context, agentfieldServerID string) (*types.AgentFieldServerDIDInfo, error) {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during get af server DID: %w", err)
	}

	query := `
		SELECT agentfield_server_id, root_did, master_seed_encrypted, created_at, last_key_rotation
		FROM did_registry WHERE agentfield_server_id = ?
	`
	row := ls.db.QueryRowContext(ctx, query, agentfieldServerID)
	info := &types.AgentFieldServerDIDInfo{}

	err := row.Scan(&info.AgentFieldServerID, &info.RootDID, &info.MasterSeed, &info.CreatedAt, &info.LastKeyRotation)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, nil // Return nil, nil for "not found"
		}
		return nil, fmt.Errorf("failed to get af server DID: %w", err)
	}
	return info, nil
}

func (ls *LocalStorage) ListAgentFieldServerDIDs(ctx context.Context) ([]*types.AgentFieldServerDIDInfo, error) {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during list af server DIDs: %w", err)
	}

	query := `
		SELECT agentfield_server_id, root_did, master_seed_encrypted, created_at, last_key_rotation
		FROM did_registry ORDER BY created_at DESC
	`
	rows, err := ls.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to list af server DIDs: %w", err)
	}
	defer rows.Close()

	var infos []*types.AgentFieldServerDIDInfo
	for rows.Next() {
		// Check context cancellation during iteration
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("context cancelled during af server DID list iteration: %w", err)
		}

		info := &types.AgentFieldServerDIDInfo{}
		err := rows.Scan(&info.AgentFieldServerID, &info.RootDID, &info.MasterSeed, &info.CreatedAt, &info.LastKeyRotation)
		if err != nil {
			return nil, fmt.Errorf("failed to scan af server DID: %w", err)
		}
		infos = append(infos, info)
	}
	return infos, nil
}

// DID Registry operations
func (ls *LocalStorage) StoreDID(ctx context.Context, did string, didDocument, publicKey, privateKeyRef, derivationPath string) error {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during store DID: %w", err)
	}

	// INSERT-only query - no ON CONFLICT clause for security
	query := `
		INSERT INTO did_registry (
			did, did_document, public_key, private_key_ref, derivation_path,
			created_at, updated_at, status
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`

	now := time.Now()
	_, err := ls.db.ExecContext(ctx, query, did, didDocument, publicKey, privateKeyRef, derivationPath, now, now, "active")
	if err != nil {
		// Check if this is a unique constraint violation (duplicate DID)
		if strings.Contains(err.Error(), "UNIQUE constraint failed") || strings.Contains(err.Error(), "did_registry.did") {
			logger.Logger.Warn().Msgf("Duplicate DID registry entry detected: %s", did)
			return &DuplicateDIDError{
				DID:  did,
				Type: "registry",
			}
		}
		return fmt.Errorf("failed to store DID: %w", err)
	}

	logger.Logger.Debug().Msgf("Successfully stored DID registry entry: %s", did)
	return nil
}

func (ls *LocalStorage) GetDID(ctx context.Context, did string) (*types.DIDRegistryEntry, error) {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during get DID: %w", err)
	}

	query := `
		SELECT did, did_document, public_key, private_key_ref, derivation_path,
			   created_at, updated_at, status
		FROM did_registry WHERE did = ?`

	row := ls.db.QueryRowContext(ctx, query, did)
	entry := &types.DIDRegistryEntry{}

	err := row.Scan(&entry.DID, &entry.DIDDocument, &entry.PublicKey, &entry.PrivateKeyRef,
		&entry.DerivationPath, &entry.CreatedAt, &entry.UpdatedAt, &entry.Status)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("DID %s not found", did)
		}
		return nil, fmt.Errorf("failed to get DID: %w", err)
	}
	return entry, nil
}

func (ls *LocalStorage) ListDIDs(ctx context.Context) ([]*types.DIDRegistryEntry, error) {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during list DIDs: %w", err)
	}

	query := `
		SELECT did, did_document, public_key, private_key_ref, derivation_path,
			   created_at, updated_at, status
		FROM did_registry ORDER BY created_at DESC`

	rows, err := ls.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to list DIDs: %w", err)
	}
	defer rows.Close()

	var entries []*types.DIDRegistryEntry
	for rows.Next() {
		// Check context cancellation during iteration
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("context cancelled during DID list iteration: %w", err)
		}

		entry := &types.DIDRegistryEntry{}
		err := rows.Scan(&entry.DID, &entry.DIDDocument, &entry.PublicKey, &entry.PrivateKeyRef,
			&entry.DerivationPath, &entry.CreatedAt, &entry.UpdatedAt, &entry.Status)
		if err != nil {
			return nil, fmt.Errorf("failed to scan DID entry: %w", err)
		}
		entries = append(entries, entry)
	}
	return entries, nil
}

// validateAgentFieldServerExists checks if a af server registry exists
func (ls *LocalStorage) validateAgentFieldServerExists(ctx context.Context, agentfieldServerID string) error {
	if agentfieldServerID == "" {
		return &ValidationError{
			Field:   "agentfield_server_id",
			Value:   agentfieldServerID,
			Reason:  "af server ID cannot be empty",
			Context: "pre-storage validation",
		}
	}

	query := `SELECT 1 FROM did_registry WHERE agentfield_server_id = ? LIMIT 1`
	var exists int
	err := ls.db.QueryRowContext(ctx, query, agentfieldServerID).Scan(&exists)
	if err != nil {
		if err == sql.ErrNoRows {
			return &ForeignKeyConstraintError{
				Table:           "agent_dids",
				Column:          "agentfield_server_id",
				ReferencedTable: "did_registry",
				ReferencedValue: agentfieldServerID,
				Operation:       "INSERT",
			}
		}
		return fmt.Errorf("failed to validate af server existence: %w", err)
	}
	return nil
}

// validateAgentDIDExists checks if an agent DID exists
func (ls *LocalStorage) validateAgentDIDExists(ctx context.Context, agentDID string) error {
	if agentDID == "" {
		return &ValidationError{
			Field:   "agent_did",
			Value:   agentDID,
			Reason:  "agent DID cannot be empty",
			Context: "pre-storage validation",
		}
	}

	query := `SELECT 1 FROM agent_dids WHERE did = ? LIMIT 1`
	var exists int
	err := ls.db.QueryRowContext(ctx, query, agentDID).Scan(&exists)
	if err != nil {
		if err == sql.ErrNoRows {
			return &ForeignKeyConstraintError{
				Table:           "component_dids",
				Column:          "agent_did",
				ReferencedTable: "agent_dids",
				ReferencedValue: agentDID,
				Operation:       "INSERT",
			}
		}
		return fmt.Errorf("failed to validate agent DID existence: %w", err)
	}
	return nil
}

// retryOnConstraintFailure executes a function with retry logic for transient constraint issues
func (ls *LocalStorage) retryOnConstraintFailure(ctx context.Context, operation func() error, maxRetries int) error {
	var lastErr error
	for attempt := 0; attempt <= maxRetries; attempt++ {
		if err := ctx.Err(); err != nil {
			return fmt.Errorf("context cancelled during retry attempt %d: %w", attempt, err)
		}

		lastErr = operation()
		if lastErr == nil {
			return nil
		}

		// Don't retry on validation errors or permanent constraint violations
		if _, isValidationErr := lastErr.(*ValidationError); isValidationErr {
			return lastErr
		}
		if _, isFKErr := lastErr.(*ForeignKeyConstraintError); isFKErr {
			return lastErr
		}
		if _, isDuplicateErr := lastErr.(*DuplicateDIDError); isDuplicateErr {
			return lastErr
		}
		if errors.Is(lastErr, ErrDIDAllocationConflict) {
			return lastErr
		}

		// Only retry on database-level transient errors
		if isSQLiteContention(lastErr) ||
			strings.Contains(lastErr.Error(), "database is temporarily unavailable") {
			if attempt < maxRetries {
				// Exponential backoff: 10ms, 20ms, 40ms
				backoff := time.Duration(10*(1<<attempt)) * time.Millisecond
				time.Sleep(backoff)
				continue
			}
		}

		// For other errors, don't retry
		return lastErr
	}
	return lastErr
}

// Agent DID operations
func (ls *LocalStorage) StoreAgentDID(ctx context.Context, agentID, agentDID, agentfieldServerDID, publicKeyJWK string, derivationIndex int) error {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during store agent DID: %w", err)
	}

	// Pre-storage validation
	if err := ls.validateAgentFieldServerExists(ctx, agentfieldServerDID); err != nil {
		return fmt.Errorf("pre-storage validation failed: %w", err)
	}

	// Validate input parameters
	if agentID == "" {
		return &ValidationError{
			Field:   "agent_node_id",
			Value:   agentID,
			Reason:  "agent ID cannot be empty",
			Context: "StoreAgentDID",
		}
	}
	if agentDID == "" {
		return &ValidationError{
			Field:   "did",
			Value:   agentDID,
			Reason:  "agent DID cannot be empty",
			Context: "StoreAgentDID",
		}
	}
	if publicKeyJWK == "" {
		return &ValidationError{
			Field:   "public_key_jwk",
			Value:   publicKeyJWK,
			Reason:  "public key JWK cannot be empty",
			Context: "StoreAgentDID",
		}
	}

	// Use transaction for data consistency
	tx, err := ls.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer func() {
		if err != nil {
			rollbackTx(tx, "StoreAgentDID")
		}
	}()

	// Execute with retry logic
	err = ls.retryOnConstraintFailure(ctx, func() error {
		// INSERT-only query - no ON CONFLICT clause for security
		query := `
			INSERT INTO agent_dids (
				agent_node_id, did, agentfield_server_id, public_key_jwk, derivation_path, registered_at, status
			) VALUES (?, ?, ?, ?, ?, ?, ?)`

		derivationPath := fmt.Sprintf("m/44'/0'/0'/%d", derivationIndex)
		_, execErr := tx.ExecContext(ctx, query, agentID, agentDID, agentfieldServerDID, publicKeyJWK, derivationPath, time.Now(), "active")
		if execErr != nil {
			// Check if this is a unique constraint violation (duplicate agent DID)
			if strings.Contains(execErr.Error(), "UNIQUE constraint failed") || strings.Contains(execErr.Error(), "agent_dids") {
				logger.Logger.Warn().Msgf("Duplicate agent DID entry detected: agent_id=%s, agentfield_server_id=%s", agentID, agentfieldServerDID)
				return &DuplicateDIDError{
					DID:  fmt.Sprintf("agent:%s@%s", agentID, agentfieldServerDID),
					Type: "agent",
				}
			}
			// Check for foreign key constraint violations
			if strings.Contains(execErr.Error(), "FOREIGN KEY constraint failed") {
				return &ForeignKeyConstraintError{
					Table:           "agent_dids",
					Column:          "agentfield_server_id",
					ReferencedTable: "did_registry",
					ReferencedValue: agentfieldServerDID,
					Operation:       "INSERT",
				}
			}
			return fmt.Errorf("failed to store agent DID: %w", execErr)
		}
		return nil
	}, 3) // Retry up to 3 times for transient errors

	if err != nil {
		return err
	}

	// Commit transaction
	if err = tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	logger.Logger.Debug().Msgf("Successfully stored agent DID entry: agent_id=%s, did=%s", agentID, agentDID)
	return nil
}

func (ls *LocalStorage) GetAgentDID(ctx context.Context, agentID string) (*types.AgentDIDInfo, error) {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during get agent DID: %w", err)
	}

	query := `
		SELECT agent_node_id, did, agentfield_server_id, public_key_jwk, derivation_path,
		       COALESCE(x25519_epoch, 0), reasoners, skills, status, registered_at
		FROM agent_dids WHERE agent_node_id = ?`

	row := ls.db.QueryRowContext(ctx, query, agentID)
	info := &types.AgentDIDInfo{}

	var reasonersJSON, skillsJSON, publicKeyJWK string
	err := row.Scan(&info.AgentNodeID, &info.DID, &info.AgentFieldServerID, &publicKeyJWK,
		&info.DerivationPath, &info.X25519Epoch, &reasonersJSON, &skillsJSON, &info.Status, &info.RegisteredAt)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("agent DID for %s not found", agentID)
		}
		return nil, fmt.Errorf("failed to get agent DID: %w", err)
	}
	info.PublicKeyJWK = json.RawMessage(publicKeyJWK)

	// Parse JSON fields
	if reasonersJSON != "" {
		if err := json.Unmarshal([]byte(reasonersJSON), &info.Reasoners); err != nil {
			return nil, fmt.Errorf("failed to parse reasoners JSON: %w", err)
		}
	} else {
		info.Reasoners = make(map[string]types.ReasonerDIDInfo)
	}

	if skillsJSON != "" {
		if err := json.Unmarshal([]byte(skillsJSON), &info.Skills); err != nil {
			return nil, fmt.Errorf("failed to parse skills JSON: %w", err)
		}
	} else {
		info.Skills = make(map[string]types.SkillDIDInfo)
	}

	return info, nil
}

// GetActiveAgentDID reads one node without scanning or changing other identities.
func (ls *LocalStorage) GetActiveAgentDID(ctx context.Context, serverID, nodeID string) (*types.AgentDIDInfo, error) {
	row := ls.db.QueryRowContext(ctx, `SELECT agent_node_id, did, agentfield_server_id, public_key_jwk,
		derivation_path, COALESCE(x25519_epoch, 0), status, registered_at
		FROM agent_dids WHERE agentfield_server_id = ? AND agent_node_id = ? AND status <> 'revoked'
		ORDER BY registered_at ASC, did ASC LIMIT 1`, serverID, nodeID)
	info := &types.AgentDIDInfo{}
	var publicKey string
	err := row.Scan(&info.AgentNodeID, &info.DID, &info.AgentFieldServerID, &publicKey,
		&info.DerivationPath, &info.X25519Epoch, &info.Status, &info.RegisteredAt)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get active agent DID: %w", err)
	}
	info.PublicKeyJWK = json.RawMessage(publicKey)
	return info, nil
}

func (ls *LocalStorage) ListAgentDIDs(ctx context.Context) ([]*types.AgentDIDInfo, error) {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during list agent DIDs: %w", err)
	}

	query := `
		SELECT agent_node_id, did, agentfield_server_id, public_key_jwk, derivation_path,
		       COALESCE(x25519_epoch, 0), reasoners, skills, status, registered_at
		FROM agent_dids ORDER BY registered_at DESC`

	rows, err := ls.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to list agent DIDs: %w", err)
	}
	defer rows.Close()

	var infos []*types.AgentDIDInfo
	for rows.Next() {
		// Check context cancellation during iteration
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("context cancelled during agent DID list iteration: %w", err)
		}

		info := &types.AgentDIDInfo{}
		var reasonersJSON, skillsJSON, publicKeyJWK string
		err := rows.Scan(&info.AgentNodeID, &info.DID, &info.AgentFieldServerID, &publicKeyJWK,
			&info.DerivationPath, &info.X25519Epoch, &reasonersJSON, &skillsJSON, &info.Status, &info.RegisteredAt)
		if err != nil {
			return nil, fmt.Errorf("failed to scan agent DID: %w", err)
		}
		info.PublicKeyJWK = json.RawMessage(publicKeyJWK)

		// Parse JSON fields
		if reasonersJSON != "" {
			if err := json.Unmarshal([]byte(reasonersJSON), &info.Reasoners); err != nil {
				return nil, fmt.Errorf("failed to parse reasoners JSON: %w", err)
			}
		} else {
			info.Reasoners = make(map[string]types.ReasonerDIDInfo)
		}

		if skillsJSON != "" {
			if err := json.Unmarshal([]byte(skillsJSON), &info.Skills); err != nil {
				return nil, fmt.Errorf("failed to parse skills JSON: %w", err)
			}
		} else {
			info.Skills = make(map[string]types.SkillDIDInfo)
		}

		infos = append(infos, info)
	}
	return infos, nil
}

// Component DID operations
func (ls *LocalStorage) StoreComponentDID(ctx context.Context, componentID, componentDID, agentDID, componentType, componentName string, derivationIndex int) error {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during store component DID: %w", err)
	}

	// Pre-storage validation
	if err := ls.validateAgentDIDExists(ctx, agentDID); err != nil {
		return fmt.Errorf("pre-storage validation failed: %w", err)
	}

	// Validate input parameters
	if componentDID == "" {
		return &ValidationError{
			Field:   "component_did",
			Value:   componentDID,
			Reason:  "component DID cannot be empty",
			Context: "StoreComponentDID",
		}
	}
	if componentType == "" {
		return &ValidationError{
			Field:   "component_type",
			Value:   componentType,
			Reason:  "component type cannot be empty",
			Context: "StoreComponentDID",
		}
	}
	if componentName == "" {
		return &ValidationError{
			Field:   "component_name",
			Value:   componentName,
			Reason:  "component name cannot be empty",
			Context: "StoreComponentDID",
		}
	}
	// Validate component type
	validTypes := map[string]bool{"reasoner": true, "skill": true}
	if !validTypes[componentType] {
		return &ValidationError{
			Field:   "component_type",
			Value:   componentType,
			Reason:  "component type must be 'reasoner' or 'skill'",
			Context: "StoreComponentDID",
		}
	}

	// Use transaction for data consistency
	tx, err := ls.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer func() {
		if err != nil {
			rollbackTx(tx, "StoreComponentDID")
		}
	}()

	// Execute with retry logic
	err = ls.retryOnConstraintFailure(ctx, func() error {
		// INSERT-only query - no ON CONFLICT clause for security
		query := `
			INSERT INTO component_dids (
				did, agent_did, component_type, function_name, public_key_jwk, derivation_path
			) VALUES (?, ?, ?, ?, ?, ?)`

		derivationPath := fmt.Sprintf("m/44'/0'/0'/%d", derivationIndex)
		// For now, use empty public key - this should be passed as a parameter in the future
		publicKeyJWK := ""
		_, execErr := tx.ExecContext(ctx, query, componentDID, agentDID, componentType, componentName, publicKeyJWK, derivationPath)
		if execErr != nil {
			// Check if this is a unique constraint violation (duplicate component DID)
			if strings.Contains(execErr.Error(), "UNIQUE constraint failed") || strings.Contains(execErr.Error(), "component_dids") {
				logger.Logger.Warn().Msgf("Duplicate component DID entry detected: agent_did=%s, function_name=%s, component_type=%s", agentDID, componentName, componentType)
				return &DuplicateDIDError{
					DID:  fmt.Sprintf("component:%s/%s@%s", componentType, componentName, agentDID),
					Type: "component",
				}
			}
			// Check for foreign key constraint violations
			if strings.Contains(execErr.Error(), "FOREIGN KEY constraint failed") {
				return &ForeignKeyConstraintError{
					Table:           "component_dids",
					Column:          "agent_did",
					ReferencedTable: "agent_dids",
					ReferencedValue: agentDID,
					Operation:       "INSERT",
				}
			}
			return fmt.Errorf("failed to store component DID: %w", execErr)
		}
		return nil
	}, 3) // Retry up to 3 times for transient errors

	if err != nil {
		return err
	}

	// Commit transaction
	if err = tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	logger.Logger.Debug().Msgf("Successfully stored component DID entry: component_did=%s, agent_did=%s, type=%s", componentDID, agentDID, componentType)
	return nil
}

func (ls *LocalStorage) GetComponentDID(ctx context.Context, componentID string) (*types.ComponentDIDInfo, error) {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during get component DID: %w", err)
	}

	// Use function_name as the componentID since there's no separate component_id column
	// A name can have retired rows from earlier registrations; prefer the active one.
	query := `
		SELECT function_name, did, agent_did, component_type, function_name,
			   derivation_path, public_key_jwk, COALESCE(status, 'active'), created_at
		FROM component_dids WHERE function_name = ?
		ORDER BY CASE WHEN COALESCE(status, 'active') = 'active' THEN 0 ELSE 1 END, created_at DESC
		LIMIT 1`

	row := ls.db.QueryRowContext(ctx, query, componentID)
	info := &types.ComponentDIDInfo{}

	var derivationPath string
	var publicKeyJWK sql.NullString
	var createdAt sql.NullTime

	err := row.Scan(&info.ComponentID, &info.ComponentDID, &info.AgentDID,
		&info.ComponentType, &info.ComponentName, &derivationPath, &publicKeyJWK, &info.Status, &createdAt)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("component DID for %s not found", componentID)
		}
		return nil, fmt.Errorf("failed to get component DID: %w", err)
	}

	if createdAt.Valid {
		info.CreatedAt = createdAt.Time
	}
	info.DerivationPath = derivationPath
	info.PublicKeyJWK = publicKeyJWK.String

	// Parse derivation index from derivation path (e.g., "m/44'/0'/0'/123" -> 123)
	if derivationPath != "" {
		parts := strings.Split(derivationPath, "/")
		if len(parts) > 0 {
			lastPart := parts[len(parts)-1]
			if derivationIndex, parseErr := strconv.Atoi(strings.Trim(lastPart, "'")); parseErr == nil {
				info.DerivationIndex = derivationIndex
			}
		}
	}

	return info, nil
}

// ListComponentDIDs lists the active component DIDs of an agent, or of every
// agent when agentDID is empty. Retired components are excluded so that a
// removed component is never loaded back into the registry.
func (ls *LocalStorage) ListComponentDIDs(ctx context.Context, agentDID string) ([]*types.ComponentDIDInfo, error) {
	return ls.listComponentDIDs(ctx, agentDID, false)
}

// ListAllComponentDIDs lists an agent's component DIDs including retired ones,
// or of every agent when agentDID is empty. Key allocation uses it so that a
// retired component's derivation path is never handed out again.
func (ls *LocalStorage) ListAllComponentDIDs(ctx context.Context, agentDID string) ([]*types.ComponentDIDInfo, error) {
	return ls.listComponentDIDs(ctx, agentDID, true)
}

func (ls *LocalStorage) listComponentDIDs(ctx context.Context, agentDID string, includeRetired bool) ([]*types.ComponentDIDInfo, error) {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during list component DIDs: %w", err)
	}

	query := `
		SELECT function_name, did, agent_did, component_type, function_name,
			   derivation_path, public_key_jwk, COALESCE(status, 'active'), created_at
		FROM component_dids`
	var conditions []string
	var args []interface{}
	if agentDID != "" {
		conditions = append(conditions, "agent_did = ?")
		args = append(args, agentDID)
	}
	if !includeRetired {
		conditions = append(conditions, "COALESCE(status, 'active') = ?")
		args = append(args, ComponentDIDStatusActive)
	}
	if len(conditions) > 0 {
		query += " WHERE " + strings.Join(conditions, " AND ")
	}
	query += " ORDER BY created_at DESC"

	rows, err := ls.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to list component DIDs: %w", err)
	}
	defer rows.Close()

	var infos []*types.ComponentDIDInfo
	for rows.Next() {
		// Check context cancellation during iteration
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("context cancelled during component DID list iteration: %w", err)
		}

		info := &types.ComponentDIDInfo{}
		var derivationPath string
		var publicKeyJWK sql.NullString
		var createdAt sql.NullTime

		err := rows.Scan(&info.ComponentID, &info.ComponentDID, &info.AgentDID,
			&info.ComponentType, &info.ComponentName, &derivationPath, &publicKeyJWK, &info.Status, &createdAt)
		if err != nil {
			return nil, fmt.Errorf("failed to scan component DID: %w", err)
		}

		if createdAt.Valid {
			info.CreatedAt = createdAt.Time
		}
		info.DerivationPath = derivationPath
		info.PublicKeyJWK = publicKeyJWK.String

		// Parse derivation index from derivation path
		if derivationPath != "" {
			parts := strings.Split(derivationPath, "/")
			if len(parts) > 0 {
				lastPart := parts[len(parts)-1]
				if derivationIndex, parseErr := strconv.Atoi(strings.Trim(lastPart, "'")); parseErr == nil {
					info.DerivationIndex = derivationIndex
				}
			}
		}

		infos = append(infos, info)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("failed to iterate component DIDs: %w", err)
	}
	return infos, nil
}

// RetireComponentDIDs retires only the given component DIDs of this agent.
// Retired rows stay in storage so their derivation paths are never reused, but
// they are no longer listed as registered components. An empty list is a no-op.
func (ls *LocalStorage) RetireComponentDIDs(ctx context.Context, agentDID string, componentDIDs []string) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during retire component DIDs: %w", err)
	}
	if len(componentDIDs) == 0 {
		return nil
	}
	if strings.TrimSpace(agentDID) == "" {
		return &ValidationError{Field: "agent_did", Value: agentDID, Reason: "agent DID cannot be empty", Context: "RetireComponentDIDs"}
	}

	placeholders := make([]string, len(componentDIDs))
	args := make([]interface{}, 0, len(componentDIDs)+2)
	args = append(args, ComponentDIDStatusRetired, agentDID)
	for i, did := range componentDIDs {
		placeholders[i] = "?"
		args = append(args, did)
	}
	query := `UPDATE component_dids SET status = ?, updated_at = CURRENT_TIMESTAMP
  WHERE agent_did = ? AND did IN (` + strings.Join(placeholders, ", ") + `)`
	tx, err := ls.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin component retirement: %w", err)
	}
	defer rollbackTx(tx, "RetireComponentDIDs")
	if _, err := tx.ExecContext(ctx, query, args...); err != nil {
		return fmt.Errorf("failed to retire component DIDs for agent %s: %w", agentDID, err)
	}
	return tx.Commit()
}

// lockDIDServerTransaction shares identity-write serialization with rotations.
// SQLite connections use BEGIN IMMEDIATE; Postgres uses a server advisory lock.
func (ls *LocalStorage) lockDIDServerTransaction(ctx context.Context, tx *sqlTx, serverID string) error {
	if ls.mode == "postgres" {
		// Bound the wait before serializing identity decisions across replicas.
		if _, err := tx.ExecContext(ctx, "SET LOCAL lock_timeout = '"+didPostgresLockTimeout+"'"); err != nil {
			return fmt.Errorf("failed to set DID transaction lock timeout: %w", err)
		}
		hash := fnv.New64a()
		_, _ = hash.Write([]byte("agentfield:did-server:" + serverID))
		if _, err := tx.ExecContext(ctx, "SELECT pg_advisory_xact_lock(?)", int64(hash.Sum64())); err != nil {
			return fmt.Errorf("failed to lock DID server transaction: %w", err)
		}
	}
	return nil
}

// AdvanceAgentX25519Epoch allocates the next epoch from durable state. The
// returned epoch is committed before callers derive or publish its keypair.
func (ls *LocalStorage) AdvanceAgentX25519Epoch(ctx context.Context, serverID, did string, memoryEpoch int) (int, error) {
	tx, err := ls.db.BeginTx(ctx, nil)
	if err != nil {
		return 0, fmt.Errorf("failed to begin X25519 rotation: %w", err)
	}
	defer rollbackTx(tx, "AdvanceAgentX25519Epoch")
	if err := ls.lockDIDServerTransaction(ctx, tx, serverID); err != nil {
		return 0, err
	}
	var epoch int
	if err := tx.QueryRowContext(ctx, "SELECT x25519_epoch FROM agent_dids WHERE did = ? AND agentfield_server_id = ?", did, serverID).Scan(&epoch); err != nil {
		return 0, fmt.Errorf("failed to read stored X25519 epoch: %w", err)
	}
	if memoryEpoch > epoch {
		epoch = memoryEpoch
	}
	if epoch == int(^uint(0)>>1) {
		return 0, fmt.Errorf("X25519 epoch exhausted for DID %s", did)
	}
	epoch++
	if _, err := tx.ExecContext(ctx, "UPDATE agent_dids SET x25519_epoch = ?, updated_at = CURRENT_TIMESTAMP WHERE did = ? AND agentfield_server_id = ?", epoch, did, serverID); err != nil {
		return 0, fmt.Errorf("failed to persist X25519 epoch: %w", err)
	}
	if _, err := tx.ExecContext(ctx, "UPDATE did_registry SET last_key_rotation = ? WHERE agentfield_server_id = ?", time.Now(), serverID); err != nil {
		return 0, fmt.Errorf("failed to persist X25519 rotation time: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return 0, fmt.Errorf("failed to commit X25519 rotation: %w", err)
	}
	return epoch, nil
}
