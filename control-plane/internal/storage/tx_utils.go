package storage

import (
	"database/sql"
	"errors"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
)

type rollbacker interface {
	Rollback() error
}

// rollbackTx attempts to rollback the transaction and logs a warning when the rollback fails.
func rollbackTx(tx rollbacker, context string) {
	if tx == nil {
		return
	}

	if err := tx.Rollback(); err != nil && !errors.Is(err, sql.ErrTxDone) {
		logger.Logger.Warn().
			Err(err).
			Str("context", context).
			Msg("transaction rollback failed")
	}
}
