package storage

import (
	"context"
	"database/sql"
	"fmt"

	"github.com/jmoiron/sqlx"
)

type sqlDatabase struct {
	*sql.DB
	mode string
}

func newSQLDatabase(db *sql.DB, mode string) *sqlDatabase {
	return &sqlDatabase{DB: db, mode: mode}
}

func (db *sqlDatabase) Mode() string {
	if db == nil {
		return ""
	}
	return db.mode
}

func (db *sqlDatabase) Begin() (*sqlTx, error) {
	if db == nil {
		return nil, fmt.Errorf("sql database is not initialized")
	}
	tx, err := db.DB.Begin()
	if err != nil {
		return nil, err
	}
	return newSQLTx(tx, db.mode), nil
}

func (db *sqlDatabase) BeginTx(ctx context.Context, opts *sql.TxOptions) (*sqlTx, error) {
	if db == nil {
		return nil, fmt.Errorf("sql database is not initialized")
	}
	tx, err := db.DB.BeginTx(ctx, opts)
	if err != nil {
		return nil, err
	}
	return newSQLTx(tx, db.mode), nil
}

func (db *sqlDatabase) rebind(query string) string {
	if db == nil {
		return query
	}
	if db.mode == "postgres" {
		return sqlx.Rebind(sqlx.DOLLAR, query)
	}
	return query
}

func (db *sqlDatabase) ExecContext(ctx context.Context, query string, args ...interface{}) (sql.Result, error) {
	return db.DB.ExecContext(ctx, db.rebind(query), args...)
}

func (db *sqlDatabase) Exec(query string, args ...interface{}) (sql.Result, error) {
	return db.DB.Exec(db.rebind(query), args...)
}

func (db *sqlDatabase) QueryContext(ctx context.Context, query string, args ...interface{}) (*sql.Rows, error) {
	return db.DB.QueryContext(ctx, db.rebind(query), args...)
}

func (db *sqlDatabase) Query(query string, args ...interface{}) (*sql.Rows, error) {
	return db.DB.Query(db.rebind(query), args...)
}

func (db *sqlDatabase) QueryRowContext(ctx context.Context, query string, args ...interface{}) *sql.Row {
	return db.DB.QueryRowContext(ctx, db.rebind(query), args...)
}

func (db *sqlDatabase) QueryRow(query string, args ...interface{}) *sql.Row {
	return db.DB.QueryRow(db.rebind(query), args...)
}

func (db *sqlDatabase) PrepareContext(ctx context.Context, query string) (*sql.Stmt, error) {
	return db.DB.PrepareContext(ctx, db.rebind(query))
}

type sqlTx struct {
	*sql.Tx
	mode string
}

func newSQLTx(tx *sql.Tx, mode string) *sqlTx {
	return &sqlTx{Tx: tx, mode: mode}
}

func (tx *sqlTx) rebind(query string) string {
	if tx == nil {
		return query
	}
	if tx.mode == "postgres" {
		return sqlx.Rebind(sqlx.DOLLAR, query)
	}
	return query
}

func (tx *sqlTx) ExecContext(ctx context.Context, query string, args ...interface{}) (sql.Result, error) {
	return tx.Tx.ExecContext(ctx, tx.rebind(query), args...)
}

func (tx *sqlTx) Exec(query string, args ...interface{}) (sql.Result, error) {
	return tx.Tx.Exec(tx.rebind(query), args...)
}

func (tx *sqlTx) QueryContext(ctx context.Context, query string, args ...interface{}) (*sql.Rows, error) {
	return tx.Tx.QueryContext(ctx, tx.rebind(query), args...)
}

func (tx *sqlTx) Query(query string, args ...interface{}) (*sql.Rows, error) {
	return tx.Tx.Query(tx.rebind(query), args...)
}

func (tx *sqlTx) QueryRowContext(ctx context.Context, query string, args ...interface{}) *sql.Row {
	return tx.Tx.QueryRowContext(ctx, tx.rebind(query), args...)
}

func (tx *sqlTx) QueryRow(query string, args ...interface{}) *sql.Row {
	return tx.Tx.QueryRow(tx.rebind(query), args...)
}
