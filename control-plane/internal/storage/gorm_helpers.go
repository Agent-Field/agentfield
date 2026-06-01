package storage

import (
	"context"
	"fmt"

	"gorm.io/driver/postgres"
	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
	"gorm.io/gorm/schema"
)

// initGormDB initializes the shared GORM handle using the existing SQLite connection.
func (ls *LocalStorage) initGormDB() error {
	if ls.db == nil {
		return fmt.Errorf("sql connection is not initialized")
	}

	if ls.gormDB != nil {
		return nil
	}

	var dialector gorm.Dialector
	switch ls.mode {
	case "postgres":
		dialector = postgres.New(postgres.Config{Conn: ls.db.DB})
	default:
		dialector = sqlite.Dialector{Conn: ls.db.DB, DriverName: "sqlite3"}
	}
	gormDB, err := gorm.Open(dialector, &gorm.Config{
		Logger:         logger.Default.LogMode(logger.Silent),
		NamingStrategy: schema.NamingStrategy{SingularTable: false},
	})
	if err != nil {
		return fmt.Errorf("failed to initialize gorm: %w", err)
	}

	ls.gormDB = gormDB
	return nil
}

// gormWithContext returns the configured GORM handle scoped to the provided context.
func (ls *LocalStorage) gormWithContext(ctx context.Context) (*gorm.DB, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled: %w", err)
	}

	if ls.gormDB == nil {
		if err := ls.initGormDB(); err != nil {
			return nil, err
		}
	}

	return ls.gormDB.WithContext(ctx), nil
}
