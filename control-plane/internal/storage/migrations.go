package storage

import (
	"context"
	"fmt"
)

func (ls *LocalStorage) autoMigrateSchema(ctx context.Context) error {
	gormDB, err := ls.gormWithContext(ctx)
	if err != nil {
		return fmt.Errorf("failed to initialize gorm for migrations: %w", err)
	}

	if ls.mode == "local" {
		if err := gormDB.Exec("PRAGMA foreign_keys = OFF").Error; err != nil {
			return fmt.Errorf("failed to disable foreign keys: %w", err)
		}
		defer func() {
			if err := gormDB.Exec("PRAGMA foreign_keys = ON").Error; err != nil {
				fmt.Printf("failed to re-enable foreign keys: %v\n", err)
			}
		}()
	}

	models := []interface{}{
		&ExecutionRecordModel{},
		&AgentExecutionModel{},
		&AgentNodeModel{},
		&AgentConfigurationModel{},
		&AgentPackageModel{},
		&WorkflowExecutionModel{},
		&WorkflowExecutionEventModel{},
		&WorkflowRunEventModel{},
		&WorkflowRunModel{},
		&WorkflowStepModel{},
		&WorkflowModel{},
		&SessionModel{},
		&DIDRegistryModel{},
		&AgentDIDModel{},
		&ComponentDIDModel{},
		&ExecutionVCModel{},
		&WorkflowVCModel{},
		&SchemaMigrationModel{},
		&ExecutionWebhookEventModel{},
		&ExecutionWebhookModel{},
		&ObservabilityWebhookModel{},
		&ObservabilityDeadLetterQueueModel{},
	}

	if err := gormDB.WithContext(ctx).AutoMigrate(models...); err != nil {
		return fmt.Errorf("failed to auto-migrate schema: %w", err)
	}

	return nil
}
