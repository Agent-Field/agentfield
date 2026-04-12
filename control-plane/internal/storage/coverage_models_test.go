package storage

import "testing"

func TestModelTableNames(t *testing.T) {
	cases := []struct {
		name string
		got  string
		want string
	}{
		{name: "ExecutionRecordModel", got: (ExecutionRecordModel{}).TableName(), want: "executions"},
		{name: "AgentExecutionModel", got: (AgentExecutionModel{}).TableName(), want: "agent_executions"},
		{name: "AgentNodeModel", got: (AgentNodeModel{}).TableName(), want: "agent_nodes"},
		{name: "AgentConfigurationModel", got: (AgentConfigurationModel{}).TableName(), want: "agent_configurations"},
		{name: "AgentPackageModel", got: (AgentPackageModel{}).TableName(), want: "agent_packages"},
		{name: "WorkflowExecutionModel", got: (WorkflowExecutionModel{}).TableName(), want: "workflow_executions"},
		{name: "WorkflowExecutionEventModel", got: (WorkflowExecutionEventModel{}).TableName(), want: "workflow_execution_events"},
		{name: "ExecutionLogEntryModel", got: (ExecutionLogEntryModel{}).TableName(), want: "execution_logs"},
		{name: "WorkflowRunEventModel", got: (WorkflowRunEventModel{}).TableName(), want: "workflow_run_events"},
		{name: "WorkflowRunModel", got: (WorkflowRunModel{}).TableName(), want: "workflow_runs"},
		{name: "WorkflowStepModel", got: (WorkflowStepModel{}).TableName(), want: "workflow_steps"},
		{name: "WorkflowModel", got: (WorkflowModel{}).TableName(), want: "workflows"},
		{name: "SessionModel", got: (SessionModel{}).TableName(), want: "sessions"},
		{name: "DIDRegistryModel", got: (DIDRegistryModel{}).TableName(), want: "did_registry"},
		{name: "AgentDIDModel", got: (AgentDIDModel{}).TableName(), want: "agent_dids"},
		{name: "ComponentDIDModel", got: (ComponentDIDModel{}).TableName(), want: "component_dids"},
		{name: "ExecutionVCModel", got: (ExecutionVCModel{}).TableName(), want: "execution_vcs"},
		{name: "WorkflowVCModel", got: (WorkflowVCModel{}).TableName(), want: "workflow_vcs"},
		{name: "SchemaMigrationModel", got: (SchemaMigrationModel{}).TableName(), want: "schema_migrations"},
		{name: "ExecutionWebhookEventModel", got: (ExecutionWebhookEventModel{}).TableName(), want: "execution_webhook_events"},
		{name: "ExecutionWebhookModel", got: (ExecutionWebhookModel{}).TableName(), want: "execution_webhooks"},
		{name: "ObservabilityWebhookModel", got: (ObservabilityWebhookModel{}).TableName(), want: "observability_webhooks"},
		{name: "ObservabilityDeadLetterQueueModel", got: (ObservabilityDeadLetterQueueModel{}).TableName(), want: "observability_dead_letter_queue"},
		{name: "DIDDocumentModel", got: (DIDDocumentModel{}).TableName(), want: "did_documents"},
		{name: "AccessPolicyModel", got: (AccessPolicyModel{}).TableName(), want: "access_policies"},
		{name: "AgentTagVCModel", got: (AgentTagVCModel{}).TableName(), want: "agent_tag_vcs"},
		{name: "ConfigStorageModel", got: (ConfigStorageModel{}).TableName(), want: "config_storage"},
	}

	for _, tc := range cases {
		if tc.got != tc.want {
			t.Fatalf("%s: got %q want %q", tc.name, tc.got, tc.want)
		}
	}
}
