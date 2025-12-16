package storage

import "time"

type ExecutionRecordModel struct {
	ID                int64      `gorm:"column:id;primaryKey;autoIncrement"`
	ExecutionID       string     `gorm:"column:execution_id;not null;uniqueIndex"`
	RunID             string     `gorm:"column:run_id;not null;index"`
	ParentExecutionID *string    `gorm:"column:parent_execution_id;index"`
	AgentNodeID       string     `gorm:"column:agent_node_id;not null;index"`
	ReasonerID        string     `gorm:"column:reasoner_id;not null;index"`
	NodeID            string     `gorm:"column:node_id;not null;index"`
	Status            string     `gorm:"column:status;not null;index"`
	InputPayload      []byte     `gorm:"column:input_payload"`
	ResultPayload     []byte     `gorm:"column:result_payload"`
	ErrorMessage      *string    `gorm:"column:error_message"`
	InputURI          *string    `gorm:"column:input_uri"`
	ResultURI         *string    `gorm:"column:result_uri"`
	SessionID         *string    `gorm:"column:session_id;index"`
	ActorID           *string    `gorm:"column:actor_id;index"`
	StartedAt         time.Time  `gorm:"column:started_at;not null;index"`
	CompletedAt       *time.Time `gorm:"column:completed_at"`
	DurationMS        *int64     `gorm:"column:duration_ms"`
	Notes             string     `gorm:"column:notes;default:'[]'"`
	CreatedAt         time.Time  `gorm:"column:created_at;autoCreateTime"`
	UpdatedAt         time.Time  `gorm:"column:updated_at;autoUpdateTime"`
}

func (ExecutionRecordModel) TableName() string { return "executions" }

type AgentExecutionModel struct {
	ID           int64     `gorm:"column:id;primaryKey;autoIncrement"`
	WorkflowID   string    `gorm:"column:workflow_id;not null;index"`
	SessionID    *string   `gorm:"column:session_id;index"`
	AgentNodeID  string    `gorm:"column:agent_node_id;not null;index"`
	ReasonerID   string    `gorm:"column:reasoner_id;not null;index"`
	InputData    []byte    `gorm:"column:input_data"`
	OutputData   []byte    `gorm:"column:output_data"`
	InputSize    int       `gorm:"column:input_size"`
	OutputSize   int       `gorm:"column:output_size"`
	DurationMS   int       `gorm:"column:duration_ms;not null"`
	Status       string    `gorm:"column:status;not null;index"`
	ErrorMessage *string   `gorm:"column:error_message"`
	UserID       *string   `gorm:"column:user_id"`
	TeamID       *string   `gorm:"column:team_id"`
	Metadata     []byte    `gorm:"column:metadata"`
	CreatedAt    time.Time `gorm:"column:created_at;autoCreateTime"`
}

func (AgentExecutionModel) TableName() string { return "agent_executions" }

type AgentNodeModel struct {
	ID                  string     `gorm:"column:id;primaryKey"`
	TeamID              string     `gorm:"column:team_id;not null;index"`
	BaseURL             string     `gorm:"column:base_url;not null"`
	Version             string     `gorm:"column:version;not null"`
	DeploymentType      string     `gorm:"column:deployment_type;default:'long_running';index"`
	InvocationURL       *string    `gorm:"column:invocation_url"`
	Reasoners           []byte     `gorm:"column:reasoners"`
	Skills              []byte     `gorm:"column:skills"`
	CommunicationConfig []byte     `gorm:"column:communication_config"`
	HealthStatus        string     `gorm:"column:health_status;not null;index"`
	LifecycleStatus     string     `gorm:"column:lifecycle_status;default:'starting';index"`
	LastHeartbeat       *time.Time `gorm:"column:last_heartbeat"`
	RegisteredAt        time.Time  `gorm:"column:registered_at;autoCreateTime"`
	Features            []byte     `gorm:"column:features"`
	Metadata            []byte     `gorm:"column:metadata"`
}

func (AgentNodeModel) TableName() string { return "agent_nodes" }

type AgentConfigurationModel struct {
	ID              int64     `gorm:"column:id;primaryKey;autoIncrement"`
	AgentID         string    `gorm:"column:agent_id;not null;index:idx_agent_config_agent_package,priority:1"`
	PackageID       string    `gorm:"column:package_id;not null;index:idx_agent_config_agent_package,priority:2"`
	Configuration   []byte    `gorm:"column:configuration;not null"`
	EncryptedFields []byte    `gorm:"column:encrypted_fields"`
	Status          string    `gorm:"column:status;not null"`
	Version         int       `gorm:"column:version;not null;default:1"`
	CreatedAt       time.Time `gorm:"column:created_at;autoCreateTime"`
	UpdatedAt       time.Time `gorm:"column:updated_at;autoUpdateTime"`
	CreatedBy       *string   `gorm:"column:created_by"`
	UpdatedBy       *string   `gorm:"column:updated_by"`
}

func (AgentConfigurationModel) TableName() string { return "agent_configurations" }

type AgentPackageModel struct {
	ID                  string    `gorm:"column:id;primaryKey"`
	Name                string    `gorm:"column:name;not null"`
	Version             string    `gorm:"column:version;not null"`
	Description         *string   `gorm:"column:description"`
	Author              *string   `gorm:"column:author"`
	Repository          *string   `gorm:"column:repository"`
	InstallPath         string    `gorm:"column:install_path;not null"`
	ConfigurationSchema []byte    `gorm:"column:configuration_schema"`
	Status              string    `gorm:"column:status;not null"`
	ConfigurationStatus string    `gorm:"column:configuration_status;not null"`
	InstalledAt         time.Time `gorm:"column:installed_at;autoCreateTime"`
	UpdatedAt           time.Time `gorm:"column:updated_at;autoUpdateTime"`
	Metadata            []byte    `gorm:"column:metadata"`
}

func (AgentPackageModel) TableName() string { return "agent_packages" }

type WorkflowExecutionModel struct {
	ID                    int64      `gorm:"column:id;primaryKey;autoIncrement"`
	WorkflowID            string     `gorm:"column:workflow_id;not null;index;index:idx_workflow_executions_workflow_status,priority:1"`
	ExecutionID           string     `gorm:"column:execution_id;not null;uniqueIndex"`
	AgentFieldRequestID   string     `gorm:"column:agentfield_request_id;not null;index"`
	RunID                 *string    `gorm:"column:run_id;index"`
	SessionID             *string    `gorm:"column:session_id;index;index:idx_workflow_executions_session_status,priority:1;index:idx_workflow_executions_session_status_time,priority:1;index:idx_workflow_executions_session_time,priority:1"`
	ActorID               *string    `gorm:"column:actor_id;index;index:idx_workflow_executions_actor_status,priority:1;index:idx_workflow_executions_actor_status_time,priority:1;index:idx_workflow_executions_actor_time,priority:1"`
	AgentNodeID           string     `gorm:"column:agent_node_id;not null;index;index:idx_workflow_executions_agent_node_status,priority:1;index:idx_workflow_executions_agent_status_time,priority:1"`
	ParentWorkflowID      *string    `gorm:"column:parent_workflow_id;index"`
	ParentExecutionID     *string    `gorm:"column:parent_execution_id;index"`
	RootWorkflowID        *string    `gorm:"column:root_workflow_id;index"`
	WorkflowDepth         int        `gorm:"column:workflow_depth;default:0"`
	ReasonerID            string     `gorm:"column:reasoner_id;not null"`
	InputData             []byte     `gorm:"column:input_data"`
	OutputData            []byte     `gorm:"column:output_data"`
	InputSize             int        `gorm:"column:input_size"`
	OutputSize            int        `gorm:"column:output_size"`
	WorkflowName          *string    `gorm:"column:workflow_name"`
	WorkflowTags          string     `gorm:"column:workflow_tags"`
	Status                string     `gorm:"column:status;not null;index;index:idx_workflow_executions_agent_node_status,priority:2;index:idx_workflow_executions_session_status,priority:2;index:idx_workflow_executions_actor_status,priority:2;index:idx_workflow_executions_workflow_status,priority:2;index:idx_workflow_executions_status_time,priority:1;index:idx_workflow_executions_session_status_time,priority:2;index:idx_workflow_executions_actor_status_time,priority:2;index:idx_workflow_executions_agent_status_time,priority:2"`
	StartedAt             time.Time  `gorm:"column:started_at;not null;index;index:idx_workflow_executions_status_time,priority:2;index:idx_workflow_executions_session_status_time,priority:3;index:idx_workflow_executions_actor_status_time,priority:3;index:idx_workflow_executions_agent_status_time,priority:3;index:idx_workflow_executions_session_time,priority:2;index:idx_workflow_executions_actor_time,priority:2"`
	CompletedAt           *time.Time `gorm:"column:completed_at"`
	DurationMS            int        `gorm:"column:duration_ms"`
	StateVersion          int        `gorm:"column:state_version;not null;default:0"`
	LastEventSequence     int        `gorm:"column:last_event_sequence;not null;default:0"`
	ActiveChildren        int        `gorm:"column:active_children;not null;default:0"`
	PendingChildren       int        `gorm:"column:pending_children;not null;default:0"`
	PendingTerminalStatus *string    `gorm:"column:pending_terminal_status"`
	StatusReason          *string    `gorm:"column:status_reason"`
	LeaseOwner            *string    `gorm:"column:lease_owner"`
	LeaseExpiresAt        *time.Time `gorm:"column:lease_expires_at"`
	ErrorMessage          *string    `gorm:"column:error_message"`
	RetryCount            int        `gorm:"column:retry_count;default:0"`
	Notes                 string     `gorm:"column:notes;default:'[]'"`
	CreatedAt             time.Time  `gorm:"column:created_at;autoCreateTime"`
	UpdatedAt             time.Time  `gorm:"column:updated_at;autoUpdateTime"`
}

func (WorkflowExecutionModel) TableName() string { return "workflow_executions" }

type WorkflowExecutionEventModel struct {
	EventID           int64     `gorm:"column:event_id;primaryKey;autoIncrement"`
	ExecutionID       string    `gorm:"column:execution_id;not null;index:idx_workflow_exec_events_execution,priority:1"`
	WorkflowID        string    `gorm:"column:workflow_id;not null"`
	RunID             *string   `gorm:"column:run_id;index:idx_workflow_exec_events_run,priority:1"`
	ParentExecutionID *string   `gorm:"column:parent_execution_id"`
	Sequence          int64     `gorm:"column:sequence;not null;index:idx_workflow_exec_events_execution,priority:2"`
	PreviousSequence  int64     `gorm:"column:previous_sequence;not null;default:0"`
	EventType         string    `gorm:"column:event_type;not null"`
	Status            *string   `gorm:"column:status"`
	StatusReason      *string   `gorm:"column:status_reason"`
	Payload           string    `gorm:"column:payload;default:'{}'"`
	EmittedAt         time.Time `gorm:"column:emitted_at;not null"`
	RecordedAt        time.Time `gorm:"column:recorded_at;autoCreateTime"`
}

func (WorkflowExecutionEventModel) TableName() string { return "workflow_execution_events" }

type WorkflowRunEventModel struct {
	EventID          int64     `gorm:"column:event_id;primaryKey;autoIncrement"`
	RunID            string    `gorm:"column:run_id;not null;index:idx_workflow_run_events_run,priority:1"`
	Sequence         int64     `gorm:"column:sequence;not null;index:idx_workflow_run_events_run,priority:2"`
	PreviousSequence int64     `gorm:"column:previous_sequence;not null;default:0"`
	EventType        string    `gorm:"column:event_type;not null"`
	Status           *string   `gorm:"column:status"`
	StatusReason     *string   `gorm:"column:status_reason"`
	Payload          string    `gorm:"column:payload;default:'{}'"`
	EmittedAt        time.Time `gorm:"column:emitted_at;not null"`
	RecordedAt       time.Time `gorm:"column:recorded_at;autoCreateTime"`
}

func (WorkflowRunEventModel) TableName() string { return "workflow_run_events" }

type WorkflowRunModel struct {
	RunID             string     `gorm:"column:run_id;primaryKey"`
	RootWorkflowID    string     `gorm:"column:root_workflow_id;not null;index"`
	RootExecutionID   *string    `gorm:"column:root_execution_id"`
	Status            string     `gorm:"column:status;not null;default:'pending';index"`
	TotalSteps        int        `gorm:"column:total_steps;not null;default:0"`
	CompletedSteps    int        `gorm:"column:completed_steps;not null;default:0"`
	FailedSteps       int        `gorm:"column:failed_steps;not null;default:0"`
	StateVersion      int64      `gorm:"column:state_version;not null;default:0"`
	LastEventSequence int64      `gorm:"column:last_event_sequence;not null;default:0"`
	Metadata          []byte     `gorm:"column:metadata;default:'{}'"`
	CreatedAt         time.Time  `gorm:"column:created_at;autoCreateTime;index"`
	UpdatedAt         time.Time  `gorm:"column:updated_at;autoUpdateTime;index"`
	CompletedAt       *time.Time `gorm:"column:completed_at;index"`
}

func (WorkflowRunModel) TableName() string { return "workflow_runs" }

type WorkflowStepModel struct {
	StepID       string     `gorm:"column:step_id;primaryKey"`
	RunID        string     `gorm:"column:run_id;not null;index;index:idx_workflow_steps_run_execution,priority:1;index:idx_workflow_steps_run_status,priority:1;index:idx_workflow_steps_run_priority,priority:1"`
	ParentStepID *string    `gorm:"column:parent_step_id;index"`
	ExecutionID  *string    `gorm:"column:execution_id;index:idx_workflow_steps_run_execution,priority:2"`
	AgentNodeID  *string    `gorm:"column:agent_node_id;index;index:idx_workflow_steps_agent_not_before,priority:1"`
	Target       *string    `gorm:"column:target"`
	Status       string     `gorm:"column:status;not null;default:'pending';index;index:idx_workflow_steps_run_status,priority:2;index:idx_workflow_steps_status_not_before,priority:1;index:idx_workflow_steps_agent_not_before,priority:2"`
	Attempt      int        `gorm:"column:attempt;not null;default:0"`
	Priority     int        `gorm:"column:priority;not null;default:0;index:idx_workflow_steps_run_priority,priority:2"`
	NotBefore    time.Time  `gorm:"column:not_before;not null;index:idx_workflow_steps_status_not_before,priority:2;index:idx_workflow_steps_agent_not_before,priority:3;index:idx_workflow_steps_run_priority,priority:3"`
	InputURI     *string    `gorm:"column:input_uri"`
	ResultURI    *string    `gorm:"column:result_uri"`
	ErrorMessage *string    `gorm:"column:error_message"`
	Metadata     []byte     `gorm:"column:metadata;default:'{}'"`
	StartedAt    *time.Time `gorm:"column:started_at"`
	CompletedAt  *time.Time `gorm:"column:completed_at"`
	LeasedAt     *time.Time `gorm:"column:leased_at"`
	LeaseTimeout *time.Time `gorm:"column:lease_timeout"`
	CreatedAt    time.Time  `gorm:"column:created_at;autoCreateTime;index"`
	UpdatedAt    time.Time  `gorm:"column:updated_at;autoUpdateTime;index"`
}

func (WorkflowStepModel) TableName() string { return "workflow_steps" }

type WorkflowModel struct {
	WorkflowID           string     `gorm:"column:workflow_id;primaryKey"`
	WorkflowName         *string    `gorm:"column:workflow_name"`
	WorkflowTags         string     `gorm:"column:workflow_tags"`
	SessionID            *string    `gorm:"column:session_id;index"`
	ActorID              *string    `gorm:"column:actor_id;index"`
	ParentWorkflowID     *string    `gorm:"column:parent_workflow_id"`
	ParentExecutionID    *string    `gorm:"column:parent_execution_id"`
	RootWorkflowID       *string    `gorm:"column:root_workflow_id"`
	WorkflowDepth        int        `gorm:"column:workflow_depth;default:0"`
	TotalExecutions      int        `gorm:"column:total_executions;default:0"`
	SuccessfulExecutions int        `gorm:"column:successful_executions;default:0"`
	FailedExecutions     int        `gorm:"column:failed_executions;default:0"`
	TotalDurationMS      int        `gorm:"column:total_duration_ms;default:0"`
	Status               string     `gorm:"column:status;not null"`
	StartedAt            time.Time  `gorm:"column:started_at;not null"`
	CompletedAt          *time.Time `gorm:"column:completed_at"`
	CreatedAt            time.Time  `gorm:"column:created_at;autoCreateTime"`
	UpdatedAt            time.Time  `gorm:"column:updated_at;autoUpdateTime"`
}

func (WorkflowModel) TableName() string { return "workflows" }

type SessionModel struct {
	SessionID       string    `gorm:"column:session_id;primaryKey"`
	ActorID         *string   `gorm:"column:actor_id;index"`
	SessionName     *string   `gorm:"column:session_name"`
	ParentSessionID *string   `gorm:"column:parent_session_id"`
	RootSessionID   *string   `gorm:"column:root_session_id;index"`
	TotalWorkflows  int       `gorm:"column:total_workflows;default:0"`
	TotalExecutions int       `gorm:"column:total_executions;default:0"`
	TotalDurationMS int       `gorm:"column:total_duration_ms;default:0"`
	StartedAt       time.Time `gorm:"column:started_at;not null"`
	LastActivityAt  time.Time `gorm:"column:last_activity_at;not null"`
	CreatedAt       time.Time `gorm:"column:created_at;autoCreateTime"`
	UpdatedAt       time.Time `gorm:"column:updated_at;autoUpdateTime"`
}

func (SessionModel) TableName() string { return "sessions" }

type DIDRegistryModel struct {
	AgentFieldServerID  string    `gorm:"column:agentfield_server_id;primaryKey"`
	MasterSeedEncrypted []byte    `gorm:"column:master_seed_encrypted;not null"`
	RootDID             string    `gorm:"column:root_did;not null;unique"`
	AgentNodes          string    `gorm:"column:agent_nodes;default:'{}'"`
	TotalDIDs           int       `gorm:"column:total_dids;default:0"`
	CreatedAt           time.Time `gorm:"column:created_at;autoCreateTime"`
	LastKeyRotation     time.Time `gorm:"column:last_key_rotation;autoCreateTime"`
}

func (DIDRegistryModel) TableName() string { return "did_registry" }

type AgentDIDModel struct {
	DID                string    `gorm:"column:did;primaryKey"`
	AgentNodeID        string    `gorm:"column:agent_node_id;not null;index"`
	AgentFieldServerID string    `gorm:"column:agentfield_server_id;not null;index"`
	PublicKeyJWK       string    `gorm:"column:public_key_jwk;not null"`
	DerivationPath     string    `gorm:"column:derivation_path;not null"`
	Reasoners          string    `gorm:"column:reasoners;default:'{}'"`
	Skills             string    `gorm:"column:skills;default:'{}'"`
	Status             string    `gorm:"column:status;not null;default:'active'"`
	RegisteredAt       time.Time `gorm:"column:registered_at;autoCreateTime"`
	CreatedAt          time.Time `gorm:"column:created_at;autoCreateTime"`
	UpdatedAt          time.Time `gorm:"column:updated_at;autoUpdateTime"`
}

func (AgentDIDModel) TableName() string { return "agent_dids" }

type ComponentDIDModel struct {
	DID            string    `gorm:"column:did;primaryKey"`
	AgentDID       string    `gorm:"column:agent_did;not null;index"`
	ComponentType  string    `gorm:"column:component_type;not null;index"`
	FunctionName   string    `gorm:"column:function_name;not null"`
	PublicKeyJWK   string    `gorm:"column:public_key_jwk;not null"`
	DerivationPath string    `gorm:"column:derivation_path;not null"`
	Capabilities   string    `gorm:"column:capabilities;default:'[]'"`
	Tags           string    `gorm:"column:tags;default:'[]'"`
	ExposureLevel  string    `gorm:"column:exposure_level;not null;default:'private'"`
	CreatedAt      time.Time `gorm:"column:created_at;autoCreateTime"`
	UpdatedAt      time.Time `gorm:"column:updated_at;autoUpdateTime"`
}

func (ComponentDIDModel) TableName() string { return "component_dids" }

type ExecutionVCModel struct {
	VCID              string    `gorm:"column:vc_id;primaryKey"`
	ExecutionID       string    `gorm:"column:execution_id;not null;index;index:idx_execution_vcs_execution_unique,priority:1"`
	WorkflowID        string    `gorm:"column:workflow_id;not null;index"`
	SessionID         string    `gorm:"column:session_id;not null;index"`
	IssuerDID         string    `gorm:"column:issuer_did;not null;index;index:idx_execution_vcs_execution_unique,priority:2"`
	TargetDID         *string   `gorm:"column:target_did;index;index:idx_execution_vcs_execution_unique,priority:3"`
	CallerDID         string    `gorm:"column:caller_did;not null;index"`
	VCDocument        string    `gorm:"column:vc_document;not null"`
	Signature         string    `gorm:"column:signature;not null"`
	StorageURI        string    `gorm:"column:storage_uri;default:''"`
	DocumentSizeBytes int64     `gorm:"column:document_size_bytes;default:0"`
	InputHash         string    `gorm:"column:input_hash;not null"`
	OutputHash        string    `gorm:"column:output_hash;not null"`
	Status            string    `gorm:"column:status;not null;default:'pending';index"`
	ParentVCID        *string   `gorm:"column:parent_vc_id;index"`
	ChildVCIDs        string    `gorm:"column:child_vc_ids;default:'[]'"`
	CreatedAt         time.Time `gorm:"column:created_at;autoCreateTime;index"`
	UpdatedAt         time.Time `gorm:"column:updated_at;autoUpdateTime"`
}

func (ExecutionVCModel) TableName() string { return "execution_vcs" }

type WorkflowVCModel struct {
	WorkflowVCID      string     `gorm:"column:workflow_vc_id;primaryKey"`
	WorkflowID        string     `gorm:"column:workflow_id;not null;index"`
	SessionID         string     `gorm:"column:session_id;not null;index"`
	ComponentVCIDs    string     `gorm:"column:component_vc_ids;default:'[]'"`
	Status            string     `gorm:"column:status;not null;default:'pending';index"`
	StartTime         time.Time  `gorm:"column:start_time;autoCreateTime;index"`
	EndTime           *time.Time `gorm:"column:end_time;index"`
	TotalSteps        int        `gorm:"column:total_steps;default:0"`
	CompletedSteps    int        `gorm:"column:completed_steps;default:0"`
	StorageURI        string     `gorm:"column:storage_uri;default:''"`
	DocumentSizeBytes int64      `gorm:"column:document_size_bytes;default:0"`
	CreatedAt         time.Time  `gorm:"column:created_at;autoCreateTime;index"`
	UpdatedAt         time.Time  `gorm:"column:updated_at;autoUpdateTime"`
}

func (WorkflowVCModel) TableName() string { return "workflow_vcs" }

type SchemaMigrationModel struct {
	Version     string    `gorm:"column:version;primaryKey"`
	AppliedAt   time.Time `gorm:"column:applied_at;autoCreateTime"`
	Description string    `gorm:"column:description"`
}

func (SchemaMigrationModel) TableName() string { return "schema_migrations" }

type ExecutionWebhookEventModel struct {
	ID           int64     `gorm:"column:id;primaryKey;autoIncrement"`
	ExecutionID  string    `gorm:"column:execution_id;not null;index"`
	EventType    string    `gorm:"column:event_type;not null"`
	Status       string    `gorm:"column:status;not null"`
	HTTPStatus   *int      `gorm:"column:http_status"`
	Payload      *string   `gorm:"column:payload"`
	ResponseBody *string   `gorm:"column:response_body"`
	ErrorMessage *string   `gorm:"column:error_message"`
	CreatedAt    time.Time `gorm:"column:created_at;autoCreateTime"`
}

func (ExecutionWebhookEventModel) TableName() string { return "execution_webhook_events" }

type ExecutionWebhookModel struct {
	ExecutionID   string     `gorm:"column:execution_id;primaryKey"`
	URL           string     `gorm:"column:url;not null"`
	Secret        *string    `gorm:"column:secret"`
	Headers       string     `gorm:"column:headers;default:'{}'"`
	Status        string     `gorm:"column:status;not null;default:'pending'"`
	AttemptCount  int        `gorm:"column:attempt_count;not null;default:0"`
	NextAttemptAt *time.Time `gorm:"column:next_attempt_at"`
	LastAttemptAt *time.Time `gorm:"column:last_attempt_at"`
	LastError     *string    `gorm:"column:last_error"`
	CreatedAt     time.Time  `gorm:"column:created_at;autoCreateTime"`
	UpdatedAt     time.Time  `gorm:"column:updated_at;autoUpdateTime"`
}

func (ExecutionWebhookModel) TableName() string { return "execution_webhooks" }
