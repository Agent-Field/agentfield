package apicatalog

// DefaultEntries returns the curated endpoint catalog for the AgentField control plane.
// Routes not listed here are auto-registered with generic summaries.
func DefaultEntries() []EndpointEntry {
	return []EndpointEntry{
		// --- Health ---
		{Method: "GET", Path: "/health", Group: "health", Summary: "Server health check", AuthLevel: "public", Tags: []string{"health", "monitoring"}},
		{Method: "GET", Path: "/api/v1/health", Group: "health", Summary: "API health check", AuthLevel: "public", Tags: []string{"health", "monitoring"}},
		{Method: "GET", Path: "/metrics", Group: "health", Summary: "Prometheus metrics", AuthLevel: "public", Tags: []string{"metrics", "monitoring", "prometheus"}},

		// --- Discovery ---
		{Method: "GET", Path: "/api/v1/discovery/capabilities", Group: "discovery", Summary: "Discover agent capabilities, reasoners, and skills", AuthLevel: "api_key", Tags: []string{"discovery", "capabilities", "agents", "reasoners", "skills"},
			Parameters: []ParamEntry{
				{Name: "agent_ids", In: "query", Type: "string", Desc: "Comma-separated agent IDs to filter"},
				{Name: "reasoner_pattern", In: "query", Type: "string", Desc: "Regex pattern to filter reasoners"},
				{Name: "skill_pattern", In: "query", Type: "string", Desc: "Regex pattern to filter skills"},
				{Name: "tags", In: "query", Type: "string", Desc: "Comma-separated tags to filter"},
				{Name: "include_input_schema", In: "query", Type: "bool", Desc: "Include input schemas"},
				{Name: "include_output_schema", In: "query", Type: "bool", Desc: "Include output schemas"},
				{Name: "format", In: "query", Type: "string", Desc: "Response format: json, openai, xml, markdown"},
				{Name: "limit", In: "query", Type: "int", Desc: "Max results"},
				{Name: "offset", In: "query", Type: "int", Desc: "Pagination offset"},
			},
		},

		// --- Nodes ---
		{Method: "POST", Path: "/api/v1/nodes/register", Group: "nodes", Summary: "Register a new agent node", AuthLevel: "api_key", Tags: []string{"nodes", "register", "agents"}},
		{Method: "POST", Path: "/api/v1/nodes", Group: "nodes", Summary: "Register a new agent node (alias)", AuthLevel: "api_key", Tags: []string{"nodes", "register", "agents"}},
		{Method: "POST", Path: "/api/v1/nodes/register-serverless", Group: "nodes", Summary: "Register a serverless agent", AuthLevel: "api_key", Tags: []string{"nodes", "register", "serverless"}},
		{Method: "GET", Path: "/api/v1/nodes", Group: "nodes", Summary: "List all registered nodes", AuthLevel: "api_key", Tags: []string{"nodes", "list"}},
		{Method: "GET", Path: "/api/v1/nodes/:node_id", Group: "nodes", Summary: "Get a specific node by ID", AuthLevel: "api_key", Tags: []string{"nodes", "get"},
			Parameters: []ParamEntry{{Name: "node_id", In: "path", Required: true, Type: "string", Desc: "Node ID"}},
		},
		{Method: "POST", Path: "/api/v1/nodes/:node_id/heartbeat", Group: "nodes", Summary: "Send node heartbeat", AuthLevel: "api_key", Tags: []string{"nodes", "heartbeat", "health"}},
		{Method: "DELETE", Path: "/api/v1/nodes/:node_id/monitoring", Group: "nodes", Summary: "Unregister agent from monitoring", AuthLevel: "api_key", Tags: []string{"nodes", "monitoring"}},
		{Method: "GET", Path: "/api/v1/nodes/:node_id/status", Group: "nodes", Summary: "Get unified node status", AuthLevel: "api_key", Tags: []string{"nodes", "status"}},
		{Method: "POST", Path: "/api/v1/nodes/:node_id/status/refresh", Group: "nodes", Summary: "Force refresh node status", AuthLevel: "api_key", Tags: []string{"nodes", "status", "refresh"}},
		{Method: "POST", Path: "/api/v1/nodes/status/bulk", Group: "nodes", Summary: "Bulk node status query", AuthLevel: "api_key", Tags: []string{"nodes", "status", "bulk"}},
		{Method: "POST", Path: "/api/v1/nodes/status/refresh", Group: "nodes", Summary: "Refresh all node statuses", AuthLevel: "api_key", Tags: []string{"nodes", "status", "refresh"}},
		{Method: "POST", Path: "/api/v1/nodes/:node_id/start", Group: "nodes", Summary: "Start a node", AuthLevel: "api_key", Tags: []string{"nodes", "lifecycle", "start"}},
		{Method: "POST", Path: "/api/v1/nodes/:node_id/stop", Group: "nodes", Summary: "Stop a node", AuthLevel: "api_key", Tags: []string{"nodes", "lifecycle", "stop"}},
		{Method: "POST", Path: "/api/v1/nodes/:node_id/lifecycle/status", Group: "nodes", Summary: "Update node lifecycle status", AuthLevel: "api_key", Tags: []string{"nodes", "lifecycle", "status"}},
		{Method: "PATCH", Path: "/api/v1/nodes/:node_id/status", Group: "nodes", Summary: "Lease-based status update", AuthLevel: "api_key", Tags: []string{"nodes", "status", "lease"}},
		{Method: "POST", Path: "/api/v1/nodes/:node_id/actions/ack", Group: "nodes", Summary: "Acknowledge node action", AuthLevel: "api_key", Tags: []string{"nodes", "actions"}},
		{Method: "POST", Path: "/api/v1/nodes/:node_id/shutdown", Group: "nodes", Summary: "Graceful node shutdown", AuthLevel: "api_key", Tags: []string{"nodes", "lifecycle", "shutdown"}},
		{Method: "POST", Path: "/api/v1/actions/claim", Group: "nodes", Summary: "Claim pending actions", AuthLevel: "api_key", Tags: []string{"nodes", "actions", "claim"}},

		// --- UI: node logs (proxy to agent NDJSON) ---
		{Method: "GET", Path: "/api/ui/v1/nodes/:nodeId/logs", Group: "ui-nodes", Summary: "Proxy agent process logs (NDJSON tail or follow)", AuthLevel: "api_key", Tags: []string{"ui", "nodes", "logs", "observability"},
			Parameters: []ParamEntry{
				{Name: "nodeId", In: "path", Required: true, Type: "string", Desc: "Agent node ID"},
				{Name: "tail_lines", In: "query", Type: "int", Desc: "Last N log lines"},
				{Name: "since_seq", In: "query", Type: "int", Desc: "Return lines with seq greater than this"},
				{Name: "follow", In: "query", Type: "string", Desc: "1 or true for chunked stream"},
			},
		},
		{Method: "GET", Path: "/api/ui/v1/settings/node-log-proxy", Group: "ui-settings", Summary: "Effective node log proxy limits and env lock flags", AuthLevel: "api_key", Tags: []string{"ui", "settings", "logs"}},
		{Method: "PUT", Path: "/api/ui/v1/settings/node-log-proxy", Group: "ui-settings", Summary: "Update node log proxy limits (persisted to DB config blob)", AuthLevel: "api_key", Tags: []string{"ui", "settings", "logs"}},

		// --- Execute ---
		{Method: "POST", Path: "/api/v1/execute/:target", Group: "execute", Summary: "Execute a reasoner or skill synchronously", AuthLevel: "api_key", Tags: []string{"execute", "reasoner", "skill", "sync"},
			Parameters: []ParamEntry{{Name: "target", In: "path", Required: true, Type: "string", Desc: "Target in format agent_id.reasoner_id or agent_id.skill_id"}},
			RequestBody: &BodyEntry{ContentType: "application/json", Fields: map[string]string{"input": "object - Input payload", "session_id": "string - Optional session ID", "run_id": "string - Optional run ID", "workflow_id": "string - Optional workflow ID"}},
		},
		{Method: "POST", Path: "/api/v1/execute/async/:target", Group: "execute", Summary: "Execute a reasoner or skill asynchronously", AuthLevel: "api_key", Tags: []string{"execute", "reasoner", "skill", "async"},
			Parameters: []ParamEntry{{Name: "target", In: "path", Required: true, Type: "string", Desc: "Target in format agent_id.reasoner_id or agent_id.skill_id"}},
		},
		{Method: "POST", Path: "/api/v1/reasoners/:reasoner_id", Group: "execute", Summary: "Execute a reasoner (legacy endpoint)", AuthLevel: "api_key", Tags: []string{"execute", "reasoner", "legacy"}},
		{Method: "POST", Path: "/api/v1/skills/:skill_id", Group: "execute", Summary: "Execute a skill (legacy endpoint)", AuthLevel: "api_key", Tags: []string{"execute", "skill", "legacy"}},

		// --- Executions ---
		{Method: "GET", Path: "/api/v1/executions/:execution_id", Group: "executions", Summary: "Get execution status", AuthLevel: "api_key", Tags: []string{"executions", "status"},
			Parameters: []ParamEntry{{Name: "execution_id", In: "path", Required: true, Type: "string", Desc: "Execution ID"}},
		},
		{Method: "POST", Path: "/api/v1/executions/batch-status", Group: "executions", Summary: "Batch execution status query", AuthLevel: "api_key", Tags: []string{"executions", "status", "batch"}},
		{Method: "POST", Path: "/api/v1/executions/:execution_id/status", Group: "executions", Summary: "Update execution status", AuthLevel: "api_key", Tags: []string{"executions", "status", "update"}},
		{Method: "POST", Path: "/api/v1/executions/:execution_id/cancel", Group: "executions", Summary: "Cancel a running execution", AuthLevel: "api_key", Tags: []string{"executions", "cancel"}},
		{Method: "POST", Path: "/api/v1/executions/:execution_id/pause", Group: "executions", Summary: "Pause an execution", AuthLevel: "api_key", Tags: []string{"executions", "pause"}},
		{Method: "POST", Path: "/api/v1/executions/:execution_id/resume", Group: "executions", Summary: "Resume a paused execution", AuthLevel: "api_key", Tags: []string{"executions", "resume"}},

		// --- Approval ---
		{Method: "POST", Path: "/api/v1/executions/:execution_id/request-approval", Group: "approval", Summary: "Request approval for an execution", AuthLevel: "api_key", Tags: []string{"approval", "request"}},
		{Method: "GET", Path: "/api/v1/executions/:execution_id/approval-status", Group: "approval", Summary: "Get approval status", AuthLevel: "api_key", Tags: []string{"approval", "status"}},
		{Method: "POST", Path: "/api/v1/agents/:node_id/executions/:execution_id/request-approval", Group: "approval", Summary: "Request approval (agent-scoped)", AuthLevel: "api_key", Tags: []string{"approval", "request", "agent-scoped"}},
		{Method: "GET", Path: "/api/v1/agents/:node_id/executions/:execution_id/approval-status", Group: "approval", Summary: "Get approval status (agent-scoped)", AuthLevel: "api_key", Tags: []string{"approval", "status", "agent-scoped"}},
		{Method: "POST", Path: "/api/v1/webhooks/approval-response", Group: "approval", Summary: "Webhook for approval responses", AuthLevel: "api_key", Tags: []string{"approval", "webhook"}},

		// --- Execution notes ---
		{Method: "POST", Path: "/api/v1/executions/note", Group: "executions", Summary: "Add an execution note (app.note())", AuthLevel: "api_key", Tags: []string{"executions", "notes"}},
		{Method: "GET", Path: "/api/v1/executions/:execution_id/notes", Group: "executions", Summary: "Get execution notes", AuthLevel: "api_key", Tags: []string{"executions", "notes"}},
		{Method: "POST", Path: "/api/v1/workflow/executions/events", Group: "executions", Summary: "Submit workflow execution events", AuthLevel: "api_key", Tags: []string{"executions", "events", "workflow"}},

		// --- Memory ---
		{Method: "POST", Path: "/api/v1/memory/set", Group: "memory", Summary: "Set a memory value", AuthLevel: "api_key", Tags: []string{"memory", "set"},
			RequestBody: &BodyEntry{ContentType: "application/json", Fields: map[string]string{"scope": "string - global|agent|session|run", "scope_id": "string", "key": "string", "value": "any"}},
		},
		{Method: "POST", Path: "/api/v1/memory/get", Group: "memory", Summary: "Get a memory value", AuthLevel: "api_key", Tags: []string{"memory", "get"}},
		{Method: "POST", Path: "/api/v1/memory/delete", Group: "memory", Summary: "Delete a memory value", AuthLevel: "api_key", Tags: []string{"memory", "delete"}},
		{Method: "GET", Path: "/api/v1/memory/list", Group: "memory", Summary: "List memory values for a scope", AuthLevel: "api_key", Tags: []string{"memory", "list"}},

		// --- Vector Memory ---
		{Method: "POST", Path: "/api/v1/memory/vector", Group: "memory", Summary: "Store a vector embedding", AuthLevel: "api_key", Tags: []string{"memory", "vector", "embeddings"}},
		{Method: "GET", Path: "/api/v1/memory/vector/:key", Group: "memory", Summary: "Get a vector by key", AuthLevel: "api_key", Tags: []string{"memory", "vector"}},
		{Method: "POST", Path: "/api/v1/memory/vector/search", Group: "memory", Summary: "Similarity search over vectors", AuthLevel: "api_key", Tags: []string{"memory", "vector", "search", "similarity"}},
		{Method: "DELETE", Path: "/api/v1/memory/vector/:key", Group: "memory", Summary: "Delete a vector", AuthLevel: "api_key", Tags: []string{"memory", "vector", "delete"}},

		// --- DID ---
		{Method: "GET", Path: "/api/v1/did/document/:agent_id", Group: "did", Summary: "Get DID document for agent", AuthLevel: "public", Tags: []string{"did", "identity", "document"}},
		{Method: "GET", Path: "/api/v1/did/resolve/:did", Group: "did", Summary: "Resolve a DID to its document", AuthLevel: "public", Tags: []string{"did", "identity", "resolve"}},
		{Method: "GET", Path: "/api/v1/did/issuer-public-keys", Group: "did", Summary: "Get issuer public keys", AuthLevel: "public", Tags: []string{"did", "identity", "keys"}},
		{Method: "GET", Path: "/api/v1/did/workflow/:workflow_id/vc-chain", Group: "did", Summary: "Get VC chain for workflow", AuthLevel: "api_key", Tags: []string{"did", "vc", "workflow", "audit"}},
		{Method: "POST", Path: "/api/v1/did/verify-audit", Group: "did", Summary: "Verify exported provenance JSON (VC chain or bare VC)", AuthLevel: "api_key", Tags: []string{"did", "vc", "verify", "audit"}},

		// --- Agentic API ---
		{Method: "GET", Path: "/api/v1/agentic/discover", Group: "agentic", Summary: "Search API endpoints by keyword, group, or method", AuthLevel: "api_key", Tags: []string{"agentic", "discover", "api", "search"}},
		{Method: "POST", Path: "/api/v1/agentic/query", Group: "agentic", Summary: "Unified resource query (runs, executions, agents)", AuthLevel: "api_key", Tags: []string{"agentic", "query", "runs", "executions"}},
		{Method: "GET", Path: "/api/v1/agentic/run/:run_id", Group: "agentic", Summary: "Complete run overview with DAG, agents, and notes", AuthLevel: "api_key", Tags: []string{"agentic", "run", "overview", "dag"}},
		{Method: "GET", Path: "/api/v1/agentic/agent/:agent_id/summary", Group: "agentic", Summary: "Agent summary with recent executions and metrics", AuthLevel: "api_key", Tags: []string{"agentic", "agent", "summary", "metrics"}},
		{Method: "POST", Path: "/api/v1/agentic/batch", Group: "agentic", Summary: "Execute up to 20 API operations in one request", AuthLevel: "api_key", Tags: []string{"agentic", "batch", "operations"}},
		{Method: "GET", Path: "/api/v1/agentic/status", Group: "agentic", Summary: "System status overview", AuthLevel: "api_key", Tags: []string{"agentic", "status", "system", "health"}},

		// --- Agentic KB ---
		{Method: "GET", Path: "/api/v1/agentic/kb/topics", Group: "agentic-kb", Summary: "List knowledge base topics with article counts", AuthLevel: "public", Tags: []string{"kb", "topics", "knowledge"}},
		{Method: "GET", Path: "/api/v1/agentic/kb/articles", Group: "agentic-kb", Summary: "Search and filter knowledge base articles", AuthLevel: "public", Tags: []string{"kb", "articles", "search"}},
		{Method: "GET", Path: "/api/v1/agentic/kb/articles/:article_id", Group: "agentic-kb", Summary: "Get full article content", AuthLevel: "public", Tags: []string{"kb", "article", "content"}},
		{Method: "GET", Path: "/api/v1/agentic/kb/guide", Group: "agentic-kb", Summary: "Goal-oriented reading path for building agents", AuthLevel: "public", Tags: []string{"kb", "guide", "learning", "onboarding"}},

		// --- Settings ---
		{Method: "GET", Path: "/api/v1/settings/webhooks", Group: "settings", Summary: "List observability webhooks", AuthLevel: "api_key", Tags: []string{"settings", "webhooks", "observability"}},
		{Method: "POST", Path: "/api/v1/settings/webhooks", Group: "settings", Summary: "Create observability webhook", AuthLevel: "api_key", Tags: []string{"settings", "webhooks", "observability"}},
		{Method: "DELETE", Path: "/api/v1/settings/webhooks/:webhook_id", Group: "settings", Summary: "Delete observability webhook", AuthLevel: "api_key", Tags: []string{"settings", "webhooks"}},

		// --- Admin ---
		{Method: "GET", Path: "/api/v1/admin/tags/pending", Group: "admin", Summary: "List pending tag approval requests", AuthLevel: "admin", Tags: []string{"admin", "tags", "approval"}},
		{Method: "POST", Path: "/api/v1/admin/tags/approve", Group: "admin", Summary: "Approve a tag request", AuthLevel: "admin", Tags: []string{"admin", "tags", "approval"}},
		{Method: "POST", Path: "/api/v1/admin/tags/reject", Group: "admin", Summary: "Reject a tag request", AuthLevel: "admin", Tags: []string{"admin", "tags", "approval"}},
		{Method: "GET", Path: "/api/v1/admin/policies", Group: "admin", Summary: "List access policies", AuthLevel: "admin", Tags: []string{"admin", "policies", "access"}},
		{Method: "POST", Path: "/api/v1/admin/policies", Group: "admin", Summary: "Create access policy", AuthLevel: "admin", Tags: []string{"admin", "policies", "access"}},
		{Method: "DELETE", Path: "/api/v1/admin/policies/:policy_id", Group: "admin", Summary: "Delete access policy", AuthLevel: "admin", Tags: []string{"admin", "policies"}},

		// --- Connector ---
		{Method: "GET", Path: "/api/v1/connector/config", Group: "connector", Summary: "Get configuration files", AuthLevel: "connector", Tags: []string{"connector", "config"}},
		{Method: "PUT", Path: "/api/v1/connector/config", Group: "connector", Summary: "Update configuration files", AuthLevel: "connector", Tags: []string{"connector", "config"}},
	}
}
