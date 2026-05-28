package types

import "time"

// ObservabilityWebhookConfig represents the global observability webhook configuration.
// Only one configuration exists (singleton with id="global").
type ObservabilityWebhookConfig struct {
	ID        string            `json:"id" db:"id"`
	URL       string            `json:"url" db:"url"`
	Secret    *string           `json:"-" db:"secret"` // Hidden from JSON responses
	HasSecret bool              `json:"has_secret"`    // Indicates if a secret is configured
	Headers   map[string]string `json:"headers,omitempty" db:"headers"`
	Enabled   bool              `json:"enabled" db:"enabled"`
	CreatedAt time.Time         `json:"created_at" db:"created_at"`
	UpdatedAt time.Time         `json:"updated_at" db:"updated_at"`
}

// ObservabilityWebhookConfigRequest is the API request for creating/updating webhook config.
type ObservabilityWebhookConfigRequest struct {
	URL     string            `json:"url" binding:"required,url"`
	Secret  *string           `json:"secret,omitempty"`
	Headers map[string]string `json:"headers,omitempty"`
	Enabled *bool             `json:"enabled,omitempty"` // Defaults to true if not specified
}

// ObservabilityWebhookConfigResponse is the API response for webhook config.
type ObservabilityWebhookConfigResponse struct {
	Configured bool                        `json:"configured"`
	Config     *ObservabilityWebhookConfig `json:"config,omitempty"`
}

// ObservabilityEvent is the normalized envelope for all events sent to the webhook.
type ObservabilityEvent struct {
	EventType   string      `json:"event_type"`   // e.g., "execution.completed", "node.online"
	EventSource string      `json:"event_source"` // "execution", "node", "reasoner"
	Timestamp   string      `json:"timestamp"`    // RFC3339
	Data        interface{} `json:"data"`         // Event-specific payload
}

// ObservabilityEventBatch groups multiple events for batch delivery.
type ObservabilityEventBatch struct {
	BatchID    string               `json:"batch_id"`
	EventCount int                  `json:"event_count"`
	Events     []ObservabilityEvent `json:"events"`
	Timestamp  string               `json:"timestamp"` // RFC3339
}

// ObservabilityForwarderStatus provides current forwarder state for the status endpoint.
type ObservabilityForwarderStatus struct {
	Enabled          bool       `json:"enabled"`
	WebhookURL       string     `json:"webhook_url,omitempty"`
	QueueDepth       int        `json:"queue_depth"`
	EventsForwarded  int64      `json:"events_forwarded"`
	EventsDropped    int64      `json:"events_dropped"`
	DeadLetterCount  int64      `json:"dead_letter_count"`
	LastForwardedAt  *time.Time `json:"last_forwarded_at,omitempty"`
	LastError        *string    `json:"last_error,omitempty"`
}

// ObservabilityDeadLetterEntry represents an event that failed to deliver.
type ObservabilityDeadLetterEntry struct {
	ID             int64     `json:"id" db:"id"`
	EventType      string    `json:"event_type" db:"event_type"`
	EventSource    string    `json:"event_source" db:"event_source"`
	EventTimestamp time.Time `json:"event_timestamp" db:"event_timestamp"`
	Payload        string    `json:"payload" db:"payload"` // JSON string
	ErrorMessage   string    `json:"error_message" db:"error_message"`
	RetryCount     int       `json:"retry_count" db:"retry_count"`
	CreatedAt      time.Time `json:"created_at" db:"created_at"`
}

// ObservabilityDeadLetterListResponse is the response for listing DLQ entries.
type ObservabilityDeadLetterListResponse struct {
	Entries    []ObservabilityDeadLetterEntry `json:"entries"`
	TotalCount int64                          `json:"total_count"`
}

// ObservabilityRedriveResponse is the response for the redrive operation.
type ObservabilityRedriveResponse struct {
	Success   bool   `json:"success"`
	Message   string `json:"message"`
	Processed int    `json:"processed"`
	Failed    int    `json:"failed"`
}
