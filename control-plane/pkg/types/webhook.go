package types

import "time"

const (
	// Execution webhook lifecycle states
	ExecutionWebhookStatusPending    = "pending"
	ExecutionWebhookStatusDelivering = "delivering"
	ExecutionWebhookStatusDelivered  = "delivered"
	ExecutionWebhookStatusFailed     = "failed"

	// Execution webhook event types
	WebhookEventExecutionCompleted = "execution.completed"
	WebhookEventExecutionFailed    = "execution.failed"
)

// ExecutionWebhook captures the persisted webhook registration metadata for an execution.
type ExecutionWebhook struct {
	ExecutionID   string            `json:"execution_id" db:"execution_id"`
	URL           string            `json:"url" db:"url"`
	Secret        *string           `json:"-" db:"secret"`
	Headers       map[string]string `json:"headers,omitempty" db:"headers"`
	Status        string            `json:"status" db:"status"`
	AttemptCount  int               `json:"attempt_count" db:"attempt_count"`
	NextAttemptAt *time.Time        `json:"next_attempt_at,omitempty" db:"next_attempt_at"`
	LastAttemptAt *time.Time        `json:"last_attempt_at,omitempty" db:"last_attempt_at"`
	LastError     *string           `json:"last_error,omitempty" db:"last_error"`
	CreatedAt     time.Time         `json:"created_at" db:"created_at"`
	UpdatedAt     time.Time         `json:"updated_at" db:"updated_at"`
}

// ExecutionWebhookStateUpdate represents the mutable fields when recording delivery attempts.
type ExecutionWebhookStateUpdate struct {
	Status        string
	AttemptCount  int
	NextAttemptAt *time.Time
	LastAttemptAt *time.Time
	LastError     *string
}

// ExecutionWebhookPayload defines the shape Haxen sends to webhook consumers.
type ExecutionWebhookPayload struct {
	Event        string      `json:"event"`
	ExecutionID  string      `json:"execution_id"`
	RunID        string      `json:"workflow_id"`
	Status       string      `json:"status"`
	Target       string      `json:"target"`
	TargetType   string      `json:"type"`
	DurationMS   *int64      `json:"duration_ms,omitempty"`
	Result       interface{} `json:"result,omitempty"`
	ErrorMessage *string     `json:"error_message,omitempty"`
	Timestamp    string      `json:"timestamp"`
}

// CloneWithoutSecret returns a shallow copy of the webhook metadata without the secret.
func (w *ExecutionWebhook) CloneWithoutSecret() *ExecutionWebhook {
	if w == nil {
		return nil
	}
	headersCopy := make(map[string]string, len(w.Headers))
	for k, v := range w.Headers {
		headersCopy[k] = v
	}
	return &ExecutionWebhook{
		ExecutionID:   w.ExecutionID,
		URL:           w.URL,
		Headers:       headersCopy,
		Status:        w.Status,
		AttemptCount:  w.AttemptCount,
		NextAttemptAt: w.NextAttemptAt,
		LastAttemptAt: w.LastAttemptAt,
		LastError:     w.LastError,
		CreatedAt:     w.CreatedAt,
		UpdatedAt:     w.UpdatedAt,
	}
}
