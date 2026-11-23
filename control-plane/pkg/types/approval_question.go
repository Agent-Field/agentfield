package types

// Approval kinds a paused execution can ask with. They tell a door how to
// show the request: a plan to read and judge, a question with choices, or a
// plain go/no-go.
const (
	ApprovalKindPlanReview = "plan_review"
	ApprovalKindQuestion   = "question"
	ApprovalKindApproval   = "approval"
)

// DefaultApprovalOptions are the option labels a request carries when it
// names none: the three decisions the control plane resolves an approval with.
var DefaultApprovalOptions = []string{"approve", "request_changes", "reject"}

// ApprovalQuestion is what one approval round asks a person: its kind, a
// one-line title, the full markdown body and the option labels. It is kept by
// the control plane so a door with no external approval service (no hax) can
// still show the whole question.
type ApprovalQuestion struct {
	ExecutionID       string   `json:"execution_id"`
	ApprovalRequestID string   `json:"approval_request_id"`
	Kind              string   `json:"kind"`
	Question          string   `json:"question,omitempty"`
	Body              string   `json:"body,omitempty"`
	Options           []string `json:"options,omitempty"`
}
