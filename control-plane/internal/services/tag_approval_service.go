package services

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/google/uuid"
)

// ErrNotPendingApproval indicates the agent is not in pending_approval status.
var ErrNotPendingApproval = errors.New("agent is not pending approval")

// TagApprovalResult holds the outcome of evaluating proposed tags against approval rules.
type TagApprovalResult struct {
	AutoApproved    []string
	ManualReview    []string
	Forbidden       []string
	AllAutoApproved bool
}

// TagApprovalStorage defines storage operations needed by the tag approval service.
type TagApprovalStorage interface {
	GetAgent(ctx context.Context, id string) (*types.AgentNode, error)
	RegisterAgent(ctx context.Context, node *types.AgentNode) error
	ListAgentsByLifecycleStatus(ctx context.Context, status types.AgentLifecycleStatus) ([]*types.AgentNode, error)
	GetAgentDID(ctx context.Context, agentID string) (*types.AgentDIDInfo, error)
	StoreAgentTagVC(ctx context.Context, agentID, agentDID, vcID, vcDocument, signature string, issuedAt time.Time, expiresAt *time.Time) error
	RevokeAgentTagVC(ctx context.Context, agentID string) error
}

// TagApprovalVCService defines the VC signing operations needed by the tag approval service.
type TagApprovalVCService interface {
	GetDIDService() *DIDService
	SignAgentTagVC(vc *types.AgentTagVCDocument) (*types.VCProof, error)
}

// TagApprovalService evaluates proposed tags against approval rules and manages
// the tag approval workflow for agents.
type TagApprovalService struct {
	config    config.TagApprovalRulesConfig
	storage   TagApprovalStorage
	vcService TagApprovalVCService // optional, can be nil
	onRevoke  func(ctx context.Context, agentID string)
	mu        sync.RWMutex
}

// NewTagApprovalService creates a new tag approval service.
func NewTagApprovalService(cfg config.TagApprovalRulesConfig, storage TagApprovalStorage) *TagApprovalService {
	defaultMode := cfg.DefaultMode
	if defaultMode == "" {
		defaultMode = "auto"
	}
	cfg.DefaultMode = defaultMode

	return &TagApprovalService{
		config:  cfg,
		storage: storage,
	}
}

// SetOnRevokeCallback sets a callback invoked after tags are revoked,
// used to clear status caches and presence leases.
// Must be called during initialization before any concurrent use.
func (s *TagApprovalService) SetOnRevokeCallback(fn func(ctx context.Context, agentID string)) {
	s.onRevoke = fn
}

// SetVCService sets the VC service for tag VC issuance (optional dependency).
// Must be called during initialization before any concurrent use.
func (s *TagApprovalService) SetVCService(vcService TagApprovalVCService) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.vcService = vcService
}

// IsEnabled returns true if any tag approval rules require non-auto behavior.
func (s *TagApprovalService) IsEnabled() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.config.Rules) > 0 || s.config.DefaultMode != "auto"
}

// EvaluateTags evaluates a set of proposed tags against the configured approval rules.
func (s *TagApprovalService) EvaluateTags(proposedTags []string) TagApprovalResult {
	s.mu.RLock()
	defer s.mu.RUnlock()

	result := TagApprovalResult{}

	for _, tag := range proposedTags {
		normalized := strings.ToLower(strings.TrimSpace(tag))
		if normalized == "" {
			continue
		}

		mode := s.getTagApprovalMode(normalized)
		switch mode {
		case "auto":
			result.AutoApproved = append(result.AutoApproved, normalized)
		case "manual":
			result.ManualReview = append(result.ManualReview, normalized)
		case "forbidden":
			result.Forbidden = append(result.Forbidden, normalized)
		default:
			// Unknown mode treated as manual for safety
			result.ManualReview = append(result.ManualReview, normalized)
		}
	}

	result.AllAutoApproved = len(result.ManualReview) == 0 && len(result.Forbidden) == 0
	return result
}

// getTagApprovalMode returns the approval mode for a specific tag.
func (s *TagApprovalService) getTagApprovalMode(tag string) string {
	for _, rule := range s.config.Rules {
		for _, ruleTag := range rule.Tags {
			normalized := strings.ToLower(strings.TrimSpace(ruleTag))
			if normalized == tag {
				return rule.Approval
			}
		}
	}
	return s.config.DefaultMode
}

// CollectAllProposedTags extracts all proposed tags from an agent's reasoners, skills,
// and agent-level ProposedTags field.
func CollectAllProposedTags(agent *types.AgentNode) []string {
	seen := make(map[string]struct{})
	var tags []string

	add := func(tag string) {
		normalized := strings.ToLower(strings.TrimSpace(tag))
		if normalized == "" {
			return
		}
		if _, exists := seen[normalized]; exists {
			return
		}
		seen[normalized] = struct{}{}
		tags = append(tags, normalized)
	}

	// Collect agent-level proposed tags (sent by SDKs as top-level proposed_tags).
	for _, t := range agent.ProposedTags {
		add(t)
	}

	for _, r := range agent.Reasoners {
		proposed := r.ProposedTags
		if len(proposed) == 0 {
			proposed = r.Tags
		}
		for _, t := range proposed {
			add(t)
		}
	}

	for _, sk := range agent.Skills {
		proposed := sk.ProposedTags
		if len(proposed) == 0 {
			proposed = sk.Tags
		}
		for _, t := range proposed {
			add(t)
		}
	}

	return tags
}

// ApproveAgentTags approves an agent's tags, setting approved_tags and transitioning
// the lifecycle status from pending_approval to starting.
func (s *TagApprovalService) ApproveAgentTags(ctx context.Context, agentID string, approvedTags []string, approvedBy string) error {
	agent, err := s.storage.GetAgent(ctx, agentID)
	if err != nil {
		return err
	}

	if agent.LifecycleStatus != types.AgentStatusPendingApproval {
		return fmt.Errorf("%w: agent %s (current status: %s)", ErrNotPendingApproval, agentID, agent.LifecycleStatus)
	}

	agent.ApprovedTags = approvedTags
	agent.LifecycleStatus = types.AgentStatusStarting

	// Set approved tags on each reasoner and skill
	approvedSet := make(map[string]struct{})
	for _, t := range approvedTags {
		approvedSet[strings.ToLower(strings.TrimSpace(t))] = struct{}{}
	}

	for i := range agent.Reasoners {
		var approved []string
		proposed := agent.Reasoners[i].ProposedTags
		if len(proposed) == 0 {
			proposed = agent.Reasoners[i].Tags
		}
		for _, t := range proposed {
			if _, ok := approvedSet[strings.ToLower(strings.TrimSpace(t))]; ok {
				approved = append(approved, t)
			}
		}
		agent.Reasoners[i].ApprovedTags = approved
	}

	for i := range agent.Skills {
		var approved []string
		proposed := agent.Skills[i].ProposedTags
		if len(proposed) == 0 {
			proposed = agent.Skills[i].Tags
		}
		for _, t := range proposed {
			if _, ok := approvedSet[strings.ToLower(strings.TrimSpace(t))]; ok {
				approved = append(approved, t)
			}
		}
		agent.Skills[i].ApprovedTags = approved
	}

	if err := s.storage.RegisterAgent(ctx, agent); err != nil {
		return err
	}

	logger.Logger.Info().
		Str("agent_id", agentID).
		Strs("approved_tags", approvedTags).
		Str("approved_by", approvedBy).
		Msg("Agent tags approved")

	// Issue a signed Agent Tag VC (non-fatal on failure)
	s.issueTagVC(ctx, agentID, approvedTags, approvedBy)

	return nil
}

// ApproveAgentTagsPerSkill approves tags at per-skill/per-reasoner granularity.
func (s *TagApprovalService) ApproveAgentTagsPerSkill(ctx context.Context, agentID string, skillTags map[string][]string, reasonerTags map[string][]string, approvedBy string) error {
	agent, err := s.storage.GetAgent(ctx, agentID)
	if err != nil {
		return err
	}

	if agent.LifecycleStatus != types.AgentStatusPendingApproval {
		return fmt.Errorf("%w: agent %s (current status: %s)", ErrNotPendingApproval, agentID, agent.LifecycleStatus)
	}

	for i := range agent.Reasoners {
		if tags, ok := reasonerTags[agent.Reasoners[i].ID]; ok {
			agent.Reasoners[i].ApprovedTags = tags
		}
	}

	for i := range agent.Skills {
		if tags, ok := skillTags[agent.Skills[i].ID]; ok {
			agent.Skills[i].ApprovedTags = tags
		}
	}

	// Collect all approved tags for the agent-level field
	seen := make(map[string]struct{})
	var allApproved []string
	for _, r := range agent.Reasoners {
		for _, t := range r.ApprovedTags {
			normalized := strings.ToLower(strings.TrimSpace(t))
			if _, exists := seen[normalized]; !exists {
				seen[normalized] = struct{}{}
				allApproved = append(allApproved, normalized)
			}
		}
	}
	for _, sk := range agent.Skills {
		for _, t := range sk.ApprovedTags {
			normalized := strings.ToLower(strings.TrimSpace(t))
			if _, exists := seen[normalized]; !exists {
				seen[normalized] = struct{}{}
				allApproved = append(allApproved, normalized)
			}
		}
	}

	agent.ApprovedTags = allApproved
	agent.LifecycleStatus = types.AgentStatusStarting

	if err := s.storage.RegisterAgent(ctx, agent); err != nil {
		return err
	}

	logger.Logger.Info().
		Str("agent_id", agentID).
		Str("approved_by", approvedBy).
		Msg("Agent tags approved (per-skill)")

	// Issue a signed Agent Tag VC (non-fatal on failure)
	s.issueTagVC(ctx, agentID, allApproved, approvedBy)

	return nil
}

// RejectAgentTags rejects an agent's proposed tags.
func (s *TagApprovalService) RejectAgentTags(ctx context.Context, agentID string, rejectedBy string, reason string) error {
	agent, err := s.storage.GetAgent(ctx, agentID)
	if err != nil {
		return err
	}

	if agent.LifecycleStatus != types.AgentStatusPendingApproval {
		return fmt.Errorf("%w: agent %s (current status: %s)", ErrNotPendingApproval, agentID, agent.LifecycleStatus)
	}

	agent.LifecycleStatus = types.AgentStatusOffline
	agent.ApprovedTags = nil

	// Clear approved tags on all skills/reasoners
	for i := range agent.Reasoners {
		agent.Reasoners[i].ApprovedTags = nil
	}
	for i := range agent.Skills {
		agent.Skills[i].ApprovedTags = nil
	}

	if err := s.storage.RegisterAgent(ctx, agent); err != nil {
		return err
	}

	logger.Logger.Info().
		Str("agent_id", agentID).
		Str("rejected_by", rejectedBy).
		Str("reason", reason).
		Msg("Agent tags rejected")

	return nil
}

// RevokeAgentTags revokes an agent's approved tags, transitions it back to
// pending_approval, and revokes its tag VC. Works on agents in any lifecycle status.
func (s *TagApprovalService) RevokeAgentTags(ctx context.Context, agentID string, revokedBy string, reason string) error {
	agent, err := s.storage.GetAgent(ctx, agentID)
	if err != nil {
		return err
	}

	// Check if agent tags are already revoked (no approved tags and already pending)
	if agent.LifecycleStatus == types.AgentStatusPendingApproval && len(agent.ApprovedTags) == 0 {
		return fmt.Errorf("already_revoked: agent tags are already revoked")
	}

	// Revoke the agent tag VC (non-fatal if no VC exists)
	if err := s.storage.RevokeAgentTagVC(ctx, agentID); err != nil {
		logger.Logger.Warn().Err(err).Str("agent_id", agentID).Msg("Failed to revoke agent tag VC (may not exist)")
	}

	// Clear approved tags
	agent.ApprovedTags = nil
	for i := range agent.Reasoners {
		agent.Reasoners[i].ApprovedTags = nil
	}
	for i := range agent.Skills {
		agent.Skills[i].ApprovedTags = nil
	}
	agent.LifecycleStatus = types.AgentStatusPendingApproval

	if err := s.storage.RegisterAgent(ctx, agent); err != nil {
		return err
	}

	logger.Logger.Info().
		Str("agent_id", agentID).
		Str("revoked_by", revokedBy).
		Str("reason", reason).
		Msg("Agent tags revoked")

	if s.onRevoke != nil {
		s.onRevoke(ctx, agentID)
	}

	return nil
}

// ListPendingAgents returns all agents currently in pending_approval status.
func (s *TagApprovalService) ListPendingAgents(ctx context.Context) ([]*types.AgentNode, error) {
	return s.storage.ListAgentsByLifecycleStatus(ctx, types.AgentStatusPendingApproval)
}

// ProcessRegistrationTags evaluates tags at registration time and returns the result.
// The caller should use this to decide whether to set the agent to pending or auto-approve.
//
// If the agent already carries ApprovedTags (preserved from a previous registration),
// and the proposed tags haven't changed, the existing approval state is kept intact.
// This prevents forcing re-approval after every CP restart or agent reconnect.
func (s *TagApprovalService) ProcessRegistrationTags(agent *types.AgentNode) TagApprovalResult {
	allProposed := CollectAllProposedTags(agent)
	agent.ProposedTags = allProposed

	result := s.EvaluateTags(allProposed)

	// If the agent already has approved tags (re-registration), check whether the
	// proposed tags are still covered by the existing approval. If so, keep the
	// current approval state and don't force the agent back to pending_approval.
	if len(agent.ApprovedTags) > 0 {
		existingApproved := make(map[string]struct{})
		for _, t := range agent.ApprovedTags {
			existingApproved[strings.ToLower(strings.TrimSpace(t))] = struct{}{}
		}

		// Check that every manual-review tag is already approved.
		allCovered := true
		for _, t := range result.ManualReview {
			if _, ok := existingApproved[strings.ToLower(strings.TrimSpace(t))]; !ok {
				allCovered = false
				break
			}
		}

		if allCovered {
			// Existing approval still covers all proposed tags — keep it.
			// Don't touch agent.ApprovedTags or agent.LifecycleStatus.
			return result
		}

		// Some new tags need approval. Only require approval for the new ones;
		// keep previously-approved tags in the approved set.
		for _, t := range result.AutoApproved {
			existingApproved[strings.ToLower(strings.TrimSpace(t))] = struct{}{}
		}
		merged := make([]string, 0, len(existingApproved))
		for t := range existingApproved {
			merged = append(merged, t)
		}
		agent.ApprovedTags = merged
		agent.LifecycleStatus = types.AgentStatusPendingApproval
		return result
	}

	if result.AllAutoApproved {
		// Auto-approve: set approved tags immediately
		agent.ApprovedTags = result.AutoApproved
		for i := range agent.Reasoners {
			agent.Reasoners[i].ApprovedTags = agent.Reasoners[i].Tags
			if len(agent.Reasoners[i].ApprovedTags) == 0 {
				agent.Reasoners[i].ApprovedTags = agent.Reasoners[i].ProposedTags
			}
		}
		for i := range agent.Skills {
			agent.Skills[i].ApprovedTags = agent.Skills[i].Tags
			if len(agent.Skills[i].ApprovedTags) == 0 {
				agent.Skills[i].ApprovedTags = agent.Skills[i].ProposedTags
			}
		}
	} else {
		// Needs approval: only auto-approved tags are set
		agent.ApprovedTags = result.AutoApproved
		agent.LifecycleStatus = types.AgentStatusPendingApproval
	}

	return result
}

// IssueAutoApprovedTagsVC issues a Tag VC for auto-approved agents during registration.
// This must be called AFTER the agent is stored and DID is registered.
func (s *TagApprovalService) IssueAutoApprovedTagsVC(ctx context.Context, agentID string, approvedTags []string) {
	s.issueTagVC(ctx, agentID, approvedTags, "system:auto-approved")
}

// issueTagVC creates and stores a signed Agent Tag VC for an agent.
// This is non-fatal — if VC issuance fails, the tag approval still succeeds.
func (s *TagApprovalService) issueTagVC(ctx context.Context, agentID string, approvedTags []string, approvedBy string) {
	s.mu.RLock()
	vcSvc := s.vcService
	s.mu.RUnlock()
	if vcSvc == nil {
		return
	}

	// Get agent's DID
	agentDIDInfo, err := s.storage.GetAgentDID(ctx, agentID)
	if err != nil {
		logger.Logger.Warn().Err(err).Str("agent_id", agentID).Msg("Cannot issue tag VC: agent DID not found")
		return
	}

	// Get control plane issuer DID
	var issuerDID string
	didService := vcSvc.GetDIDService()
	if didService != nil {
		if rootDID, err := didService.GetControlPlaneIssuerDID(); err == nil {
			issuerDID = rootDID
		}
	}
	if issuerDID == "" {
		logger.Logger.Warn().Str("agent_id", agentID).Msg("Cannot issue tag VC: no issuer DID available")
		return
	}

	// Build the VC document
	now := time.Now()
	vcID := fmt.Sprintf("urn:agentfield:agent-tag-vc:%s", uuid.New().String())

	vc := &types.AgentTagVCDocument{
		Context: []string{
			"https://www.w3.org/2018/credentials/v1",
		},
		Type: []string{
			"VerifiableCredential",
			"AgentTagCredential",
		},
		ID:           vcID,
		Issuer:       issuerDID,
		IssuanceDate: now.Format(time.RFC3339),
		CredentialSubject: types.AgentTagVCCredentialSubject{
			ID:      agentDIDInfo.DID,
			AgentID: agentID,
			Permissions: types.AgentTagVCPermissions{
				Tags:           approvedTags,
				AllowedCallees: []string{"*"},
			},
			ApprovedBy: approvedBy,
			ApprovedAt: now.Format(time.RFC3339),
		},
	}

	// Sign the VC
	proof, err := vcSvc.SignAgentTagVC(vc)
	if err != nil {
		logger.Logger.Warn().Err(err).Str("agent_id", agentID).Msg("Failed to sign agent tag VC")
		return
	}
	vc.Proof = proof

	// Serialize the VC document
	vcDocJSON, err := json.Marshal(vc)
	if err != nil {
		logger.Logger.Warn().Err(err).Str("agent_id", agentID).Msg("Failed to marshal agent tag VC")
		return
	}

	// Extract signature value for storage
	signature := ""
	if proof != nil {
		signature = proof.ProofValue
	}

	// Store the VC
	if err := s.storage.StoreAgentTagVC(ctx, agentID, agentDIDInfo.DID, vcID, string(vcDocJSON), signature, now, nil); err != nil {
		logger.Logger.Warn().Err(err).Str("agent_id", agentID).Msg("Failed to store agent tag VC")
		return
	}

	proofType := "none"
	if proof != nil {
		proofType = proof.Type
	}
	logger.Logger.Info().
		Str("agent_id", agentID).
		Str("vc_id", vcID).
		Str("proof_type", proofType).
		Msg("Agent tag VC issued")
}

