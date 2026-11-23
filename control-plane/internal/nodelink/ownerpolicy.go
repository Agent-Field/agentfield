package nodelink

import (
	"net/http"
	"sync"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var runOwnerClaims = promauto.NewCounterVec(prometheus.CounterOpts{
	Name: "agentfield_run_owner_claims_total",
	Help: "Number of run owner claims by proof result.",
}, []string{"result"})

const ownerWarningInterval = 10 * time.Minute

type ownerClaim struct {
	owner string
	scope string
}

// OwnerPolicy checks proofs of owner consent for HTTP owner claims.
type OwnerPolicy struct {
	Signer       *ProofSigner
	RequireProof bool
	Now          func() time.Time

	mu        sync.Mutex
	warned    map[ownerClaim]time.Time
	nextPrune time.Time
}

// Check applies the policy only to requests that name an owner. Calls made
// without a wired policy keep the legacy behavior.
func (p *OwnerPolicy) Check(owner, proof, scope, run string) error {
	if p == nil || owner == "" {
		return nil
	}
	if proof != "" {
		var err error
		if p.Signer == nil {
			err = refuse(http.StatusForbidden, "run_owner_proof_invalid: proof verification is unavailable")
		} else {
			err = p.Signer.Verify(proof, owner, scope, run)
		}
		if err != nil {
			runOwnerClaims.WithLabelValues("refused_invalid_proof").Inc()
			return err
		}
		runOwnerClaims.WithLabelValues("proven").Inc()
		return nil
	}
	if p.RequireProof {
		runOwnerClaims.WithLabelValues("refused_missing_proof").Inc()
		return refuse(http.StatusForbidden, "run_owner_proof_required: supply a proof of owner consent")
	}
	runOwnerClaims.WithLabelValues("unproven_accepted").Inc()
	if p.shouldWarn(ownerClaim{owner, scope}) {
		logger.Logger.Warn().Str("owner", owner).Str("scope", scope).Msg("accepted an unproven run owner claim")
	}
	return nil
}

func (p *OwnerPolicy) shouldWarn(claim ownerClaim) bool {
	now := time.Now()
	if p.Now != nil {
		now = p.Now()
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.warned == nil {
		p.warned = make(map[ownerClaim]time.Time)
	}
	// Keep only recent claims so arbitrary owner names do not accumulate.
	if !now.Before(p.nextPrune) {
		for key, until := range p.warned {
			if !now.Before(until) {
				delete(p.warned, key)
			}
		}
		p.nextPrune = now.Add(ownerWarningInterval)
	}
	if now.Before(p.warned[claim]) {
		return false
	}
	p.warned[claim] = now.Add(ownerWarningInterval)
	return true
}
