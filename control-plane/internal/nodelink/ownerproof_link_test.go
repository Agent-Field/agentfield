package nodelink

import (
	"encoding/json"
	"net/http"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

func TestDelegateLinkContractA1A2A12A13(t *testing.T) {
	t.Run("proof and link actions in strict mode", func(t *testing.T) {
		r := newRig(t, t.TempDir())
		signer := NewProofSigner([]byte("example signing key"))
		r.pipe.Owners = &OwnerPolicy{Signer: signer, RequireProof: true}
		require.NoError(t, r.store.RegisterAgent(t.Context(), &types.AgentNode{ID: "my-app", Version: "1", LastHeartbeat: time.Now()}))
		app := r.dial("my-app", nil, nil)
		worker := r.dial("steerbox", []string{KindSteer}, func(_ Frame, d DeliverBody) *Ack {
			if d.Message == "check this" {
				return &Ack{Outcome: OutcomeAccepted}
			}
			return &Ack{Outcome: OutcomeRejected}
		})

		ok := app.call(Frame{T: TDelegate}, map[string]any{
			"scopes": []string{ScopeRead, ScopeOwn, ScopeRead}, "ttl_seconds": 1000000,
			"run": "my-run", "owner": "other",
		})
		require.Equal(t, http.StatusOK, ok.Code, ok.Error)
		require.Equal(t, 86400, ok.ExpiresIn)
		_, err := time.Parse(time.RFC3339, ok.ExpiresAt)
		require.NoError(t, err)
		require.Equal(t, []string{ScopeOwn, ScopeRead}, ok.Scopes)
		require.NoError(t, signer.Verify(ok.Token, "my-app", ScopeOwn, "my-run"))
		requireOwnerProofRefusal(t, signer.Verify(ok.Token, "other", ScopeOwn, "my-run"), "run_owner_proof_invalid")

		for _, scopes := range [][]string{nil, {"unknown"}} {
			bad := app.call(Frame{T: TDelegate}, DelegateUp{Scopes: scopes})
			require.Equal(t, http.StatusBadRequest, bad.Code)
			require.Contains(t, bad.Error, "scope")
			require.Empty(t, bad.Token)
		}

		now := time.Now().UTC()
		require.NoError(t, r.store.CreateExecutionRecord(t.Context(), &types.Execution{
			ExecutionID: "my-exec", RunID: "my-run", AgentNodeID: "steerbox", NodeID: "steerbox", ReasonerID: "intake",
			Status: types.ExecutionStatusRunning, StartedAt: now, CreatedAt: now, UpdatedAt: now,
		}))
		require.NoError(t, r.pipe.Own(t.Context(), "my-run", "my-app"))
		require.Equal(t, http.StatusOK, app.call(Frame{T: TSubscribe}, Subscribe{}).Code)
		var run RunState
		require.NoError(t, json.Unmarshal(app.next(TRun).Body, &run))
		require.Equal(t, "my-run", run.RunID)
		require.Equal(t, http.StatusOK, worker.call(Frame{T: TEntries, Exec: "my-exec"}, EntriesUp{Entries: entries(1)}).Code)
		require.Equal(t, []int64{1}, app.collect(map[string]int64{"my-exec": 1}, 0)["my-exec"])
		require.Equal(t, http.StatusOK, app.call(Frame{T: TSteer, Exec: "my-exec"}, SteerUp{Message: "check this"}).Code)
		ack := app.call(Frame{T: TEntriesAck, Exec: "my-exec"}, EntriesAck{Through: 1})
		require.Equal(t, http.StatusOK, ack.Code)
		require.EqualValues(t, 1, ack.Dropped)
	})

	for _, name := range []string{"no signer", "no pipe"} {
		t.Run(name, func(t *testing.T) {
			r := newRig(t, t.TempDir())
			if name == "no pipe" {
				r.hub.Pipe = nil
			}
			app := r.dial("other", nil, nil)
			ok := app.call(Frame{T: TDelegate}, DelegateUp{Scopes: []string{ScopeOwn}})
			require.Equal(t, http.StatusNotImplemented, ok.Code)
			require.Empty(t, ok.Token)
		})
	}
}
