package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/agenturl"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
)

// ── THE HARNESS RELAY AND THE STEER, ON THE CONTROL PLANE ───────────────────
//
//	POST /api/v1/executions/:id/relay       {entries:[{seq,kind,data}]}  node → pipe
//	GET  /api/v1/executions/:id/relay?after=N&limit=M                    pipe → WorkAF
//	POST /api/v1/executions/:id/relay/ack   {through}                    WorkAF has them
//	GET  /api/v1/relay/count[?execution_id=]                             what the pipe holds
//	POST /api/v1/executions/:id/steer       {message}                    WorkAF → node
//
// storage/relay.go says why the transcript only passes through. The steer is
// the request the control plane sends the execution's node at
// /_internal/executions/:id/steer through agenturl's client, so it travels
// down the node's link when the node has no address, and over HTTP when it
// has one. It is answered only when the node answered: 200 means the note is
// in a running codeaf harness's mailbox, 409 carries the node's refusal (it is
// not running the codeaf harness) or the control plane's (the step ended).

// RelayStore is the relay's storage, the local store's (storage/relay.go).
type RelayStore interface {
	RelayPut(ctx context.Context, executionID string, entries []storage.RelayEntry) (int, error)
	RelayAfter(ctx context.Context, executionID string, after int64, limit int) ([]storage.RelayEntry, error)
	RelayAck(ctx context.Context, executionID string, through int64) (int64, error)
	RelayPrune(ctx context.Context, olderThan time.Time) error
	RelayCount(ctx context.Context, executionID string) (int64, error)
}

func relayOf(store any) (RelayStore, bool) {
	r, ok := store.(RelayStore)
	return r, ok
}

// RelayPutHandler is POST /executions/:id/relay.
func RelayPutHandler(store any, retention func() time.Duration) gin.HandlerFunc {
	return func(c *gin.Context) {
		r, ok := relayOf(store)
		if !ok {
			c.JSON(http.StatusNotImplemented, gin.H{"error": "this storage has no harness relay"})
			return
		}
		var body struct {
			Entries []storage.RelayEntry `json:"entries"`
		}
		if err := c.ShouldBindJSON(&body); err != nil || len(body.Entries) == 0 {
			c.JSON(http.StatusBadRequest, gin.H{"error": "entries: [{seq, kind, data}] is required"})
			return
		}
		var through int64
		for _, e := range body.Entries {
			if e.Seq <= 0 || strings.TrimSpace(e.Kind) == "" {
				c.JSON(http.StatusBadRequest, gin.H{"error": "every entry has a seq from 1 and a kind"})
				return
			}
			if e.Seq > through {
				through = e.Seq
			}
		}
		stored, err := r.RelayPut(c.Request.Context(), c.Param("execution_id"), body.Entries)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		if d := retention(); d > 0 {
			_ = r.RelayPrune(c.Request.Context(), time.Now().Add(-d))
		}
		c.JSON(http.StatusOK, gin.H{"stored": stored, "through": through})
	}
}

// RelayReadHandler is GET /executions/:id/relay?after=N.
func RelayReadHandler(store any) gin.HandlerFunc {
	return func(c *gin.Context) {
		r, ok := relayOf(store)
		if !ok {
			c.JSON(http.StatusNotImplemented, gin.H{"error": "this storage has no harness relay"})
			return
		}
		after, _ := strconv.ParseInt(c.DefaultQuery("after", "0"), 10, 64)
		limit, _ := strconv.Atoi(c.DefaultQuery("limit", "500"))
		entries, err := r.RelayAfter(c.Request.Context(), c.Param("execution_id"), after, limit)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		next := after
		if len(entries) > 0 {
			next = entries[len(entries)-1].Seq
		}
		c.JSON(http.StatusOK, gin.H{"entries": entries, "next": next})
	}
}

// RelayAckHandler is POST /executions/:id/relay/ack {through}.
func RelayAckHandler(store any) gin.HandlerFunc {
	return func(c *gin.Context) {
		r, ok := relayOf(store)
		if !ok {
			c.JSON(http.StatusNotImplemented, gin.H{"error": "this storage has no harness relay"})
			return
		}
		var body struct {
			Through int64 `json:"through"`
		}
		if err := c.ShouldBindJSON(&body); err != nil || body.Through <= 0 {
			c.JSON(http.StatusBadRequest, gin.H{"error": "through: the last seq kept is required"})
			return
		}
		dropped, err := r.RelayAck(c.Request.Context(), c.Param("execution_id"), body.Through)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, gin.H{"dropped": dropped})
	}
}

// RelayCountHandler is GET /relay/count.
func RelayCountHandler(store any) gin.HandlerFunc {
	return func(c *gin.Context) {
		r, ok := relayOf(store)
		if !ok {
			c.JSON(http.StatusNotImplemented, gin.H{"error": "this storage has no harness relay"})
			return
		}
		n, err := r.RelayCount(c.Request.Context(), c.Query("execution_id"))
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, gin.H{"held": n})
	}
}

// SteerStore is what the steer reads.
type SteerStore interface {
	GetExecutionRecord(ctx context.Context, executionID string) (*types.Execution, error)
	GetAgent(ctx context.Context, id string) (*types.AgentNode, error)
}

// SteerClient sends the steer to the node; tests replace it.
var SteerClient = func() *http.Client { return agenturl.NewClient(20 * time.Second) }

// SteerExecutionHandler is POST /executions/:id/steer {message}.
func SteerExecutionHandler(store SteerStore, internalToken string) gin.HandlerFunc {
	return func(c *gin.Context) {
		id := c.Param("execution_id")
		var body struct {
			Message string `json:"message"`
		}
		if err := c.ShouldBindJSON(&body); err != nil || strings.TrimSpace(body.Message) == "" || len(body.Message) > 4000 {
			c.JSON(http.StatusBadRequest, gin.H{"error": "message is 1 to 4000 bytes"})
			return
		}
		ctx := c.Request.Context()
		exec, err := store.GetExecutionRecord(ctx, id)
		if err != nil || exec == nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "no such execution"})
			return
		}
		if types.IsTerminalExecutionStatus(string(exec.Status)) {
			c.JSON(http.StatusConflict, gin.H{"error": "this step has ended"})
			return
		}
		agent, err := store.GetAgent(ctx, exec.AgentNodeID)
		if err != nil || agent == nil {
			c.JSON(http.StatusConflict, gin.H{"error": "the node that runs this step is not registered"})
			return
		}
		target, err := agenturl.ForNode(agent.ID, agent.BaseURL, "_internal", "executions", id, "steer")
		if err != nil {
			c.JSON(http.StatusConflict, gin.H{"error": "the node has no address and no link"})
			return
		}
		raw, _ := json.Marshal(map[string]string{"message": strings.TrimSpace(body.Message)})
		req, _ := http.NewRequestWithContext(ctx, http.MethodPost, target.String(), bytes.NewReader(raw))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Execution-ID", id)
		if internalToken != "" {
			req.Header.Set("Authorization", "Bearer "+internalToken)
		}
		via := "http"
		if _, linked := agenturl.LinkNode(target.Host); linked {
			via = "link"
		}
		resp, err := SteerClient().Do(req)
		if err != nil {
			c.JSON(http.StatusConflict, gin.H{"error": "the node did not take the note: " + err.Error()})
			return
		}
		defer resp.Body.Close()
		answer, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		if resp.StatusCode >= 300 {
			var refusal struct {
				Error string `json:"error"`
			}
			_ = json.Unmarshal(answer, &refusal)
			if refusal.Error == "" {
				refusal.Error = strings.TrimSpace(string(answer))
			}
			c.JSON(http.StatusConflict, gin.H{"error": refusal.Error})
			return
		}
		c.JSON(http.StatusOK, gin.H{"ok": true, "via": via})
	}
}
