package handlers

import (
	"net/http"
	"strconv"
	"strings"

	"github.com/Agent-Field/agentfield/control-plane/internal/nodelink"
	"github.com/gin-gonic/gin"
)

// ── THE HARNESS RELAY AND THE STEER, OVER HTTP ──────────────────────────────
//
//	POST /api/v1/executions/:id/relay       {entries:[{seq,kind,data}]}  node → pipe
//	GET  /api/v1/executions/:id/relay?after=N&limit=M                    pipe → owner
//	POST /api/v1/executions/:id/relay/ack   {through}                    owner has them
//	GET  /api/v1/relay/count[?execution_id=]                             what the pipe holds
//	POST /api/v1/executions/:id/steer       {message}                    owner → node
//
// These are the HTTP fallback of the link's entries, entries_ack and steer
// frames, and they are the same code: every one is a call on the one
// nodelink.Pipe (nodelink/pipe.go says why the transcript only passes
// through). A caller that names itself with X-AgentField-Run-Owner is held to
// the runs it owns, exactly as a link is held to its own node's runs.

// RunOwnerHeader names the node a call speaks for: on an execute it makes
// that node the run's owner; on a relay read, ack or steer it scopes the
// call to the runs that node owns.
const RunOwnerHeader = "X-AgentField-Run-Owner"

func pipeRefusal(c *gin.Context, err error) {
	c.JSON(nodelink.CodeOf(err), gin.H{"error": err.Error()})
}

// RelayPutHandler is POST /executions/:id/relay.
func RelayPutHandler(pipe *nodelink.Pipe) gin.HandlerFunc {
	return func(c *gin.Context) {
		var body nodelink.EntriesUp
		if err := c.ShouldBindJSON(&body); err != nil || len(body.Entries) == 0 {
			c.JSON(http.StatusBadRequest, gin.H{"error": "entries: [{seq, kind, data}] is required"})
			return
		}
		stored, through, err := pipe.Put(c.Request.Context(), c.Param("execution_id"), body.Entries)
		if err != nil {
			pipeRefusal(c, err)
			return
		}
		c.JSON(http.StatusOK, gin.H{"stored": stored, "through": through})
	}
}

// RelayReadHandler is GET /executions/:id/relay?after=N.
func RelayReadHandler(pipe *nodelink.Pipe) gin.HandlerFunc {
	return func(c *gin.Context) {
		after, _ := strconv.ParseInt(c.DefaultQuery("after", "0"), 10, 64)
		limit, _ := strconv.Atoi(c.DefaultQuery("limit", "500"))
		entries, err := pipe.Read(c.Request.Context(), strings.TrimSpace(c.GetHeader(RunOwnerHeader)), c.Param("execution_id"), after, limit)
		if err != nil {
			pipeRefusal(c, err)
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
func RelayAckHandler(pipe *nodelink.Pipe) gin.HandlerFunc {
	return func(c *gin.Context) {
		var body struct {
			Through int64 `json:"through"`
		}
		_ = c.ShouldBindJSON(&body)
		dropped, err := pipe.Ack(c.Request.Context(), strings.TrimSpace(c.GetHeader(RunOwnerHeader)), c.Param("execution_id"), body.Through)
		if err != nil {
			pipeRefusal(c, err)
			return
		}
		c.JSON(http.StatusOK, gin.H{"dropped": dropped})
	}
}

// RelayCountHandler is GET /relay/count.
func RelayCountHandler(pipe *nodelink.Pipe) gin.HandlerFunc {
	return func(c *gin.Context) {
		n, err := pipe.Store.RelayCount(c.Request.Context(), c.Query("execution_id"))
		if err != nil {
			pipeRefusal(c, err)
			return
		}
		c.JSON(http.StatusOK, gin.H{"held": n})
	}
}

// SteerExecutionHandler is POST /executions/:id/steer {message}: 200
// {ok, via} once the node took the note; 404 no such execution; 409 with
// person words when the step ended, the node refused, or the node is gone.
func SteerExecutionHandler(pipe *nodelink.Pipe) gin.HandlerFunc {
	return func(c *gin.Context) {
		var body struct {
			Message string `json:"message"`
		}
		_ = c.ShouldBindJSON(&body)
		via, err := pipe.Steer(c.Request.Context(), strings.TrimSpace(c.GetHeader(RunOwnerHeader)), c.Param("execution_id"), body.Message)
		if err != nil {
			pipeRefusal(c, err)
			return
		}
		c.JSON(http.StatusOK, gin.H{"ok": true, "via": via})
	}
}
