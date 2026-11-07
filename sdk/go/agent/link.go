package agent

// ── THE NODE LINK, FROM THE SDK'S SIDE ──────────────────────────────────────
//
// The control plane's node link (control-plane/internal/nodelink) is one
// WebSocket a node opens and keeps open, so the control plane can reach a
// node it cannot dial. This SDK opens it after registering and serves the two
// deliveries a Go agent needs pushed to it: a person's STEER note for a codeaf
// harness run, and CANCEL. Its hello says so (accepts: steer, cancel), and the
// control plane keeps sending dispatches over HTTP to this node, exactly as
// before — a Go agent still serves /execute on its own address.
//
// It is on by default and costs nothing when the control plane has no link:
// a control plane that answers the upgrade 404 is asked again only every few
// minutes. AGENTFIELD_NODE_LINK=off turns it off.

import (
	"encoding/json"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/gorilla/websocket"
)

const linkVersion = 1

type linkFrame struct {
	V    int             `json:"v"`
	T    string          `json:"t"`
	ID   string          `json:"id,omitempty"`
	Exec string          `json:"exec,omitempty"`
	Seq  int64           `json:"seq,omitempty"`
	TS   string          `json:"ts,omitempty"`
	Body json.RawMessage `json:"body,omitempty"`
}

type linkDeliver struct {
	Kind    string `json:"kind"`
	Message string `json:"message,omitempty"`
	Reason  string `json:"reason,omitempty"`
}

func linkEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("AGENTFIELD_NODE_LINK"))) {
	case "0", "off", "false", "no":
		return false
	}
	return true
}

// linkURL is the control plane's link address for this node.
func (a *Agent) linkURL() (string, bool) {
	base, err := url.Parse(strings.TrimSpace(a.cfg.AgentFieldURL))
	if err != nil || base.Host == "" {
		return "", false
	}
	switch base.Scheme {
	case "https":
		base.Scheme = "wss"
	default:
		base.Scheme = "ws"
	}
	base.Path = strings.TrimRight(base.Path, "/") + "/api/v1/nodes/" + url.PathEscape(a.cfg.NodeID) + "/link"
	return base.String(), true
}

func (a *Agent) startLink() {
	if !linkEnabled() || a.cfg.NodeID == "" {
		return
	}
	target, ok := a.linkURL()
	if !ok {
		return
	}
	go a.linkLoop(target)
}

func (a *Agent) linkLoop(target string) {
	backoff := time.Second
	for {
		up, status := a.linkOnce(target)
		if up {
			backoff = time.Second
		}
		wait := backoff
		if status == http.StatusNotFound || status == http.StatusUnauthorized || status == http.StatusForbidden {
			// No link on this control plane (or not for us): ask again rarely.
			wait = 5 * time.Minute
		}
		select {
		case <-a.stopLease:
			return
		case <-time.After(wait):
		}
		if backoff < 30*time.Second {
			backoff *= 2
		}
	}
}

// linkOnce holds one link until it drops. It says whether the link came up,
// and the HTTP status a refused upgrade answered.
func (a *Agent) linkOnce(target string) (bool, int) {
	header := http.Header{}
	if a.cfg.APIKey != "" {
		header.Set("X-API-Key", a.cfg.APIKey)
	} else if a.cfg.Token != "" {
		header.Set("Authorization", "Bearer "+a.cfg.Token)
	}
	header.Set("X-Agent-Instance-ID", a.cfg.NodeID)
	dialer := websocket.Dialer{HandshakeTimeout: 10 * time.Second}
	conn, resp, err := dialer.Dial(target, header)
	if err != nil {
		if resp != nil {
			return false, resp.StatusCode
		}
		return false, 0
	}
	defer conn.Close()
	stop := make(chan struct{})
	defer close(stop)
	go func() {
		select {
		case <-a.stopLease:
			_ = conn.Close()
		case <-stop:
		}
	}()
	send := func(f linkFrame, body any) error {
		f.V = linkVersion
		f.TS = time.Now().UTC().Format(time.RFC3339Nano)
		if body != nil {
			raw, _ := json.Marshal(body)
			f.Body = raw
		}
		raw, _ := json.Marshal(f)
		_ = conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
		return conn.WriteMessage(websocket.TextMessage, raw)
	}
	if err := send(linkFrame{T: "hello", Seq: 1}, map[string]any{"instance_id": a.cfg.NodeID, "sdk": "go", "version": a.cfg.Version, "accepts": []string{"steer", "cancel"}}); err != nil {
		return false, 0
	}
	a.logger.Printf("node link up (%s)", target)
	for {
		_, raw, err := conn.ReadMessage()
		if err != nil {
			a.logger.Printf("node link down: %v", err)
			return true, 0
		}
		var f linkFrame
		if json.Unmarshal(raw, &f) != nil || f.T != "deliver" {
			continue
		}
		var d linkDeliver
		_ = json.Unmarshal(f.Body, &d)
		ack := map[string]any{"outcome": "accepted"}
		switch d.Kind {
		case "steer":
			if err := a.steerFromControlPlane(f.Exec, d.Message); err != nil {
				ack = map[string]any{"outcome": "rejected", "error": err.Error()}
			} else {
				a.logger.Printf("steer for %s arrived over the link", f.Exec)
			}
		case "cancel":
			a.CancelExecution(f.Exec)
		default:
			ack = map[string]any{"outcome": "rejected", "error": "this node takes " + d.Kind + " over HTTP, not the link"}
		}
		if err := send(linkFrame{T: "ack", ID: f.ID, Exec: f.Exec}, ack); err != nil {
			return true, 0
		}
	}
}
