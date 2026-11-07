package nodelink

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// ── THE LINK AS THE TRANSPORT OF A NODE WITH NO ADDRESS ─────────────────────
//
// RoundTripLink is agenturl.LinkCarrier: a request the control plane would
// have dialed to "<node>.nodelink" goes down that node's link as one http
// delivery, and the node's ack carries the response back. Everything the
// control plane already does to a node — dispatch a reasoner, a nested call,
// cancel, steer, probe — works unchanged on a node that has no inbound port.
//
// A REDELIVERY IS NOT A SECOND REQUEST. An http delivery waits for its ack
// for as long as the caller waits (a synchronous reasoner may take minutes)
// and the outbox redelivers it on every reconnect; the node answers a
// delivery id it has already seen with the answer it gave, or nothing while
// it is still working on it, and runs nothing twice.

// LinkWait is how long a request waits for a node's link to come up before
// it fails as a dial would: long enough for a node that is reconnecting
// after a control plane restart.
var LinkWait = 20 * time.Second

// ErrNoLink is a link-only node with no live link that serves http.
var ErrNoLink = errors.New("node link: the node has no live link")

// RoundTripLink carries req to node down its link.
func (h *Hub) RoundTripLink(node string, req *http.Request) (*http.Response, error) {
	h.init()
	ctx := req.Context()
	deadline := time.Now().Add(LinkWait)
	for !h.LiveFor(node, KindHTTP) {
		if time.Now().After(deadline) {
			return nil, fmt.Errorf("%w: %s", ErrNoLink, node)
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(200 * time.Millisecond):
		}
	}
	var body []byte
	if req.Body != nil {
		raw, err := io.ReadAll(io.LimitReader(req.Body, MaxFrame))
		_ = req.Body.Close()
		if err != nil {
			return nil, err
		}
		body = raw
	}
	headers := map[string]string{}
	for k, v := range req.Header {
		if len(v) > 0 {
			headers[k] = v[0]
		}
	}
	path := req.URL.RequestURI()
	ttl := 6 * time.Hour
	if d, ok := ctx.Deadline(); ok {
		ttl = time.Until(d)
	}
	done := make(chan Outcome, 1)
	id := h.Deliver(node, Delivery{Kind: KindHTTP, ExecutionID: req.Header.Get("X-Execution-ID"), TTL: ttl,
		HTTP:     &HTTPRequest{Method: req.Method, Path: path, Headers: headers, Body: body},
		OnSettle: func(o Outcome) { done <- o }})
	var o Outcome
	select {
	case o = <-done:
	case <-ctx.Done():
		h.settle(id, Outcome{Outcome: OutcomeExpired, Error: "caller gave up"})
		return nil, ctx.Err()
	}
	switch o.Outcome {
	case OutcomeAccepted, OutcomeDone, OutcomeDuplicate:
	case OutcomeExpired:
		return nil, fmt.Errorf("%w: %s did not answer", ErrNoLink, node)
	default:
		return nil, fmt.Errorf("node link: %s refused the request: %s", node, o.Error)
	}
	var res HTTPResult
	if err := json.Unmarshal(o.Result, &res); err != nil || res.Code == 0 {
		return nil, fmt.Errorf("node link: %s answered with no response", node)
	}
	resp := &http.Response{StatusCode: res.Code, Status: fmt.Sprintf("%d %s", res.Code, http.StatusText(res.Code)),
		Proto: "HTTP/1.1", ProtoMajor: 1, ProtoMinor: 1, Header: http.Header{}, Request: req,
		Body: io.NopCloser(bytes.NewReader(res.Body)), ContentLength: int64(len(res.Body))}
	for k, v := range res.Headers {
		resp.Header.Set(k, v)
	}
	if resp.Header.Get("Content-Type") == "" && strings.HasPrefix(strings.TrimSpace(string(res.Body)), "{") {
		resp.Header.Set("Content-Type", "application/json")
	}
	return resp, nil
}
