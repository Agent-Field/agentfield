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

// reasonerPath is the reasoner a dispatchable request names: POST
// /reasoners/<name> (or /skills/<name>) on the node's own address.
func reasonerPath(req *http.Request) (string, bool) {
	if req.Method != http.MethodPost {
		return "", false
	}
	p := strings.Trim(req.URL.Path, "/")
	for _, kind := range []string{"reasoners/", "skills/"} {
		if name, ok := strings.CutPrefix(p, kind); ok && name != "" && !strings.Contains(name, "/") {
			return name, true
		}
	}
	return "", false
}

// RoundTripLink carries req to node down its link.
//
// TWO DELIVERY KINDS CARRY A REQUEST. A node whose hello accepts "http" (the
// Go SDK) takes the request itself. A node that speaks only the original
// pair, dispatch and cancel (the codeaf factory), takes a reasoner call as a
// dispatch: the same body and headers an HTTP dispatch carries, and its ack
// read back the way the HTTP answer would be — accepted is 202, done is 200
// with the result, rejected is 422 with the node's words. Anything else such
// a node cannot serve fails as an unreachable node does.
func (h *Hub) RoundTripLink(node string, req *http.Request) (*http.Response, error) {
	h.init()
	ctx := req.Context()
	reasoner, dispatchable := reasonerPath(req)
	kind := func() string {
		if h.LiveFor(node, KindHTTP) {
			return KindHTTP
		}
		if dispatchable && h.LiveFor(node, KindDispatch) {
			return KindDispatch
		}
		return ""
	}
	deadline := time.Now().Add(LinkWait)
	k := kind()
	for k == "" {
		if time.Now().After(deadline) {
			return nil, fmt.Errorf("%w: %s", ErrNoLink, node)
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(200 * time.Millisecond):
		}
		k = kind()
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
			headers[headerSpelling(k)] = v[0]
		}
	}
	path := req.URL.RequestURI()
	ttl := 6 * time.Hour
	if d, ok := ctx.Deadline(); ok {
		ttl = time.Until(d)
	}
	done := make(chan Outcome, 1)
	d := Delivery{Kind: k, ExecutionID: req.Header.Get("X-Execution-ID"), TTL: ttl, OnSettle: func(o Outcome) { done <- o }}
	if k == KindHTTP {
		d.HTTP = &HTTPRequest{Method: req.Method, Path: path, Headers: headers, Body: body}
	} else {
		d.Dispatch = &Dispatch{Reasoner: reasoner, Input: body, Headers: headers}
	}
	id := h.Deliver(node, d)
	var o Outcome
	select {
	case o = <-done:
	case <-ctx.Done():
		h.settle(id, Outcome{Outcome: OutcomeExpired, Error: "caller gave up"})
		return nil, ctx.Err()
	}
	if o.Outcome == OutcomeExpired {
		return nil, fmt.Errorf("%w: %s did not answer", ErrNoLink, node)
	}
	if k == KindDispatch {
		return dispatchResponse(req, o), nil
	}
	switch o.Outcome {
	case OutcomeAccepted, OutcomeDone, OutcomeDuplicate:
	default:
		return nil, fmt.Errorf("node link: %s refused the request: %s", node, o.Error)
	}
	var res HTTPResult
	if err := json.Unmarshal(o.Result, &res); err != nil || res.Code == 0 {
		return nil, fmt.Errorf("node link: %s answered with no response", node)
	}
	return response(req, res.Code, res.Headers, res.Body), nil
}

// headerSpelling is a canonical Go header key in the spelling the control
// plane writes it everywhere else (X-Execution-ID, X-AgentField-Request-ID),
// so a node reading the delivery's header map by key finds what the trigger
// dispatcher's dispatch would have carried.
func headerSpelling(k string) string {
	parts := strings.Split(k, "-")
	for i, p := range parts {
		switch p {
		case "Id":
			parts[i] = "ID"
		case "Vc":
			parts[i] = "VC"
		case "Did":
			parts[i] = "DID"
		case "Agentfield":
			parts[i] = "AgentField"
		}
	}
	return strings.Join(parts, "-")
}

// dispatchResponse is a dispatch ack read as the HTTP answer it replaces.
func dispatchResponse(req *http.Request, o Outcome) *http.Response {
	switch o.Outcome {
	case OutcomeAccepted, OutcomeDuplicate:
		return response(req, http.StatusAccepted, nil, []byte(`{"status":"accepted"}`))
	case OutcomeDone:
		body := []byte(o.Result)
		if len(body) == 0 {
			body = []byte(`{}`)
		}
		return response(req, http.StatusOK, nil, body)
	default:
		raw, _ := json.Marshal(map[string]string{"error": o.Error})
		return response(req, http.StatusUnprocessableEntity, nil, raw)
	}
}

func response(req *http.Request, code int, headers map[string]string, body []byte) *http.Response {
	resp := &http.Response{StatusCode: code, Status: fmt.Sprintf("%d %s", code, http.StatusText(code)),
		Proto: "HTTP/1.1", ProtoMajor: 1, ProtoMinor: 1, Header: http.Header{}, Request: req,
		Body: io.NopCloser(bytes.NewReader(body)), ContentLength: int64(len(body))}
	for k, v := range headers {
		resp.Header.Set(k, v)
	}
	if resp.Header.Get("Content-Type") == "" && strings.HasPrefix(strings.TrimSpace(string(body)), "{") {
		resp.Header.Set("Content-Type", "application/json")
	}
	return resp
}
