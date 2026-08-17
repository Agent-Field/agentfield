// AgentField triggers demo — sample Go agent.
//
// Three deterministic reasoners, each wired to a different Source plugin:
//
//   - handle_payment   ← Stripe webhook (Stripe-Signature HMAC)
//   - handle_pr        ← GitHub webhook (X-Hub-Signature-256 HMAC)
//   - handle_tick      ← cron schedule (every minute)
//
// No LLM calls. Each reasoner transforms its inbound event into a small,
// deterministic record and writes it to per-agent memory, so the UI's event
// log and run detail surfaces show real data flowing through.
//
// When the agent registers with the control plane, the OnEvent / OnSchedule
// declarations auto-create code-managed Trigger rows. The CP returns the
// public ingest URL for each, visible at http://localhost:8080/ui/triggers.
//
// Equivalent to examples/triggers-demo/agent.py and
// examples/triggers-demo-ts/agent.ts — same reasoner shapes, same memory
// keys, same records, driven by the same scripts/fire-events.sh.
//
// The reasoners are declared as eventReasoner / scheduleReasoner values rather
// than inline inside registration calls, so main_test.go can exercise the
// handlers and assert the trigger wiring without needing a control plane.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/sdk/go/agent"
	"github.com/Agent-Field/agentfield/sdk/go/triggers"
)

// eventReasoner is a webhook-triggered reasoner: the trigger options plus the
// handler factory. The factory takes the agent so the handler can reach
// Memory() without a package-level global.
type eventReasoner struct {
	Name    string
	Opts    triggers.EventOpts
	Handler func(*agent.Agent) agent.HandlerFunc
}

// scheduleReasoner is a cron-triggered reasoner.
type scheduleReasoner struct {
	Name    string
	Cron    string
	Handler func(*agent.Agent) agent.HandlerFunc
}

// eventReasoners are the demo's webhook-driven reasoners.
func eventReasoners() []eventReasoner {
	return []eventReasoner{paymentReasoner(), pullRequestReasoner()}
}

// scheduleReasoners are the demo's cron-driven reasoners.
func scheduleReasoners() []scheduleReasoner {
	return []scheduleReasoner{tickReasoner()}
}

func main() {
	nodeID := envOr("AGENT_NODE_ID", "triggers-demo-agent")
	agentFieldURL := envOr("AGENTFIELD_URL", "http://localhost:8080")
	port := envOr("PORT", "8001")

	publicURL := strings.TrimSpace(os.Getenv("AGENT_CALLBACK_URL"))
	if publicURL == "" {
		publicURL = "http://localhost:" + port
	}

	app, err := agent.New(agent.Config{
		NodeID:        nodeID,
		Version:       "1.0.0",
		AgentFieldURL: agentFieldURL,
		ListenAddress: ":" + port,
		PublicURL:     publicURL,
	})
	if err != nil {
		log.Fatalf("failed to create agent: %v", err)
	}

	for _, r := range eventReasoners() {
		app.OnEvent(r.Opts, r.Name, r.Handler(app))
	}
	for _, r := range scheduleReasoners() {
		app.OnSchedule(r.Cron, r.Name, r.Handler(app))
	}

	fmt.Fprintf(os.Stderr,
		"AgentField triggers demo (Go) — sample agent starting\n"+
			"  node_id            = %s\n"+
			"  agentfield_server  = %s\n"+
			"  callback url       = %s\n"+
			"  reasoners          = handle_payment (stripe), handle_pr (github), handle_tick (cron)\n",
		nodeID, agentFieldURL, publicURL)

	go heartbeat(nodeID)

	if err := app.Run(context.Background()); err != nil {
		log.Fatalf("agent exited: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Stripe — payment events
//
// The Stripe source plugin verifies Stripe-Signature: t=<ts>,v1=<hmac> over
// "<ts>.<body>" using the secret read from STRIPE_DEMO_SECRET on the CP host.
// The Transform pulls the bits we actually care about out of Stripe's fairly
// nested envelope so the reasoner body stays clean.
// ---------------------------------------------------------------------------

func paymentReasoner() eventReasoner {
	return eventReasoner{
		Name: "handle_payment",
		Opts: triggers.EventOpts{
			Source:    "stripe",
			Types:     []string{"payment_intent.succeeded"},
			SecretEnv: "STRIPE_DEMO_SECRET",
			Transform: stripeToPayment,
		},
		Handler: func(app *agent.Agent) agent.HandlerFunc {
			return func(ctx context.Context, payment map[string]any) (any, error) {
				tc := triggers.FromContext(ctx)

				record := map[string]any{
					"kind":             "payment",
					"stripe_id":        payment["id"],
					"amount_cents":     payment["amount"],
					"currency":         payment["currency"],
					"customer":         payment["customer"],
					"received_via":     receivedVia(tc),
					"trigger_event_id": eventID(tc),
				}

				key := fmt.Sprintf("payment:%v", record["stripe_id"])
				if err := app.Memory().Set(ctx, key, record); err != nil {
					// Memory is the demo's observable output, so surface the
					// failure rather than returning a record never persisted.
					return nil, fmt.Errorf("persisting %s: %w", key, err)
				}

				log.Printf("[handle_payment] saved %v", record)
				return record, nil
			}
		},
	}
}

// stripeToPayment flattens Stripe's data.object into the fields the reasoner
// needs. Runs before the handler on trigger dispatches, skipped on direct calls.
func stripeToPayment(event map[string]any) any {
	obj := nestedMap(event, "data", "object")

	currency := obj["currency"]
	if currency == nil || currency == "" {
		currency = "usd"
	}
	metadata := obj["metadata"]
	if metadata == nil {
		metadata = map[string]any{}
	}

	return map[string]any{
		"id":       obj["id"],
		"amount":   obj["amount"],
		"currency": currency,
		"customer": obj["customer"],
		"status":   obj["status"],
		"metadata": metadata,
	}
}

// ---------------------------------------------------------------------------
// GitHub — pull-request events
//
// The GitHub source verifies X-Hub-Signature-256 = sha256=<hmac of body>
// using the secret from GITHUB_DEMO_SECRET, and reads X-GitHub-Event plus
// X-GitHub-Delivery for the event type and idempotency key.
// ---------------------------------------------------------------------------

func pullRequestReasoner() eventReasoner {
	return eventReasoner{
		Name: "handle_pr",
		Opts: triggers.EventOpts{
			Source:    "github",
			Types:     []string{"pull_request"},
			SecretEnv: "GITHUB_DEMO_SECRET",
		},
		Handler: func(app *agent.Agent) agent.HandlerFunc {
			return func(ctx context.Context, event map[string]any) (any, error) {
				tc := triggers.FromContext(ctx)

				pr := mapAt(event, "pull_request")
				user := mapAt(pr, "user")
				repo := mapAt(event, "repository")

				// The top-level "number" is authoritative when present; some
				// event types only carry it on the nested object.
				number := event["number"]
				if number == nil {
					number = pr["number"]
				}

				record := map[string]any{
					"kind":         "pull_request",
					"action":       event["action"],
					"number":       number,
					"title":        pr["title"],
					"html_url":     pr["html_url"],
					"user":         user["login"],
					"repo":         repo["full_name"],
					"received_via": receivedVia(tc),
					"delivery_id":  idempotencyKey(tc),
				}

				// Only persist with both parts of the key, matching the Python
				// demo: a malformed payload must not write "pr:<nil>#<nil>".
				if record["repo"] != nil && record["repo"] != "" && number != nil {
					key := fmt.Sprintf("pr:%v#%v", record["repo"], number)
					if err := app.Memory().Set(ctx, key, record); err != nil {
						return nil, fmt.Errorf("persisting %s: %w", key, err)
					}
				}

				log.Printf("[handle_pr] saved %v", record)
				return record, nil
			}
		},
	}
}

// ---------------------------------------------------------------------------
// Cron — periodic tick
//
// The cron source runs as a LoopSource inside the CP, emitting a "tick" event
// each time the schedule fires. The agent sees the same dispatch shape as any
// other webhook delivery, so the reasoner code path is identical.
// ---------------------------------------------------------------------------

const tickCounterKey = "cron:tick:count"

func tickReasoner() scheduleReasoner {
	return scheduleReasoner{
		Name: "handle_tick",
		Cron: "* * * * *",
		Handler: func(app *agent.Agent) agent.HandlerFunc {
			return func(ctx context.Context, _ map[string]any) (any, error) {
				tc := triggers.FromContext(ctx)

				current, err := app.Memory().Get(ctx, tickCounterKey)
				if err != nil {
					return nil, fmt.Errorf("reading %s: %w", tickCounterKey, err)
				}

				record := map[string]any{
					"count":         previousCount(current) + 1,
					"last_fired_at": firedAt(tc),
					"received_via":  receivedVia(tc),
				}

				if err := app.Memory().Set(ctx, tickCounterKey, record); err != nil {
					return nil, fmt.Errorf("persisting %s: %w", tickCounterKey, err)
				}

				log.Printf("[handle_tick] %v", record)
				return record, nil
			}
		},
	}
}

// previousCount reads the counter out of a stored record, tolerating both a
// missing key (first tick) and the numeric widening a JSON round-trip through
// the control plane introduces: an int written today comes back as a float64,
// so both have to be handled or the counter silently resets to zero.
func previousCount(stored any) int {
	rec, ok := stored.(map[string]any)
	if !ok {
		return 0
	}
	switch v := rec["count"].(type) {
	case int:
		return v
	case int64:
		return int(v)
	case float64:
		return int(v)
	default:
		return 0
	}
}

// ---------------------------------------------------------------------------
// Trigger-context helpers
//
// Each mirrors the Python demo's `trigger.x if trigger else <fallback>`, so a
// reasoner invoked directly (no envelope) records "direct_call" rather than
// failing on a nil context.
// ---------------------------------------------------------------------------

func receivedVia(tc *triggers.Context) string {
	if tc == nil {
		return "direct_call"
	}
	return tc.Source
}

func eventID(tc *triggers.Context) any {
	if tc == nil {
		return nil
	}
	return tc.EventID
}

func idempotencyKey(tc *triggers.Context) any {
	if tc == nil {
		return nil
	}
	return tc.IdempotencyKey
}

func firedAt(tc *triggers.Context) any {
	if tc == nil {
		return nil
	}
	return tc.ReceivedAt.Format(time.RFC3339)
}

// ---------------------------------------------------------------------------
// Small utilities
// ---------------------------------------------------------------------------

// mapAt returns the nested object at key, or an empty map when it is absent or
// not an object, so callers can index the result without a nil check.
func mapAt(m map[string]any, key string) map[string]any {
	if m == nil {
		return map[string]any{}
	}
	nested, ok := m[key].(map[string]any)
	if !ok {
		return map[string]any{}
	}
	return nested
}

// nestedMap walks a chain of object keys, returning an empty map if any hop is
// missing or not an object.
func nestedMap(m map[string]any, keys ...string) map[string]any {
	current := m
	for _, k := range keys {
		current = mapAt(current, k)
	}
	return current
}

func envOr(key, fallback string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return fallback
}

// heartbeat surfaces in container logs that the agent is alive between events,
// so an idle demo does not look hung.
func heartbeat(nodeID string) {
	for n := 0; ; n++ {
		log.Printf("[%s] alive heartbeat #%d", nodeID, n)
		time.Sleep(30 * time.Second)
	}
}
