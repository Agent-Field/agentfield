package agent

import (
	"fmt"
	"strings"
)

// ─────────────────────────────────────────────────────────────────────────────
// RouterOptions
// ─────────────────────────────────────────────────────────────────────────────

// RouterOptions controls how a Router is mounted onto another Router or Agent.
type RouterOptions struct {
	// Prefix is prepended to every registered name using dot-notation.
	// E.g. Prefix "users" turns "get-profile" into "users.get-profile".
	Prefix string

	// Tags are merged into the tags of every handler in the child router.
	Tags []string
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal registration records
// ─────────────────────────────────────────────────────────────────────────────

type routerHandlerEntry struct {
	name    string
	handler HandlerFunc
	opts    []ReasonerOption
}

type routerChildEntry struct {
	router *Router
	opts   RouterOptions
}

// ─────────────────────────────────────────────────────────────────────────────
// Router
// ─────────────────────────────────────────────────────────────────────────────

// Router groups related reasoners and skills into a reusable, namespaced
// module that can be mounted onto an Agent (or another Router) with a prefix.
//
// Example:
//
//	userRouter := agent.NewRouter()
//	userRouter.RegisterReasoner("get-profile", getProfileFn,
//	    agent.WithDescription("Fetch a user by ID"))
//	userRouter.RegisterSkill("validate-email", validateEmailFn)
//
//	a.IncludeRouter(userRouter, agent.RouterOptions{
//	    Prefix: "users",
//	    Tags:   []string{"user-management"},
//	})
//	// Registers: users.get-profile, users.validate-email
type Router struct {
	entries  []routerHandlerEntry
	children []routerChildEntry
}

// NewRouter returns an empty, ready-to-use Router.
func NewRouter() *Router {
	return &Router{}
}

// RegisterReasoner adds a reasoner handler to the router.
// opts accepts any ReasonerOption, e.g. WithDescription, WithReasonerTags.
func (r *Router) RegisterReasoner(name string, handler HandlerFunc, opts ...ReasonerOption) {
	r.entries = append(r.entries, routerHandlerEntry{
		name:    name,
		handler: handler,
		opts:    opts,
	})
}

// RegisterSkill adds a skill handler to the router.
// In the current Go SDK, skills share the same registration path as reasoners;
// the skill/reasoner distinction is enforced at the control-plane level.
// opts accepts any ReasonerOption, e.g. WithDescription, WithReasonerTags.
func (r *Router) RegisterSkill(name string, handler HandlerFunc, opts ...ReasonerOption) {
	// Skills are registered identically to reasoners in the Go SDK today.
	r.entries = append(r.entries, routerHandlerEntry{
		name:    name,
		handler: handler,
		opts:    opts,
	})
}

// IncludeRouter nests another Router inside this one. When this router is
// later mounted on an Agent, the child router's handlers are prefixed with
// opts.Prefix (relative to whatever prefix this router itself carries).
func (r *Router) IncludeRouter(child *Router, opts RouterOptions) {
	r.children = append(r.children, routerChildEntry{router: child, opts: opts})
}

// ─────────────────────────────────────────────────────────────────────────────
// Flattening
// ─────────────────────────────────────────────────────────────────────────────

// withAppendTags returns a ReasonerOption that *appends* tags to whatever tags
// are already present on the Reasoner. This is used internally to layer
// RouterOptions.Tags on top of per-handler WithReasonerTags without clobbering.
func withAppendTags(tags ...string) ReasonerOption {
	return func(r *Reasoner) {
		r.Tags = mergeTags(r.Tags, tags)
	}
}

// flatten recursively walks the router tree, applying prefix and tag
// inheritance, and returns a flat list of resolved handler entries.
// It is a pure read — the original router is never mutated.
func (r *Router) flatten(prefix string, inheritedTags []string) []routerHandlerEntry {
	var out []routerHandlerEntry

	for _, e := range r.entries {
		name := joinName(prefix, e.name)
		opts := make([]ReasonerOption, len(e.opts))
		copy(opts, e.opts)

		// Append inherited tags last so they merge without overwriting explicit
		// per-handler tags set via WithReasonerTags.
		if len(inheritedTags) > 0 {
			opts = append(opts, withAppendTags(inheritedTags...))
		}

		out = append(out, routerHandlerEntry{
			name:    name,
			handler: e.handler,
			opts:    opts,
		})
	}

	for _, c := range r.children {
		childPrefix := joinName(prefix, c.opts.Prefix)
		childTags := mergeTags(inheritedTags, c.opts.Tags)
		out = append(out, c.router.flatten(childPrefix, childTags)...)
	}

	return out
}

// ─────────────────────────────────────────────────────────────────────────────
// Agent.IncludeRouter
// ─────────────────────────────────────────────────────────────────────────────

// IncludeRouter flattens all handlers from router into the Agent, prepending
// opts.Prefix to every name and merging opts.Tags into every handler's tags.
//
// It delegates to the existing Agent.RegisterReasoner so all middleware
// (DID auth, VC generation, tracing) fires without any changes.
//
// Example — flat mount:
//
//	orderRouter := agent.NewRouter()
//	orderRouter.RegisterReasoner("create", createOrderFn)
//	orderRouter.RegisterReasoner("cancel", cancelOrderFn)
//
//	a.IncludeRouter(orderRouter, agent.RouterOptions{
//	    Prefix: "orders",
//	    Tags:   []string{"order-management"},
//	})
//	// Registers: orders.create, orders.cancel
//
// Example — nested mount:
//
//	adminRouter := agent.NewRouter()
//	adminRouter.IncludeRouter(userRouter,  agent.RouterOptions{Prefix: "users"})
//	adminRouter.IncludeRouter(orderRouter, agent.RouterOptions{Prefix: "orders"})
//
//	a.IncludeRouter(adminRouter, agent.RouterOptions{Prefix: "admin"})
//	// Registers: admin.users.get-profile, admin.orders.create, …
func (a *Agent) IncludeRouter(router *Router, opts RouterOptions) {
	for _, e := range router.flatten(opts.Prefix, opts.Tags) {
		a.RegisterReasoner(e.name, e.handler, e.opts...)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

// joinName concatenates prefix and name with a dot, skipping the dot when
// either part is empty.
func joinName(prefix, name string) string {
	prefix = strings.TrimSpace(prefix)
	name = strings.TrimSpace(name)
	switch {
	case prefix == "":
		return name
	case name == "":
		return prefix
	default:
		return fmt.Sprintf("%s.%s", prefix, name)
	}
}

// mergeTags deduplicates and concatenates two tag slices, preserving order.
func mergeTags(a, b []string) []string {
	seen := make(map[string]struct{}, len(a)+len(b))
	out := make([]string, 0, len(a)+len(b))
	for _, t := range append(a, b...) {
		if _, ok := seen[t]; !ok {
			seen[t] = struct{}{}
			out = append(out, t)
		}
	}
	return out
}
