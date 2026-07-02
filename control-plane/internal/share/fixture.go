package share

import "fmt"

// strptr / f64 / i64 are small helpers to build pointer-valued fixture fields.
func strptr(s string) *string   { return &s }
func f64(v float64) *float64    { return &v }
func i64(v int64) *int64        { return &v }

// DemoBundle returns a realistic security-audit run: a recon/hunt/prove
// pipeline with three layers of fan-out, mixed statuses, one failed branch,
// and plausible per-node costs summing to about $0.42. It exists so `af share
// --demo` produces a product-quality artifact with no control plane, and so
// the template can be developed offline.
func DemoBundle() *Bundle {
	base := "2026-07-02T14:20:00Z"
	_ = base

	type spec struct {
		id     string
		agent  string
		fn     string
		status string
		start  string
		end    string
		dur    int64
		cost   float64
		model  string
		in     string
		out    string
		err    string
		parent string
	}

	// t builds an RFC3339 timestamp at 14:20:SS on the fixture day.
	t := func(sec int) string {
		return fmt.Sprintf("2026-07-02T14:20:%02dZ", sec)
	}

	specs := []spec{
		// Layer 0 — orchestrator root
		{"exec-000", "orchestrator", "run_audit", "succeeded", t(0), t(58), 58210, 0.0142, "claude-opus-4-8",
			`{"target":"acme-payments.internal","scope":["api","web","auth"],"depth":"deep"}`,
			`{"findings":7,"critical":1,"high":2,"proven":4,"report":"acme-audit-2026-07-02.md"}`, "", ""},

		// Layer 1 — recon surface mapping (fan-out of 4)
		{"exec-010", "recon", "map_attack_surface", "succeeded", t(1), t(9), 8140, 0.0121, "claude-sonnet-4-6",
			`{"target":"acme-payments.internal","modules":["api","web","auth","admin"]}`,
			`{"endpoints":142,"auth_flows":6,"public_forms":11,"admin_panels":2,"tech":["nginx","postgres","node18"]}`, "", "exec-000"},
		{"exec-011", "recon", "enumerate_endpoints", "succeeded", t(2), t(11), 9020, 0.0104, "claude-sonnet-4-6",
			`{"base":"https://api.acme-payments.internal","openapi":true}`,
			`{"routes":142,"unauthenticated":18,"deprecated":4,"graphql":true,"rate_limited":false}`, "", "exec-010"},
		{"exec-012", "recon", "fingerprint_stack", "succeeded", t(2), t(8), 5980, 0.0067, "claude-haiku-4-5",
			`{"headers":true,"tls":true,"cookies":true}`,
			`{"server":"nginx/1.24","runtime":"node/18.19","orm":"prisma","session":"jwt","cdn":"cloudflare"}`, "", "exec-010"},
		{"exec-013", "recon", "harvest_secrets", "succeeded", t(3), t(14), 11200, 0.0093, "claude-sonnet-4-6",
			`{"sources":["js_bundles","source_maps","git_history"]}`,
			`{"api_keys_found":0,"leaked_tokens":1,"source_maps_public":true,"note":"stripe test key in bundle.js"}`, "", "exec-010"},

		// Layer 2 — hunters (fan-out of ~10, one branch fails)
		{"exec-100", "hunter", "sql_injection", "succeeded", t(9), t(21), 12340, 0.0212, "claude-sonnet-4-6",
			`{"endpoints":18,"params":["id","order","filter"],"blind":true}`,
			`{"candidates":3,"confirmed_reflection":2,"time_based":1,"top":"/api/orders?filter= (postgres error leak)"}`, "", "exec-011"},
		{"exec-101", "hunter", "auth_bypass", "succeeded", t(9), t(24), 14880, 0.0234, "claude-opus-4-8",
			`{"flows":6,"jwt":true,"oauth":false}`,
			`{"candidates":2,"jwt_alg_confusion":true,"idor_on":"/api/users/{id}/cards","note":"HS256 accepts RS256 pubkey"}`, "", "exec-011"},
		{"exec-102", "hunter", "idor", "succeeded", t(10), t(19), 8420, 0.0098, "claude-sonnet-4-6",
			`{"objects":["cards","invoices","payouts"],"seq_ids":true}`,
			`{"candidates":4,"confirmed":2,"objects":["cards","payouts"],"note":"sequential integer ids, no ownership check"}`, "", "exec-011"},
		{"exec-103", "hunter", "xss_reflected", "succeeded", t(11), t(18), 7010, 0.0072, "claude-haiku-4-5",
			`{"forms":11,"params":42,"csp":"weak"}`,
			`{"candidates":5,"reflected":3,"stored":0,"note":"search box reflects unescaped q param"}`, "", "exec-011"},
		{"exec-104", "hunter", "ssrf", "succeeded", t(12), t(22), 9640, 0.0158, "claude-sonnet-4-6",
			`{"features":["webhook_url","avatar_import","pdf_render"]}`,
			`{"candidates":2,"confirmed":1,"vector":"webhook_url -> 169.254.169.254","note":"cloud metadata reachable"}`, "", "exec-011"},
		{"exec-105", "hunter", "secrets_scan", "succeeded", t(12), t(20), 7880, 0.0061, "claude-haiku-4-5",
			`{"leaked_tokens":1,"source":"bundle.js"}`,
			`{"validated":1,"scope":"stripe_test","live":false,"note":"test-mode key, low severity"}`, "", "exec-013"},
		{"exec-106", "hunter", "rate_limit_abuse", "succeeded", t(13), t(23), 9310, 0.0084, "claude-sonnet-4-6",
			`{"routes":142,"rate_limited":false}`,
			`{"abusable":["/api/login","/api/otp","/api/coupon"],"note":"otp brute forceable, no lockout"}`, "", "exec-011"},
		{"exec-107", "hunter", "csrf", "succeeded", t(13), t(19), 6120, 0.0059, "claude-haiku-4-5",
			`{"state_changing":34,"samesite":"lax"}`,
			`{"candidates":1,"note":"password change lacks csrf token; samesite=lax mitigates most"}`, "", "exec-011"},
		{"exec-108", "hunter", "deserialization", "failed", t(14), t(16), 2110, 0.0031, "claude-sonnet-4-6",
			`{"formats":["json","yaml"],"endpoints":["/api/import"]}`,
			"",
			"harness aborted: /api/import returned 502 for all probe payloads; upstream unreachable during scan window", "exec-011"},
		{"exec-109", "hunter", "metadata_exposure", "succeeded", t(14), t(25), 10480, 0.0102, "claude-sonnet-4-6",
			`{"vector":"webhook_url -> 169.254.169.254"}`,
			`{"reachable":true,"iam_role":"acme-payments-worker","creds_dumpable":true,"note":"chain from ssrf finding"}`, "", "exec-104"},
		{"exec-110", "hunter", "graphql_introspection", "succeeded", t(10), t(17), 6740, 0.0079, "claude-sonnet-4-6",
			`{"graphql":true,"introspection":"unknown"}`,
			`{"introspection_enabled":true,"types":214,"mutations":38,"note":"full schema disclosed, batching allowed"}`, "", "exec-011"},
		{"exec-111", "hunter", "path_traversal", "succeeded", t(11), t(20), 8630, 0.0088, "claude-sonnet-4-6",
			`{"file_params":["template","invoice","export"]}`,
			`{"candidates":2,"confirmed":1,"vector":"/api/invoice?file=../../etc/passwd","note":"read outside webroot"}`, "", "exec-011"},
		{"exec-112", "hunter", "open_redirect", "succeeded", t(12), t(18), 5410, 0.0052, "claude-haiku-4-5",
			`{"redirect_params":["next","return_to","callback"]}`,
			`{"candidates":3,"confirmed":2,"note":"login next= allows external host, phishing vector"}`, "", "exec-011"},
		{"exec-113", "hunter", "mass_assignment", "succeeded", t(13), t(22), 8940, 0.0097, "claude-sonnet-4-6",
			`{"models":["user","account"],"orm":"prisma"}`,
			`{"candidates":1,"confirmed":1,"field":"role","note":"PATCH /api/users/{id} accepts role=admin"}`, "", "exec-011"},
		{"exec-114", "hunter", "cors_misconfig", "succeeded", t(13), t(19), 5980, 0.0054, "claude-haiku-4-5",
			`{"origin_reflection":"unknown","credentials":true}`,
			`{"reflects_origin":true,"allow_credentials":true,"note":"any origin can read authed responses"}`, "", "exec-012"},

		// Layer 2b — deep dive spawned by the sql_injection hunter
		{"exec-120", "hunter", "dump_schema", "succeeded", t(21), t(30), 8710, 0.0164, "claude-sonnet-4-6",
			`{"vector":"error_based","endpoint":"/api/orders","dbms":"postgres"}`,
			`{"tables":41,"sensitive":["users","cards","payment_intents"],"note":"column names enumerated, no data pulled"}`, "", "exec-100"},

		// Layer 3 — provers (verify the promising hunter candidates)
		{"exec-200", "prover", "verify_finding", "succeeded", t(21), t(34), 13020, 0.0245, "claude-opus-4-8",
			`{"finding":"sql_injection","endpoint":"/api/orders","type":"error_based"}`,
			`{"proven":true,"severity":"high","dbms":"postgres","extracted":"schema+1 row (redacted)","cvss":8.2}`, "", "exec-100"},
		{"exec-201", "prover", "verify_finding", "succeeded", t(24), t(41), 16900, 0.0298, "claude-opus-4-8",
			`{"finding":"auth_bypass","vector":"jwt_alg_confusion"}`,
			`{"proven":true,"severity":"critical","impact":"full account takeover","cvss":9.6,"note":"forged admin token accepted"}`, "", "exec-101"},
		{"exec-202", "prover", "verify_finding", "succeeded", t(19), t(29), 9740, 0.0179, "claude-sonnet-4-6",
			`{"finding":"idor","object":"cards"}`,
			`{"proven":true,"severity":"high","impact":"read any user card metadata","cvss":7.5}`, "", "exec-102"},
		{"exec-203", "prover", "verify_finding", "succeeded", t(22), t(33), 10610, 0.0201, "claude-sonnet-4-6",
			`{"finding":"ssrf","vector":"webhook_url"}`,
			`{"proven":true,"severity":"high","chained_with":"metadata_exposure","cvss":8.1}`, "", "exec-104"},
		{"exec-204", "prover", "verify_finding", "succeeded", t(18), t(26), 7290, 0.0088, "claude-sonnet-4-6",
			`{"finding":"xss_reflected","param":"q"}`,
			`{"proven":false,"reason":"csp blocks inline script execution in modern browsers","severity":"low","cvss":3.1}`, "", "exec-103"},
		{"exec-205", "prover", "verify_finding", "succeeded", t(23), t(31), 7640, 0.0091, "claude-sonnet-4-6",
			`{"finding":"rate_limit_abuse","route":"/api/otp"}`,
			`{"proven":true,"severity":"medium","impact":"otp brute force ~10k/min","cvss":6.5}`, "", "exec-106"},
		{"exec-206", "prover", "verify_finding", "succeeded", t(25), t(36), 10200, 0.0201, "claude-opus-4-8",
			`{"finding":"metadata_exposure","iam_role":"acme-payments-worker"}`,
			`{"proven":true,"severity":"critical","impact":"cloud credential theft","cvss":9.1,"note":"depends on ssrf"}`, "", "exec-109"},
		{"exec-207", "prover", "verify_finding", "succeeded", t(20), t(28), 7420, 0.0158, "claude-sonnet-4-6",
			`{"finding":"path_traversal","endpoint":"/api/invoice"}`,
			`{"proven":true,"severity":"high","impact":"arbitrary file read","cvss":7.7}`, "", "exec-111"},
		{"exec-208", "prover", "verify_finding", "succeeded", t(22), t(30), 7130, 0.0141, "claude-sonnet-4-6",
			`{"finding":"mass_assignment","field":"role"}`,
			`{"proven":true,"severity":"critical","impact":"privilege escalation to admin","cvss":9.0}`, "", "exec-113"},
		{"exec-209", "prover", "verify_finding", "succeeded", t(19), t(26), 6280, 0.0082, "claude-sonnet-4-6",
			`{"finding":"cors_misconfig","reflects_origin":true}`,
			`{"proven":true,"severity":"medium","impact":"cross-origin data theft when authed","cvss":6.1}`, "", "exec-114"},
		{"exec-210", "prover", "verify_finding", "failed", t(20), t(23), 2740, 0.0044, "claude-sonnet-4-6",
			`{"finding":"open_redirect","param":"next"}`,
			"",
			"prover inconclusive: target rate-limited the verification probes after 40 requests; requeued for retry", "exec-112"},

		// Layer 4 — reporter (fan-in)
		{"exec-300", "reporter", "synthesize_report", "succeeded", t(41), t(57), 15870, 0.0212, "claude-opus-4-8",
			`{"proven_findings":6,"critical":2,"high":3,"medium":1,"low":1}`,
			`{"report":"acme-audit-2026-07-02.md","exec_summary":true,"remediation_tickets":6,"pages":14}`, "", "exec-000"},
		{"exec-301", "reporter", "score_severity", "succeeded", t(41), t(48), 6410, 0.0055, "claude-haiku-4-5",
			`{"findings":6,"framework":"cvss3.1"}`,
			`{"max_cvss":9.6,"avg_cvss":7.9,"critical":["auth_bypass","metadata_exposure"]}`, "", "exec-300"},
	}

	// modelFor maps each agent role to a realistic current model string. The
	// orchestrator and reporter run on the strongest model (opus); the recon,
	// hunter, and prover workers run on sonnet. This keeps the fixture aligned
	// with how a real audit run would allocate models by role.
	modelFor := func(agent string) string {
		switch agent {
		case "orchestrator", "reporter":
			return "anthropic/claude-opus-4-8"
		default: // recon, hunter, prover
			return "anthropic/claude-sonnet-4-6"
		}
	}

	nodes := make([]BundleNode, 0, len(specs))
	edges := make([]BundleEdge, 0, len(specs))
	var totalCost float64
	var maxEndSec int64

	for _, s := range specs {
		var errPtr *string
		if s.err != "" {
			errPtr = strptr(s.err)
		}
		node := BundleNode{
			ID:            s.id,
			Agent:         s.agent,
			Func:          s.fn,
			Status:        s.status,
			StartedAt:     s.start,
			EndedAt:       s.end,
			DurationMS:    i64(s.dur),
			CostUSD:       f64(s.cost),
			Model:         strptr(modelFor(s.agent)),
			InputPreview:  TruncatePreview(s.in),
			OutputPreview: TruncatePreview(s.out),
			Error:         errPtr,
		}
		nodes = append(nodes, node)
		if s.parent != "" {
			edges = append(edges, BundleEdge{From: s.parent, To: s.id})
		}
		totalCost += s.cost
	}

	// Derive wall-clock duration from the latest end timestamp (root started at :00).
	maxEndSec = 58
	durationMS := maxEndSec * 1000

	return &Bundle{
		Version:     BundleVersion,
		WorkflowID:  "run-a1b2c3d4-sec-audit",
		Title:       "Security audit: dvga.example.com",
		GeneratedAt: "2026-07-02T14:21:03Z",
		Totals: BundleTotals{
			Agents:     len(nodes),
			DurationMS: durationMS,
			CostUSD:    f64(roundCost(totalCost)),
		},
		Nodes: nodes,
		Edges: edges,
	}
}

// roundCost rounds a USD figure to cents so the totals strip reads cleanly.
func roundCost(v float64) float64 {
	return float64(int64(v*100+0.5)) / 100
}
