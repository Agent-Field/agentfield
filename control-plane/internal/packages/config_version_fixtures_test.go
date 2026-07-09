package packages

import (
	"fmt"
	"testing"
)

// ─────────────────────────────────────────────────────────────────────────────
// MAINTENANCE CONTRACT — read this before touching agentfield-package.yaml parsing
// ─────────────────────────────────────────────────────────────────────────────
//
// This file is the executable spec for the manifest schema version (`config_version`).
// Each supported version has ONE golden fixture below: a canonical, fully-populated
// agentfield-package.yaml plus an assertion of the exact structure the reader is
// expected to extract from it. Keep these fixtures in lockstep with the human docs
// in docs/installing-agent-nodes.md — if you change what a version means in the
// docs, a fixture here should change too, and vice-versa.
//
// The rules, enforced by TestConfigVersionFixtureCoverage:
//
//  1. NET-NEW VERSION → NET-NEW FIXTURE. When you bump CurrentConfigVersion (a
//     breaking format change), ADD a new versionFixture entry for it. The coverage
//     guard fails until every version in [0, CurrentConfigVersion] has a fixture.
//
//  2. DON'T REWRITE OLD FIXTURES INTO THE NEW SHAPE. An older version's fixture is
//     a backwards-compatibility guarantee: manifests authored at v(N) must keep
//     parsing forever. Edit an old fixture only to keep it *valid* (e.g. a helper
//     rename), never to migrate it to the new structure. The new structure belongs
//     in the new version's fixture.
//
//  3. ADDITIVE CHANGES DON'T BUMP THE VERSION (so they don't get a new fixture).
//     A new *optional* field is not breaking — extend the CURRENT version's fixture
//     to cover it and assert the reader picks it up. Reserve version bumps (and new
//     fixtures) for changes that alter or remove existing fields.
//
// So: growing the current format = edit the current fixture. Breaking the format =
// bump CurrentConfigVersion + add a fixture. Old fixtures are append-only history.

// versionFixture is one golden agentfield-package.yaml sample for a schema version,
// paired with an assertion of the structure a correct reader extracts from it.
type versionFixture struct {
	version int
	// sample is a complete, representative manifest for this schema version. It
	// should exercise every section a reader is expected to understand at this
	// version so the test genuinely proves structure extraction, not just that the
	// file parses.
	sample string
	// verify asserts the parsed structure matches what this version promises.
	verify func(t *testing.T, md *PackageMetadata)
}

// versionFixtures holds exactly one entry per supported config_version. Append a
// new entry when CurrentConfigVersion is bumped; never delete or migrate an entry.
var versionFixtures = []versionFixture{
	{
		// v0 — the original, pre-versioning format: no `config_version` key at all.
		// This fixture guards that legacy manifests keep parsing unchanged forever.
		version: 0,
		sample: `name: legacy-node
version: 0.1.0
description: A node authored before config_version existed
author: Agent-Field
entrypoint:
  start: python -m legacy_node.app
  healthcheck: /healthz
agent_node:
  node_id: legacy-node
  default_port: 8010
dependencies:
  python:
    - httpx>=0.27
  nodes:
    - af://registry/swe-planner
user_environment:
  required:
    - name: OPENROUTER_API_KEY
      description: LLM provider key
      type: secret
      scope: global
  optional:
    - name: LEGACY_MODEL
      description: Override the model
      default: openrouter/moonshotai/kimi-k2
`,
		verify: func(t *testing.T, md *PackageMetadata) {
			t.Helper()
			if md.ConfigVersion != "" {
				t.Errorf("v0 fixture should omit config_version, got %q", md.ConfigVersion)
			}
			if md.Name != "legacy-node" || md.Version != "0.1.0" {
				t.Errorf("basics: name=%q version=%q", md.Name, md.Version)
			}
			if md.Entrypoint.Start != "python -m legacy_node.app" {
				t.Errorf("entrypoint.start = %q", md.Entrypoint.Start)
			}
			if got := md.HealthcheckPath(); got != "/healthz" {
				t.Errorf("HealthcheckPath() = %q, want /healthz", got)
			}
			if md.AgentNode.NodeID != "legacy-node" || md.AgentNode.DefaultPort != 8010 {
				t.Errorf("agent_node = %+v", md.AgentNode)
			}
			if len(md.Dependencies.Python) != 1 || md.Dependencies.Python[0] != "httpx>=0.27" {
				t.Errorf("dependencies.python = %v", md.Dependencies.Python)
			}
			if len(md.Dependencies.Nodes) != 1 || md.Dependencies.Nodes[0] != "af://registry/swe-planner" {
				t.Errorf("dependencies.nodes = %v", md.Dependencies.Nodes)
			}
			if len(md.UserEnvironment.Required) != 1 {
				t.Fatalf("expected 1 required env var, got %d", len(md.UserEnvironment.Required))
			}
			req := md.UserEnvironment.Required[0]
			if req.Name != "OPENROUTER_API_KEY" || req.Type != "secret" || req.SecretScope("legacy-node") != "global" {
				t.Errorf("required[0] = %+v (scope=%s)", req, req.SecretScope("legacy-node"))
			}
			if len(md.UserEnvironment.Optional) != 1 || md.UserEnvironment.Optional[0].Default != "openrouter/moonshotai/kimi-k2" {
				t.Errorf("optional = %+v", md.UserEnvironment.Optional)
			}
		},
	},
	{
		// v1 — same fields as v0, now explicitly versioned. Mirrors the documented
		// example in docs/installing-agent-nodes.md. Extend THIS fixture when the
		// current format grows an additive optional field.
		version: 1,
		sample: `config_version: v1
name: pr-af
version: 0.2.0
description: Opens draft PRs from a task description
author: Agent-Field
entrypoint:
  start: python -m pr_af.app
  healthcheck: /health
agent_node:
  node_id: pr-af
  default_port: 8004
dependencies:
  python:
    - httpx>=0.27
  nodes:
    - af://registry/swe-planner
user_environment:
  required:
    - name: GH_TOKEN
      description: GitHub token
      type: secret
      scope: global
  require_one_of:
    - id: llm_provider
      description: an LLM provider key
      options:
        - name: ANTHROPIC_API_KEY
          description: Anthropic key (Claude)
          type: secret
        - name: OPENROUTER_API_KEY
          description: OpenRouter key
          type: secret
  optional:
    - name: PR_AF_MODEL
      description: Override the default model
      default: openrouter/moonshotai/kimi-k2
`,
		verify: func(t *testing.T, md *PackageMetadata) {
			t.Helper()
			if md.ConfigVersion != "v1" {
				t.Errorf("v1 fixture should declare config_version: v1, got %q", md.ConfigVersion)
			}
			if md.Name != "pr-af" || md.Version != "0.2.0" {
				t.Errorf("basics: name=%q version=%q", md.Name, md.Version)
			}
			if got := md.StartCommand(); len(got) == 0 || got[0] != "python" {
				t.Errorf("StartCommand() = %v", got)
			}
			if md.AgentNode.NodeID != "pr-af" || md.AgentNode.DefaultPort != 8004 {
				t.Errorf("agent_node = %+v", md.AgentNode)
			}
			if len(md.Dependencies.Nodes) != 1 || md.Dependencies.Nodes[0] != "af://registry/swe-planner" {
				t.Errorf("dependencies.nodes = %v", md.Dependencies.Nodes)
			}
			if len(md.UserEnvironment.Required) != 1 || md.UserEnvironment.Required[0].Name != "GH_TOKEN" {
				t.Errorf("required = %+v", md.UserEnvironment.Required)
			}
			// require_one_of is an additive field (added without a config_version
			// bump): the current-version fixture must exercise that the reader
			// extracts the group and its alternatives.
			if len(md.UserEnvironment.RequireOneOf) != 1 {
				t.Fatalf("expected 1 require_one_of group, got %d", len(md.UserEnvironment.RequireOneOf))
			}
			grp := md.UserEnvironment.RequireOneOf[0]
			if grp.ID != "llm_provider" {
				t.Errorf("require_one_of[0].id = %q", grp.ID)
			}
			if got := grp.OptionNames(); len(got) != 2 || got[0] != "ANTHROPIC_API_KEY" || got[1] != "OPENROUTER_API_KEY" {
				t.Errorf("require_one_of[0] options = %v", got)
			}
			if len(md.UserEnvironment.Optional) != 1 || md.UserEnvironment.Optional[0].Name != "PR_AF_MODEL" {
				t.Errorf("optional = %+v", md.UserEnvironment.Optional)
			}
		},
	},
}

// TestConfigVersionFixtures parses each version's golden manifest through the real
// reader and asserts the reader (a) treats it as the expected schema version and
// (b) extracts the structure that version promises.
func TestConfigVersionFixtures(t *testing.T) {
	for _, fx := range versionFixtures {
		fx := fx
		t.Run(fmt.Sprintf("v%d", fx.version), func(t *testing.T) {
			dir := t.TempDir()
			writeTestPackage(t, dir, fx.sample)

			md, err := ParsePackageMetadata(dir)
			if err != nil {
				t.Fatalf("v%d fixture failed to parse: %v", fx.version, err)
			}
			if got := md.ConfigVersionNumber(); got != fx.version {
				t.Fatalf("reader saw config version v%d, fixture declares v%d", got, fx.version)
			}
			fx.verify(t, md)
		})
	}
}

// TestConfigVersionFixtureCoverage is the forcing function for the maintenance
// contract at the top of this file: there must be exactly one fixture for every
// version from 0 through CurrentConfigVersion, and none beyond it. Bumping
// CurrentConfigVersion without adding the matching fixture fails here.
func TestConfigVersionFixtureCoverage(t *testing.T) {
	seen := map[int]bool{}
	maxVer := 0
	for _, fx := range versionFixtures {
		if seen[fx.version] {
			t.Errorf("duplicate fixture for v%d — one fixture per version", fx.version)
		}
		seen[fx.version] = true
		if fx.version > maxVer {
			maxVer = fx.version
		}
	}
	for v := 0; v <= CurrentConfigVersion; v++ {
		if !seen[v] {
			t.Errorf("missing golden fixture for config_version v%d — add a versionFixture entry (see MAINTENANCE CONTRACT)", v)
		}
	}
	if maxVer != CurrentConfigVersion {
		t.Errorf("highest fixture is v%d but CurrentConfigVersion is v%d — fixtures and the reader are out of step", maxVer, CurrentConfigVersion)
	}
}
