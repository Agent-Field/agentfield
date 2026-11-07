// workaf-research answers an engineering question by trying things, not by
// recalling them.
//
//	research                 entry point: {"question": "..."}
//	  plan                   a coding agent turns the question into criteria and one probe program
//	  investigate_mattn      ┐
//	  investigate_modernc    ├ in parallel: a coding agent builds and measures the probe on one driver
//	  investigate_ncruces    ┘
//	  synthesize             a coding agent compares the three measurements and recommends one
//
// The default question is "which Go SQLite driver should a small CLI use?".
// Each investigate_* reasoner is its own execution, called at the same time,
// so the run's tree shows the three as siblings.
//
// Configuration is environment only:
//
//	AGENTFIELD_URL, AGENTFIELD_API_KEY (or _FILE), AGENT_NODE_ID (default workaf-research),
//	AGENT_LISTEN (default 127.0.0.1:8003), AGENT_CALLBACK_URL,
//	RESEARCH_MODEL (default deepseek/deepseek-v4.1-flash), CODEAF_BIN,
//	OPENROUTER_API_KEY (or _FILE), RESEARCH_WORKDIR (default: a temp dir)
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"sync"

	"github.com/Agent-Field/agentfield/sdk/go/agent"
)

const defaultQuestion = "Which Go SQLite driver should a small command-line tool use: mattn/go-sqlite3, modernc.org/sqlite or ncruces/go-sqlite3? It must cross-compile for macOS, Linux and Windows from one machine, start fast and stay small."

// drivers are the candidates, one investigate_* reasoner each.
var drivers = []struct{ reasoner, module, sqlName, note string }{
	{"investigate_mattn", "github.com/mattn/go-sqlite3", "sqlite3", "needs cgo"},
	{"investigate_modernc", "modernc.org/sqlite", "sqlite", "pure Go, transpiled C"},
	{"investigate_ncruces", "github.com/ncruces/go-sqlite3", "sqlite3", "pure Go, SQLite compiled to Wasm; also import github.com/ncruces/go-sqlite3/driver and .../embed"},
}

func main() {
	listen := envOr("AGENT_LISTEN", "127.0.0.1:8003")
	app, err := agent.New(agent.Config{
		NodeID:        envOr("AGENT_NODE_ID", "workaf-research"),
		Version:       "0.1.0",
		AgentFieldURL: envOr("AGENTFIELD_URL", "http://localhost:8080"),
		APIKey:        secret("AGENTFIELD_API_KEY"),
		ListenAddress: listen,
		PublicURL:     envOr("AGENT_CALLBACK_URL", "http://"+listen),
		HarnessConfig: &agent.HarnessConfig{
			Provider: "codeaf",
			Model:    envOr("RESEARCH_MODEL", "deepseek/deepseek-v4.1-flash"),
			BinPath:  os.Getenv("CODEAF_BIN"),
			MaxTurns: 40,
			Timeout:  1200,
			Env:      map[string]string{"OPENROUTER_API_KEY": secret("OPENROUTER_API_KEY")},
		},
	})
	if err != nil {
		log.Fatalf("workaf-research: %v", err)
	}
	r := &research{app: app, root: envOr("RESEARCH_WORKDIR", os.TempDir())}

	app.RegisterReasoner("research", r.research, agent.WithDescription("Answer an engineering question by building and measuring the options."))
	app.RegisterReasoner("plan", r.plan, agent.WithDescription("Turn the question into criteria and one probe program."))
	for _, d := range drivers {
		d := d
		app.RegisterReasoner(d.reasoner, func(ctx context.Context, in map[string]any) (any, error) {
			return r.investigate(ctx, in, d.module, d.sqlName, d.note)
		}, agent.WithDescription("Build and measure the probe program on "+d.module+"."))
	}
	app.RegisterReasoner("synthesize", r.synthesize, agent.WithDescription("Compare the measurements and recommend one option."))

	if err := app.Run(context.Background()); err != nil {
		log.Fatalf("workaf-research: %v", err)
	}
}

type research struct {
	app  *agent.Agent
	root string
}

func (r *research) research(ctx context.Context, in map[string]any) (any, error) {
	question := strings.TrimSpace(str(in, "question"))
	if question == "" {
		question = defaultQuestion
	}
	plan, err := r.app.Call(ctx, "plan", map[string]any{"question": question})
	if err != nil {
		return nil, fmt.Errorf("plan: %w", err)
	}

	// The three investigations run at once.
	results := make([]map[string]any, len(drivers))
	errs := make([]error, len(drivers))
	var wg sync.WaitGroup
	for i, d := range drivers {
		wg.Add(1)
		go func(i int, reasoner string) {
			defer wg.Done()
			results[i], errs[i] = r.app.Call(ctx, reasoner, map[string]any{"question": question, "plan": plan})
		}(i, d.reasoner)
	}
	wg.Wait()
	findings := map[string]any{}
	for i, d := range drivers {
		if errs[i] != nil {
			findings[d.module] = map[string]any{"error": errs[i].Error()}
			continue
		}
		findings[d.module] = results[i]
	}

	answer, err := r.app.Call(ctx, "synthesize", map[string]any{"question": question, "plan": plan, "findings": findings})
	if err != nil {
		return nil, fmt.Errorf("synthesize: %w", err)
	}
	return map[string]any{"question": question, "criteria": plan["criteria"], "findings": findings, "answer": answer}, nil
}

// Plan is the shared yardstick every investigation uses.
type Plan struct {
	Criteria []string `json:"criteria" description:"what to measure, each one checkable by running a command"`
	Probe    string   `json:"probe" description:"what the probe program does, precisely enough that three people write the same program"`
	Commands []string `json:"commands" description:"the shell commands that build and measure the probe, with DRIVER standing for the module"`
}

func (r *research) plan(ctx context.Context, in map[string]any) (any, error) {
	dir, err := os.MkdirTemp(r.root, "plan-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(dir)
	prompt := fmt.Sprintf(`Question: %s

Plan a fair, measurable comparison. Do not answer the question. Write the
criteria, describe ONE small probe program (open a database file, create a
table, insert 10,000 rows in a transaction, query them back), and the commands
that build it natively and cross-compile it for linux/amd64 and windows/amd64,
then report binary size and run time. Keep it to what runs in a few minutes.`, str(in, "question"))
	var p Plan
	return harnessInto(ctx, r.app, prompt, dir, &p)
}

// Measurement is one driver's result.
type Measurement struct {
	Module        string            `json:"module"`
	Builds        map[string]string `json:"builds" description:"target (darwin/arm64, linux/amd64, windows/amd64) to ok or the error, verbatim and short"`
	BinaryBytes   int64             `json:"binary_bytes" description:"size of the native binary"`
	InsertQueryMS int64             `json:"insert_query_ms" description:"wall time of the probe run, natively"`
	NeedsCgo      bool              `json:"needs_cgo"`
	Notes         string            `json:"notes" description:"anything that surprised you, with the command that showed it"`
}

func (r *research) investigate(ctx context.Context, in map[string]any, module, sqlName, note string) (any, error) {
	dir, err := os.MkdirTemp(r.root, "probe-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(dir)
	prompt := fmt.Sprintf(`Question: %s

The plan every investigation follows:
%s

You measure ONE driver: %s (database/sql name %q; %s).
This directory is empty. Create a Go module, write the probe program on this
driver, run the plan's commands, and report what you measured. Report a build
that fails as the failure, not as a number.`, str(in, "question"), render(in["plan"]), module, sqlName, note)
	var m Measurement
	return harnessInto(ctx, r.app, prompt, dir, &m)
}

// Answer is the recommendation.
type Answer struct {
	Recommendation string `json:"recommendation" description:"the driver to use and why, in two or three sentences"`
	Table          string `json:"table" description:"a Markdown table of the three drivers against the criteria, from the measurements only"`
	Caveats        string `json:"caveats"`
}

func (r *research) synthesize(ctx context.Context, in map[string]any) (any, error) {
	dir, err := os.MkdirTemp(r.root, "synth-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(dir)
	prompt := fmt.Sprintf(`Question: %s

Criteria and plan:
%s

What was measured, one entry per driver:
%s

Recommend one driver. Use only these measurements; where one is missing or a
build failed, say so rather than filling it in.`, str(in, "question"), render(in["plan"]), render(in["findings"]))
	var a Answer
	return harnessInto(ctx, r.app, prompt, dir, &a)
}
