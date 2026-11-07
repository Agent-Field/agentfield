// workaf-research answers an engineering question by trying things, not by
// recalling them.
//
//	research                 entry point: {"question": "..."}, or a WorkAF job's {"title", "body"}
//	  plan                   a coding agent names the options, the criteria and one probe program
//	  investigate            ┐
//	  investigate            ├ in parallel, one per option the plan named (two to four)
//	  investigate            ┘
//	  synthesize             a coding agent compares the measurements and recommends one
//
// The question comes from the call: "question", else the ticket's title and
// body. Only a call with neither uses the default ("which Go SQLite driver
// should a small CLI use?"). Each investigate call is its own execution,
// called at the same time, so the run's tree shows them as siblings.
//
// Configuration is environment only:
//
//	AGENTFIELD_URL, AGENTFIELD_API_KEY (or _FILE), AGENT_NODE_ID (default research),
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

// mostOptions bounds how many investigations one question runs at once.
const mostOptions = 4

func main() {
	listen := envOr("AGENT_LISTEN", "127.0.0.1:8003")
	app, err := agent.New(agent.Config{
		NodeID:        envOr("AGENT_NODE_ID", "research"),
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
	app.RegisterReasoner("investigate", r.investigate, agent.WithDescription("Build and measure the probe program on one option the plan named."))
	app.RegisterReasoner("synthesize", r.synthesize, agent.WithDescription("Compare the measurements and recommend one option."))

	if err := app.Run(context.Background()); err != nil {
		log.Fatalf("workaf-research: %v", err)
	}
}

type research struct {
	app  *agent.Agent
	root string
}

// question is what the run is asked. A direct call sends {"question"}; a
// WorkAF job sends its ticket as {"job", "title", "body", "event"}, and the
// ticket IS the question. Only a call with neither gets the default.
func question(in map[string]any) string {
	if q := strings.TrimSpace(str(in, "question")); q != "" {
		return q
	}
	title, body := strings.TrimSpace(str(in, "title")), strings.TrimSpace(str(in, "body"))
	if ev, ok := in["event"].(map[string]any); ok {
		if title == "" {
			title = strings.TrimSpace(str(ev, "title"))
		}
		if body == "" {
			body = strings.TrimSpace(str(ev, "body"))
		}
	}
	if q := strings.TrimSpace(title + "\n\n" + body); q != "" {
		return q
	}
	return defaultQuestion
}

func (r *research) research(ctx context.Context, in map[string]any) (any, error) {
	question := question(in)
	planned, err := r.app.Call(ctx, "plan", map[string]any{"question": question})
	if err != nil {
		return nil, fmt.Errorf("plan: %w", err)
	}
	options := optionsOf(planned)
	if len(options) == 0 {
		return nil, fmt.Errorf("plan: it named no options to compare")
	}

	// One investigation per option the plan named, all at once.
	results := make([]map[string]any, len(options))
	errs := make([]error, len(options))
	var wg sync.WaitGroup
	for i, o := range options {
		wg.Add(1)
		go func(i int, o map[string]any) {
			defer wg.Done()
			// "subject" names this call among its siblings: WorkAF labels each
			// parallel investigation by it ("Investigate goccy/go-json").
			results[i], errs[i] = r.app.Call(ctx, "investigate", map[string]any{"subject": str(o, "name"), "question": question, "plan": planned, "option": o})
		}(i, o)
	}
	wg.Wait()
	findings := map[string]any{}
	for i, o := range options {
		name := str(o, "name")
		if errs[i] != nil {
			findings[name] = map[string]any{"error": errs[i].Error()}
			continue
		}
		findings[name] = results[i]
	}

	answer, err := r.app.Call(ctx, "synthesize", map[string]any{"question": question, "plan": planned, "findings": findings})
	if err != nil {
		return nil, fmt.Errorf("synthesize: %w", err)
	}
	return map[string]any{"question": question, "criteria": planned["criteria"], "findings": findings, "answer": answer}, nil
}

// optionsOf is the plan's options, named, without repeats, at most mostOptions.
func optionsOf(planned map[string]any) []map[string]any {
	raw, _ := planned["options"].([]any)
	var out []map[string]any
	seen := map[string]bool{}
	for _, v := range raw {
		o, ok := v.(map[string]any)
		name := strings.TrimSpace(str(o, "name"))
		if !ok || name == "" || seen[name] {
			continue
		}
		seen[name] = true
		out = append(out, o)
		if len(out) == mostOptions {
			break
		}
	}
	return out
}

// Option is one candidate the question asks about.
type Option struct {
	Name string `json:"name" description:"the option as the question names it, e.g. a library's import path"`
	Note string `json:"note" description:"one line on what is distinctive about it, or how to set it up"`
}

// Plan is the shared yardstick every investigation uses.
type Plan struct {
	Options  []Option `json:"options" description:"the two to four options the question asks to compare, the ones it names first"`
	Criteria []string `json:"criteria" description:"what to measure, each one checkable by running a command"`
	Probe    string   `json:"probe" description:"what the probe program does, precisely enough that everyone writes the same program"`
	Commands []string `json:"commands" description:"the shell commands that build and measure the probe, with OPTION standing for the option"`
}

func (r *research) plan(ctx context.Context, in map[string]any) (any, error) {
	dir, err := os.MkdirTemp(r.root, "plan-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(dir)
	prompt := fmt.Sprintf(`Question: %s

Plan a fair, measurable comparison of the options this question is about. Do
not answer it. Name the two to four options it asks you to compare (the ones it
names, first; if it names none, the obvious candidates), the criteria it cares
about, ONE small probe program that exercises exactly what the question is
about on any one option, and the commands that build and measure that probe.
Keep it to what runs in a few minutes on one machine.`, str(in, "question"))
	var p Plan
	return harnessInto(ctx, r.app, prompt, dir, &p)
}

// Measurement is one option's result.
type Measurement struct {
	Option   string            `json:"option"`
	Builds   map[string]string `json:"builds" description:"each build or run command to ok or the error, verbatim and short"`
	Measured map[string]string `json:"measured" description:"each criterion to what was measured, with its unit"`
	Notes    string            `json:"notes" description:"anything that surprised you, with the command that showed it"`
}

func (r *research) investigate(ctx context.Context, in map[string]any) (any, error) {
	dir, err := os.MkdirTemp(r.root, "probe-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(dir)
	o, _ := in["option"].(map[string]any)
	prompt := fmt.Sprintf(`Question: %s

The plan every investigation follows:
%s

You measure ONE option: %s (%s).
This directory is empty. Write the probe program on this option, run the
plan's commands, and report what you measured. Report a build that fails as
the failure, not as a number. Work only inside this directory: install nothing
system-wide (no brew, no apt, no docker pulls); if a step needs a tool that is
not already here, report that as the result.`, str(in, "question"), render(in["plan"]), str(o, "name"), str(o, "note"))
	var m Measurement
	return harnessInto(ctx, r.app, prompt, dir, &m)
}

// Answer is the recommendation.
type Answer struct {
	Recommendation string `json:"recommendation" description:"the option to use and why, in two or three sentences"`
	Table          string `json:"table" description:"a Markdown table of the options against the criteria, from the measurements only"`
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

What was measured, one entry per option:
%s

Recommend one option. Use only these measurements; where one is missing or a
build failed, say so rather than filling it in.`, str(in, "question"), render(in["plan"]), render(in["findings"]))
	var a Answer
	return harnessInto(ctx, r.app, prompt, dir, &a)
}
