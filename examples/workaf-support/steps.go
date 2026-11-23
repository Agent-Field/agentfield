package main

import (
	"bufio"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"

	"github.com/Agent-Field/agentfield/sdk/go/harness"
)

// products maps the words a customer uses for a product to the repository
// that builds it. A real deployment would read this from its own catalogue.
var products = []struct {
	words []string
	repo  string
}{
	{[]string{"ledger", "total", "entries", "refund", "cents"}, "santoshkumarradha/factory-demo"},
}

// kinds are the ticket kinds triage knows, checked in order.
var kinds = []struct {
	kind  string
	words []string
}{
	{"bug", []string{"wrong", "broken", "error", "crash", "panic", "incorrect", "bug", "fails", "negative"}},
	{"how-to", []string{"how do i", "how can i", "is there a way", "can i"}},
	{"feature", []string{"would be nice", "feature", "please add", "support for"}},
}

// triage is plain code: no model, the same answer every time.
func (s *support) triage(ctx context.Context, in map[string]any) (any, error) {
	text := strings.ToLower(str(in, "ticket"))
	out := map[string]any{"kind": "question"}
	for _, k := range kinds {
		if containsAny(text, k.words) {
			out["kind"] = k.kind
			break
		}
	}
	best := 0
	for _, p := range products {
		if n := countAny(text, p.words); n > best {
			best, out["repo"] = n, p.repo
		}
	}
	return out, nil
}

// searchDocs is plain code: it greps the repository's README and Markdown
// docs, plus Go doc comments, for the words in the query.
func (s *support) searchDocs(ctx context.Context, in map[string]any) (any, error) {
	dir, err := s.repos.checkout(ctx, str(in, "repo"))
	if err != nil {
		return nil, err
	}
	terms := keywords(str(in, "query"))
	type hit struct {
		File string `json:"file"`
		Line int    `json:"line"`
		Text string `json:"text"`
	}
	var hits []hit
	_ = filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			if d.Name() == ".git" {
				return filepath.SkipDir
			}
			return nil
		}
		isDoc := strings.HasSuffix(path, ".md")
		isGo := strings.HasSuffix(path, ".go") && !strings.HasSuffix(path, "_test.go")
		if !isDoc && !isGo {
			return nil
		}
		f, err := os.Open(path)
		if err != nil {
			return nil
		}
		defer f.Close()
		rel, _ := filepath.Rel(dir, path)
		sc := bufio.NewScanner(f)
		for n := 1; sc.Scan(); n++ {
			line := sc.Text()
			if isGo && !strings.HasPrefix(strings.TrimSpace(line), "//") {
				continue
			}
			if containsAny(strings.ToLower(line), terms) {
				hits = append(hits, hit{rel, n, strings.TrimSpace(line)})
			}
		}
		return nil
	})
	if len(hits) > 20 {
		hits = hits[:20]
	}
	return map[string]any{"terms": terms, "hits": hits}, nil
}

// Findings is what investigate's coding agent must return.
type Findings struct {
	RootCause  string   `json:"root_cause" description:"one or two sentences: why the customer sees what they see"`
	Files      []string `json:"files" description:"repository files involved, with line numbers where known"`
	Reproduce  string   `json:"reproduce" description:"a command that shows the problem"`
	Confidence string   `json:"confidence" description:"high, medium or low"`
}

// investigate looks the ticket up in the docs, then has a coding agent read the
// repository to find the cause. It reads; it changes nothing.
func (s *support) investigate(ctx context.Context, in map[string]any) (any, error) {
	repo := str(in, "repo")
	ticket := str(in, "ticket")
	dir, err := s.repos.checkout(ctx, repo)
	if err != nil {
		return nil, err
	}
	docs, err := s.app.Call(ctx, "search_docs", map[string]any{"repo": repo, "query": ticket})
	if err != nil {
		return nil, fmt.Errorf("search_docs: %w", err)
	}

	prompt := fmt.Sprintf(`A customer filed this support ticket about %s:

%s

What the README and docs say about it (grep hits):
%s

Find the root cause in this repository. Read the code and run it or its tests
to confirm the behaviour the customer describes. Do not change any file.`, repo, ticket, render(docs["hits"]))

	var f Findings
	schema, err := harness.StructToJSONSchema(f)
	if err != nil {
		return nil, err
	}
	res, err := s.app.Harness(ctx, prompt, schema, &f, harness.Options{Cwd: dir, ProjectDir: dir})
	if err != nil {
		return nil, err
	}
	if res.IsError {
		return nil, fmt.Errorf("investigate: %s", res.ErrorMessage)
	}
	return map[string]any{
		"root_cause": f.RootCause,
		"files":      f.Files,
		"reproduce":  f.Reproduce,
		"confidence": f.Confidence,
		"docs_hits":  docs["hits"],
	}, nil
}

// Reply is what draft_reply's coding agent must return.
type Reply struct {
	Reply        string `json:"reply" description:"the email to the customer: plain, friendly, no internal jargon, under 150 words"`
	SuggestedFix string `json:"suggested_fix" description:"for the engineering team: what to change and why, in a few sentences"`
}

// draftReply has a coding agent make the fix in a scratch copy, so the
// suggestion is real code that passes the tests, and write the customer reply.
// The diff is read back with git; nothing is pushed.
func (s *support) draftReply(ctx context.Context, in map[string]any) (any, error) {
	repo := str(in, "repo")
	dir, err := s.repos.scratch(ctx, repo)
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(dir)

	prompt := fmt.Sprintf(`A customer filed this ticket about %s:

%s

The investigation found:
%s

1. Make the smallest code change in this repository that fixes it, add a test
   for it, and run the tests until they pass.
2. Write the reply to the customer and a short note for the engineering team.`,
		repo, str(in, "ticket"), render(in["findings"]))

	var r Reply
	schema, err := harness.StructToJSONSchema(r)
	if err != nil {
		return nil, err
	}
	res, err := s.app.Harness(ctx, prompt, schema, &r, harness.Options{Cwd: dir, ProjectDir: dir})
	if err != nil {
		return nil, err
	}
	if res.IsError {
		return nil, fmt.Errorf("draft_reply: %s", res.ErrorMessage)
	}
	diff, _ := exec.CommandContext(ctx, "git", "-C", dir, "diff").Output()
	return map[string]any{"reply": r.Reply, "suggested_fix": r.SuggestedFix, "diff": string(diff)}, nil
}

// ── repositories ────────────────────────────────────────────────────────────

// repoCache keeps one clone per repository and hands out scratch copies.
type repoCache struct {
	mu    sync.Mutex
	root  string
	token string
}

func newRepoCache(root, token string) *repoCache {
	return &repoCache{root: root, token: token}
}

var repoName = regexp.MustCompile(`^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$`)

func (c *repoCache) dir() (string, error) {
	if c.root == "" {
		d, err := os.MkdirTemp("", "workaf-support-")
		if err != nil {
			return "", err
		}
		c.root = d
	}
	return c.root, os.MkdirAll(c.root, 0o700)
}

// checkout returns a shared, read-only-by-convention clone of repo.
func (c *repoCache) checkout(ctx context.Context, repo string) (string, error) {
	if !repoName.MatchString(repo) {
		return "", fmt.Errorf("%q is not owner/name", repo)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	root, err := c.dir()
	if err != nil {
		return "", err
	}
	dir := filepath.Join(root, "clones", strings.ReplaceAll(repo, "/", "__"))
	if _, err := os.Stat(filepath.Join(dir, ".git")); err == nil {
		return dir, c.git(ctx, dir, "pull", "--ff-only", "-q")
	}
	return dir, c.git(ctx, "", "clone", "-q", "--depth", "50", "https://github.com/"+repo+".git", dir)
}

// scratch returns a private copy of repo the caller may change and must remove.
func (c *repoCache) scratch(ctx context.Context, repo string) (string, error) {
	src, err := c.checkout(ctx, repo)
	if err != nil {
		return "", err
	}
	dst, err := os.MkdirTemp(c.root, "scratch-")
	if err != nil {
		return "", err
	}
	if out, err := exec.CommandContext(ctx, "git", "clone", "-q", "--local", src, dst+"/r").CombinedOutput(); err != nil {
		return "", fmt.Errorf("git clone --local: %v: %s", err, out)
	}
	return dst + "/r", nil
}

// git runs git with the token as an HTTP header passed through the
// environment, so it is never in argv, a URL or the clone's config.
func (c *repoCache) git(ctx context.Context, dir string, args ...string) error {
	if dir != "" {
		args = append([]string{"-C", dir}, args...)
	}
	cmd := exec.CommandContext(ctx, "git", args...)
	cmd.Env = append(os.Environ(), "GIT_TERMINAL_PROMPT=0")
	if c.token != "" {
		cmd.Env = append(cmd.Env,
			"GIT_CONFIG_COUNT=1",
			"GIT_CONFIG_KEY_0=http.https://github.com/.extraheader",
			"GIT_CONFIG_VALUE_0=Authorization: Basic "+basicAuth("x-access-token", c.token))
	}
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("git %s: %v: %s", args[0], err, strings.TrimSpace(string(out)))
	}
	return nil
}

// ── small helpers ───────────────────────────────────────────────────────────

var stop = map[string]bool{"the": true, "and": true, "after": true, "when": true, "with": true, "that": true, "this": true, "from": true, "have": true, "what": true, "your": true, "about": true, "into": true, "then": true, "there": true, "wrong": true}

// keywords are the query's distinct words of four or more letters.
func keywords(q string) []string {
	seen := map[string]bool{}
	for _, w := range regexp.MustCompile(`[a-z]{4,}`).FindAllString(strings.ToLower(q), -1) {
		w = strings.TrimSuffix(w, "s")
		if !stop[w] {
			seen[w] = true
		}
	}
	out := make([]string, 0, len(seen))
	for w := range seen {
		out = append(out, w)
	}
	sort.Strings(out)
	return out
}

// render is a value as indented JSON for a prompt.
func render(v any) string {
	raw, err := json.MarshalIndent(v, "", "  ")
	if err != nil || string(raw) == "null" {
		return "(nothing)"
	}
	return string(raw)
}

func basicAuth(user, pass string) string {
	return base64.StdEncoding.EncodeToString([]byte(user + ":" + pass))
}

func containsAny(s string, words []string) bool { return countAny(s, words) > 0 }

func countAny(s string, words []string) int {
	n := 0
	for _, w := range words {
		if strings.Contains(s, w) {
			n++
		}
	}
	return n
}
