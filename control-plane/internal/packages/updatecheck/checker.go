// Package updatecheck performs cheap, read-only checks for newer git commits.
package updatecheck

import (
	"context"
	"fmt"
	"os/exec"
	"strings"
	"sync"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
)

const defaultTimeout = 10 * time.Second

type Status string

const (
	StatusCurrent   Status = "current"
	StatusAvailable Status = "available"
	StatusPinned    Status = "pinned"
	StatusUnknown   Status = "unknown"
	StatusDeferred  Status = "deferred"
	StatusError     Status = "error"
	StatusFailed    Status = "failed"
)

type Update struct {
	Status       Status    `json:"status"`
	LatestCommit string    `json:"latest_commit"`
	CheckedAt    time.Time `json:"checked_at"`
	Message      string    `json:"message"`
}

type Entry struct {
	ID              string
	Name            string
	Source          string
	Ref             string
	InstalledCommit string
}

type Result struct {
	ID              string `json:"id"`
	Name            string `json:"name"`
	InstalledCommit string `json:"installed_commit"`
	Update          Update `json:"update"`
}

// Runner is intentionally the smallest possible command boundary so checker
// tests never need a repository or the network.
type Runner interface {
	Run(ctx context.Context, args ...string) ([]byte, error)
}

type execRunner struct{}

func (execRunner) Run(ctx context.Context, args ...string) ([]byte, error) {
	return exec.CommandContext(ctx, "git", args...).CombinedOutput()
}

type Checker struct {
	runner  Runner
	timeout time.Duration
	now     func() time.Time
	checkMu sync.Mutex
	// observeCheckContext is a test seam for asserting the serialized overall
	// budget without weakening the per-package runner timeout.
	observeCheckContext func(context.Context)

	mu    sync.RWMutex
	cache map[string]Update
}

func NewChecker(runner Runner) *Checker {
	if runner == nil {
		runner = execRunner{}
	}
	return &Checker{
		runner:  runner,
		timeout: defaultTimeout,
		now:     time.Now,
		cache:   make(map[string]Update),
	}
}

func (c *Checker) Check(ctx context.Context, entries []Entry) []Result {
	c.checkMu.Lock()
	defer c.checkMu.Unlock()
	return c.checkLocked(ctx, entries)
}

// CheckWithTimeout serializes checks before starting the caller's aggregate
// budget. A queued request therefore receives its full budget once it owns the
// checker, while checkOne still caps each git command independently.
func (c *Checker) CheckWithTimeout(parent context.Context, entries []Entry, timeout time.Duration) []Result {
	c.checkMu.Lock()
	defer c.checkMu.Unlock()
	ctx, cancel := context.WithTimeout(parent, timeout)
	defer cancel()
	if c.observeCheckContext != nil {
		c.observeCheckContext(ctx)
	}
	return c.checkLocked(ctx, entries)
}

func (c *Checker) checkLocked(ctx context.Context, entries []Entry) []Result {
	checkedAt := c.now().UTC()
	results := make([]Result, 0, len(entries))
	for _, entry := range entries {
		result := Result{ID: entry.ID, Name: entry.Name, InstalledCommit: entry.InstalledCommit}
		previous := c.Cached(entry.ID)
		result.Update = c.checkOne(ctx, entry, checkedAt)
		if previous.Status == StatusFailed && result.Update.Status == StatusAvailable &&
			previous.LatestCommit != "" && previous.LatestCommit == result.Update.LatestCommit {
			result.Update.Status = StatusFailed
			result.Update.Message = previous.Message
		}
		// A transient check failure carries no remote commit identity and must
		// not erase the memo that suppresses a deterministic failed commit.
		preserveFailedMemo := previous.Status == StatusFailed && result.Update.Status == StatusError
		if !preserveFailedMemo && !(result.Update.Status == StatusError && ctx.Err() != nil) {
			c.Set(entry.ID, result.Update)
		}
		results = append(results, result)
	}
	return results
}

func (c *Checker) checkOne(parent context.Context, entry Entry, checkedAt time.Time) Update {
	update := Update{Status: StatusError, CheckedAt: checkedAt}
	info, err := packages.ParseGitURL(strings.TrimSpace(entry.Source))
	if err != nil || strings.TrimSpace(info.CloneURL) == "" {
		update.Message = "invalid recorded git source"
		return update
	}
	ref := strings.TrimSpace(entry.Ref)
	if ref == "" {
		ref = info.Ref
	}
	queryRef := ref
	if queryRef == "" {
		queryRef = "HEAD"
	}

	ctx, cancel := context.WithTimeout(parent, c.timeout)
	defer cancel()
	output, err := c.runner.Run(ctx, "ls-remote", "--quiet", info.CloneURL, queryRef)
	if err != nil {
		message := strings.TrimSpace(string(output))
		if message == "" {
			message = err.Error()
		} else {
			message = fmt.Sprintf("%v: %s", err, message)
		}
		update.Message = message
		return update
	}
	fields := strings.Fields(string(output))
	if len(fields) == 0 {
		update.Message = fmt.Sprintf("git ls-remote returned no commit for %s", queryRef)
		return update
	}
	update.LatestCommit = fields[0]
	if ref != "" {
		update.Status = StatusPinned
		update.Message = "source is pinned to an explicit ref"
		return update
	}
	if strings.TrimSpace(entry.InstalledCommit) == "" {
		update.Status = StatusAvailable
		update.Message = "installed commit is unknown; one update is required to establish provenance"
		return update
	}
	if entry.InstalledCommit == update.LatestCommit {
		update.Status = StatusCurrent
		return update
	}
	update.Status = StatusAvailable
	return update
}

func (c *Checker) Cached(id string) Update {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if update, ok := c.cache[id]; ok {
		return update
	}
	return Update{Status: StatusUnknown}
}

func (c *Checker) Set(id string, update Update) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.cache[id] = update
}

func (c *Checker) Clear(id string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.cache, id)
}
