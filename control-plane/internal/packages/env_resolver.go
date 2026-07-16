package packages

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"

	"golang.org/x/term"

	"github.com/Agent-Field/agentfield/control-plane/internal/ui"
)

// Prompter asks the user to supply a value for a missing required variable.
// It is an interface so tests can supply values without a TTY.
type Prompter interface {
	// Interactive reports whether prompting is possible (a real terminal).
	Interactive() bool
	// Prompt requests a value for the given variable. Secret types should be
	// read without echo.
	Prompt(v UserEnvironmentVar) (string, error)
	// PromptLine reads a single echoed line (trimmed), used to choose among
	// require_one_of options. It is never used for secret values.
	PromptLine(prompt string) (string, error)
}

// EnvResolver resolves an agent node's environment variables, prompting for and
// persisting missing required secrets via the SecretStore.
type EnvResolver struct {
	Store    *SecretStore
	NodeName string
	Prompter Prompter
}

// Resolve produces the KEY=VALUE map to inject into the node process.
//
// Resolution order per variable:
//  1. process environment (already exported) — used as-is, never persisted
//  2. node-scoped secret store, then global-scoped store
//  3. manifest default
//  4. (required only) prompt the user, validate, and persist encrypted
//
// require_one_of groups are satisfied when at least one option resolves; when
// none is set in an interactive session the user is asked to fill one in.
//
// Required variables (or groups) that remain unset after prompting (or in a
// non-interactive session) produce an error naming what is missing.
func (r *EnvResolver) Resolve(env UserEnvironmentConfig) (map[string]string, error) {
	resolved := map[string]string{}
	var missing []string
	var missingGroups []RequireOneOfGroup

	for _, v := range env.Required {
		val, err := r.lookup(v)
		if err != nil {
			return nil, err
		}
		if val != "" {
			resolved[v.Name] = val
			continue
		}
		got, err := r.promptAndStore(v)
		if err != nil {
			return nil, err
		}
		if got == "" {
			missing = append(missing, v.Name)
		} else {
			resolved[v.Name] = got
		}
	}

	for _, g := range env.RequireOneOf {
		satisfied, err := r.resolveGroupFromSources(g, resolved)
		if err != nil {
			return nil, err
		}
		if satisfied {
			continue
		}
		if r.Prompter == nil || !r.Prompter.Interactive() {
			missingGroups = append(missingGroups, g)
			continue
		}
		got, err := r.promptGroup(g, resolved)
		if err != nil {
			return nil, err
		}
		if !got {
			missingGroups = append(missingGroups, g)
		}
	}

	for _, v := range env.Optional {
		val, err := r.lookup(v)
		if err != nil {
			return nil, err
		}
		if val != "" {
			resolved[v.Name] = val
		}
	}

	// Node-scoped secrets the manifest does not declare are still injected: a
	// user who ran `af secrets set KEY --node <name>` has explicitly scoped
	// that value to this node, and silently skipping it (the previous
	// behavior) made operational overrides — AGENTFIELD_HARNESS_IDLE_SECONDS,
	// provider tuning, and the like — appear stored while never reaching the
	// process. Declared variables keep their full precedence chain above; for
	// undeclared ones the process environment still wins, and global secrets
	// stay manifest-driven so a shared key never leaks into every node.
	if r.NodeName != "" && r.NodeName != globalScope {
		nodeKeys, err := r.Store.List(r.NodeName)
		if err != nil {
			return nil, err
		}
		for _, key := range nodeKeys {
			if _, done := resolved[key]; done {
				continue
			}
			if val, ok := os.LookupEnv(key); ok && val != "" {
				resolved[key] = val
				continue
			}
			if val, ok, err := r.Store.Get(r.NodeName, key); err != nil {
				return nil, err
			} else if ok && val != "" {
				resolved[key] = val
			}
		}
	}

	if len(missing) > 0 || len(missingGroups) > 0 {
		return nil, missingEnvError(missing, missingGroups)
	}
	return resolved, nil
}

// lookup resolves a variable from the process environment, the secret store
// (node overriding global), then its manifest default — without prompting.
// It returns "" when the variable is unset from all of those sources.
func (r *EnvResolver) lookup(v UserEnvironmentVar) (string, error) {
	if val, ok := os.LookupEnv(v.Name); ok && val != "" {
		return val, nil
	}
	if val, ok, err := r.Store.Get(r.NodeName, v.Name); err != nil {
		return "", err
	} else if ok {
		return val, nil
	}
	if v.Default != "" {
		return v.Default, nil
	}
	return "", nil
}

// promptAndStore prompts for a variable (interactive only), persists the value
// encrypted, and returns it. It returns "" when there is no interactive prompt
// or the user skips.
func (r *EnvResolver) promptAndStore(v UserEnvironmentVar) (string, error) {
	if r.Prompter == nil || !r.Prompter.Interactive() {
		return "", nil
	}
	val, err := r.promptAndValidate(v)
	if err != nil {
		return "", err
	}
	if val == "" {
		return "", nil
	}
	if err := r.Store.Set(v.SecretScope(r.NodeName), v.Name, val); err != nil {
		return "", fmt.Errorf("failed to save %s: %w", v.Name, err)
	}
	return val, nil
}

// resolveGroupFromSources injects every option of a require_one_of group that is
// already available (env/store/default) and reports whether at least one was. A
// store read failure aborts, consistent with required/optional resolution.
func (r *EnvResolver) resolveGroupFromSources(g RequireOneOfGroup, resolved map[string]string) (bool, error) {
	found := false
	for _, opt := range g.Options {
		val, err := r.lookup(opt)
		if err != nil {
			return false, err
		}
		if val != "" {
			resolved[opt.Name] = val
			found = true
		}
	}
	return found, nil
}

// promptGroup asks the user to satisfy a require_one_of group. A single-option
// group is prompted for directly; a multi-option group is presented as a
// numbered menu so the user picks one provider, and only the chosen option is
// then prompted for and persisted. Returns true once one option is supplied.
func (r *EnvResolver) promptGroup(g RequireOneOfGroup, resolved map[string]string) (bool, error) {
	if len(g.Options) == 1 {
		return r.storeGroupOption(g.Options[0], resolved)
	}

	desc := g.Description
	if desc == "" {
		desc = "one of the following"
	}
	fmt.Printf("\n  %s\n", ui.Title("This node needs "+desc+" — choose one:"))

	width := 0
	for _, opt := range g.Options {
		if len(opt.Name) > width {
			width = len(opt.Name)
		}
	}
	nums := make([]string, len(g.Options))
	for i, opt := range g.Options {
		nums[i] = strconv.Itoa(i + 1)
		line := fmt.Sprintf("    %s  %-*s", ui.Title("["+nums[i]+"]"), width, opt.Name)
		if opt.Description != "" {
			line += "  " + ui.Muted(opt.Description)
		}
		fmt.Println(line)
	}

	for attempt := 0; attempt < 3; attempt++ {
		choice, err := r.Prompter.PromptLine(fmt.Sprintf("  Enter %s (or press Enter to skip): ", orJoin(nums)))
		if err != nil {
			return false, err
		}
		choice = strings.TrimSpace(choice)
		if choice == "" {
			return false, nil // skipped the whole group
		}
		n, convErr := strconv.Atoi(choice)
		if convErr != nil || n < 1 || n > len(g.Options) {
			fmt.Printf("  %s\n", ui.Muted("please enter "+orJoin(nums)))
			continue
		}
		return r.storeGroupOption(g.Options[n-1], resolved)
	}
	return false, nil
}

// storeGroupOption prompts for a single group option, persisting and recording
// a non-empty value. A blank answer leaves the group unsatisfied.
func (r *EnvResolver) storeGroupOption(opt UserEnvironmentVar, resolved map[string]string) (bool, error) {
	val, err := r.promptAndValidate(opt)
	if err != nil {
		return false, err
	}
	if val == "" {
		return false, nil
	}
	if err := r.Store.Set(opt.SecretScope(r.NodeName), opt.Name, val); err != nil {
		return false, fmt.Errorf("failed to save %s: %w", opt.Name, err)
	}
	resolved[opt.Name] = val
	return true, nil
}

// orJoin renders a human list joined with "or": ["1"] -> "1",
// ["1","2"] -> "1 or 2", ["1","2","3"] -> "1, 2, or 3".
func orJoin(items []string) string {
	switch len(items) {
	case 0:
		return ""
	case 1:
		return items[0]
	case 2:
		return items[0] + " or " + items[1]
	default:
		return strings.Join(items[:len(items)-1], ", ") + ", or " + items[len(items)-1]
	}
}

// missingEnvError formats a single error covering unset required variables and
// any unsatisfied require_one_of groups.
func missingEnvError(missing []string, groups []RequireOneOfGroup) error {
	var parts []string
	if len(missing) > 0 {
		parts = append(parts, "missing required environment variables: "+strings.Join(missing, ", "))
	}
	for _, g := range groups {
		parts = append(parts, "at least one of "+orJoin(g.OptionNames())+" is required")
	}
	return errors.New(strings.Join(parts, "; "))
}

// promptAndValidate prompts until the value satisfies the variable's validation
// regex (if any), or returns the raw value when no pattern is set.
func (r *EnvResolver) promptAndValidate(v UserEnvironmentVar) (string, error) {
	var re *regexp.Regexp
	if v.Validation != "" {
		compiled, err := regexp.Compile(v.Validation)
		if err != nil {
			return "", fmt.Errorf("invalid validation regex for %s: %w", v.Name, err)
		}
		re = compiled
	}

	for attempt := 0; attempt < 3; attempt++ {
		val, err := r.Prompter.Prompt(v)
		if err != nil {
			return "", err
		}
		val = strings.TrimSpace(val)
		if val == "" {
			return "", nil // user skipped
		}
		if re != nil && !re.MatchString(val) {
			fmt.Printf("  value does not match required format (%s), try again\n", v.Validation)
			continue
		}
		return val, nil
	}
	return "", fmt.Errorf("%s: too many invalid attempts", v.Name)
}

// TTYPrompter reads values from the terminal, hiding secret input.
type TTYPrompter struct{}

// Interactive reports whether stdin is a terminal.
func (TTYPrompter) Interactive() bool {
	return term.IsTerminal(int(os.Stdin.Fd()))
}

// Prompt asks for a value on stdin, reading without echo for secret types.
func (TTYPrompter) Prompt(v UserEnvironmentVar) (string, error) {
	label := v.Name
	if v.Description != "" {
		label = fmt.Sprintf("%s (%s)", v.Name, v.Description)
	}

	if v.Type == "secret" {
		fmt.Printf("  Enter %s [hidden]: ", label)
		data, err := term.ReadPassword(int(os.Stdin.Fd()))
		fmt.Println()
		if err != nil {
			return "", fmt.Errorf("failed to read secret: %w", err)
		}
		return string(data), nil
	}

	fmt.Printf("  Enter %s: ", label)
	reader := bufio.NewReader(os.Stdin)
	line, err := reader.ReadString('\n')
	if err != nil && line == "" {
		return "", fmt.Errorf("failed to read input: %w", err)
	}
	return line, nil
}

// PromptLine prints prompt and reads a single echoed line from stdin, trimmed.
func (TTYPrompter) PromptLine(prompt string) (string, error) {
	fmt.Print(prompt)
	reader := bufio.NewReader(os.Stdin)
	line, err := reader.ReadString('\n')
	if err != nil && line == "" {
		return "", fmt.Errorf("failed to read input: %w", err)
	}
	return strings.TrimSpace(line), nil
}
