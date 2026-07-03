package packages

import (
	"bufio"
	"fmt"
	"os"
	"regexp"
	"strings"

	"golang.org/x/term"
)

// Prompter asks the user to supply a value for a missing required variable.
// It is an interface so tests can supply values without a TTY.
type Prompter interface {
	// Interactive reports whether prompting is possible (a real terminal).
	Interactive() bool
	// Prompt requests a value for the given variable. Secret types should be
	// read without echo.
	Prompt(v UserEnvironmentVar) (string, error)
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
// Required variables that remain unset after prompting (or in a non-interactive
// session) are returned as a list of missing names alongside an error.
func (r *EnvResolver) Resolve(env UserEnvironmentConfig) (map[string]string, error) {
	resolved := map[string]string{}
	var missing []string

	resolveOne := func(v UserEnvironmentVar, required bool) error {
		// 1. Process environment wins and is never written to disk.
		if val, ok := os.LookupEnv(v.Name); ok && val != "" {
			resolved[v.Name] = val
			return nil
		}
		// 2. Secret store (node overrides global).
		if val, ok, err := r.Store.Get(r.NodeName, v.Name); err != nil {
			return err
		} else if ok {
			resolved[v.Name] = val
			return nil
		}
		// 3. Manifest default.
		if v.Default != "" {
			resolved[v.Name] = v.Default
			return nil
		}
		if !required {
			return nil // optional and unset: leave it out
		}
		// 4. Prompt (required only).
		if r.Prompter == nil || !r.Prompter.Interactive() {
			missing = append(missing, v.Name)
			return nil
		}
		val, err := r.promptAndValidate(v)
		if err != nil {
			return err
		}
		if val == "" {
			missing = append(missing, v.Name)
			return nil
		}
		if err := r.Store.Set(v.SecretScope(r.NodeName), v.Name, val); err != nil {
			return fmt.Errorf("failed to save %s: %w", v.Name, err)
		}
		resolved[v.Name] = val
		return nil
	}

	for _, v := range env.Required {
		if err := resolveOne(v, true); err != nil {
			return nil, err
		}
	}
	for _, v := range env.Optional {
		if err := resolveOne(v, false); err != nil {
			return nil, err
		}
	}

	if len(missing) > 0 {
		return nil, fmt.Errorf("missing required environment variables: %s", strings.Join(missing, ", "))
	}
	return resolved, nil
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
