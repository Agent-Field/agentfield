package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/spf13/cobra"
)

// HarnessProviderHealth is the machine-readable result of a provider preflight.
type HarnessProviderHealth struct {
	Provider       string   `json:"provider"`
	Binary         string   `json:"binary,omitempty"`
	Installed      bool     `json:"installed"`
	Version        string   `json:"version,omitempty"`
	Auth           string   `json:"auth"`
	Usable         bool     `json:"usable"`
	InstallCommand string   `json:"install_command"`
	AuthEnvVars    []string `json:"auth_env_vars"`
	Issues         []string `json:"issues"`
}

type harnessProviderSpec struct {
	Name           string
	Binary         string
	InstallCommand string
	AuthEnvVars    []string
	// VersionArgs are tried in order until one produces output; empty means
	// {"--version"}.
	VersionArgs [][]string
}

var harnessProviderSpecs = []harnessProviderSpec{
	{Name: "aforge", Binary: "aforge", InstallCommand: "af aforge ensure", AuthEnvVars: []string{"OPENROUTER_API_KEY"}, VersionArgs: [][]string{{"version"}, {"--version"}}},
	// claude-code has no Binary: the Python provider runs on the
	// claude_agent_sdk pip package (which bundles its own CLI), not on a
	// globally installed `claude` binary. See claudeCodeHealth.
	{Name: "claude-code", InstallCommand: "pip install 'agentfield[harness-claude]'", AuthEnvVars: []string{"ANTHROPIC_API_KEY"}},
	{Name: "codex", Binary: "codex", InstallCommand: "npm install -g @openai/codex", AuthEnvVars: []string{"OPENAI_API_KEY"}},
	{Name: "gemini", Binary: "gemini", InstallCommand: "npm install -g @google/gemini-cli", AuthEnvVars: []string{"GEMINI_API_KEY", "GOOGLE_API_KEY"}},
	{Name: "opencode", Binary: "opencode", InstallCommand: "curl -fsSL https://opencode.ai/install | bash", AuthEnvVars: []string{}},
}

// NewHarnessCommand builds harness-related environment checks.
func NewHarnessCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:           "harness",
		Short:         "Inspect and manage coding-agent harness providers",
		SilenceUsage:  true,
		SilenceErrors: true,
	}
	cmd.AddCommand(newHarnessDoctorCommand())
	return cmd
}

func newHarnessDoctorCommand() *cobra.Command {
	var providers []string
	var jsonOut bool
	cmd := &cobra.Command{
		Use:   "doctor",
		Short: "Verify harness provider binaries, versions, and authentication",
		RunE: func(cmd *cobra.Command, args []string) error {
			reports, err := buildHarnessDoctorReports(providers)
			if err != nil {
				return err
			}
			if jsonOut {
				enc := json.NewEncoder(cmd.OutOrStdout())
				enc.SetIndent("", "  ")
				if err := enc.Encode(reports); err != nil {
					return err
				}
			} else {
				printHarnessDoctorReports(cmd, reports)
			}
			for _, report := range reports {
				if !report.Usable {
					return fmt.Errorf("requested harness provider is unavailable: %s", report.Provider)
				}
			}
			return nil
		},
	}
	cmd.Flags().StringSliceVar(&providers, "provider", nil, "Provider(s) to check: aforge, claude-code, codex, gemini, opencode")
	cmd.Flags().BoolVar(&jsonOut, "json", false, "Output structured JSON")
	return cmd
}

func buildHarnessDoctorReports(requested []string) ([]HarnessProviderHealth, error) {
	selected := map[string]bool{}
	for _, name := range requested {
		selected[strings.TrimSpace(name)] = true
	}
	if len(selected) > 0 {
		for name := range selected {
			if findHarnessProviderSpec(name) == nil {
				return nil, fmt.Errorf("unknown harness provider %q", name)
			}
		}
	}

	reports := make([]HarnessProviderHealth, 0, len(harnessProviderSpecs))
	for _, spec := range harnessProviderSpecs {
		if len(selected) > 0 && !selected[spec.Name] {
			continue
		}
		if spec.Name == "claude-code" {
			reports = append(reports, claudeCodeHealth(spec))
			continue
		}
		tool := probeHarnessBinary(spec)
		issues := []string{}
		usable := tool.Available && tool.Version != ""
		if !tool.Available {
			issues = append(issues, "binary_not_found")
		} else if tool.Version == "" {
			issues = append(issues, "version_probe_failed")
		}
		reports = append(reports, HarnessProviderHealth{
			Provider:       spec.Name,
			Binary:         tool.Path,
			Installed:      tool.Available,
			Version:        tool.Version,
			Auth:           harnessAuthStatus(spec.AuthEnvVars),
			Usable:         usable,
			InstallCommand: spec.InstallCommand,
			AuthEnvVars:    append([]string{}, spec.AuthEnvVars...),
			Issues:         issues,
		})
	}
	return reports, nil
}

// probeHarnessBinary resolves a provider binary and asks it for a version,
// trying each candidate argument set in order.
func probeHarnessBinary(spec harnessProviderSpec) ToolStatus {
	path, err := exec.LookPath(spec.Binary)
	if err != nil {
		// `af aforge ensure` installs into AgentField's own bin directory, which
		// the current shell may not have re-read into PATH yet. Looking there
		// directly is what stops the doctor from calling a binary it just
		// installed "not found".
		home := os.Getenv("AGENTFIELD_HOME")
		if home == "" {
			userHome, homeErr := os.UserHomeDir()
			if homeErr != nil {
				return ToolStatus{}
			}
			home = filepath.Join(userHome, ".agentfield")
		}
		binary := spec.Binary
		if runtime.GOOS == "windows" {
			binary += ".exe"
		}
		candidate := filepath.Join(home, "bin", binary)
		info, statErr := os.Stat(candidate)
		if statErr != nil || !info.Mode().IsRegular() || (runtime.GOOS != "windows" && info.Mode().Perm()&0o111 == 0) {
			return ToolStatus{}
		}
		path = candidate
	}

	argsSets := spec.VersionArgs
	if len(argsSets) == 0 {
		argsSets = [][]string{{"--version"}}
	}
	status := ToolStatus{Available: true, Path: path}
	for _, args := range argsSets {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		out, runErr := exec.CommandContext(ctx, path, args...).CombinedOutput()
		cancel()
		// A non-zero exit is not a version, however much it printed: CLIs
		// answer an unrecognised version flag with their whole usage text on a
		// non-zero exit, and accepting that output would file the usage banner
		// as the installed version.
		if runErr != nil {
			continue
		}
		if version := strings.Split(strings.TrimSpace(string(out)), "\n")[0]; version != "" {
			status.Version = version
			break
		}
	}
	return status
}

// claudeWrapperProbe asks a Python interpreter whether the claude_agent_sdk
// package is importable, exiting zero either way so a non-zero exit always
// means the interpreter itself is unusable.
const claudeWrapperProbe = "import importlib.util, sys; sys.stdout.write('ok' if importlib.util.find_spec('claude_agent_sdk') else 'missing')"

var pythonInterpreterCandidates = []string{"python3", "python", "py"}

// claudeCodeHealth mirrors the Python doctor's semantics for claude-code
// (sdk/python/agentfield/harness/_doctor.py::_claude_health): the provider is
// usable when the claude_agent_sdk pip package is importable — it bundles its
// own CLI — so a globally installed `claude` binary is neither necessary nor
// sufficient.
func claudeCodeHealth(spec harnessProviderSpec) HarnessProviderHealth {
	installed, pythonFound := probeClaudeWrapper()
	issues := []string{}
	if !pythonFound {
		issues = append(issues, "python_not_found")
	} else if !installed {
		issues = append(issues, "wrapper_not_installed")
	}
	return HarnessProviderHealth{
		Provider:       spec.Name,
		Installed:      installed,
		Auth:           harnessAuthStatus(spec.AuthEnvVars),
		Usable:         installed,
		InstallCommand: spec.InstallCommand,
		AuthEnvVars:    append([]string{}, spec.AuthEnvVars...),
		Issues:         issues,
	}
}

// probeClaudeWrapper reports whether claude_agent_sdk is importable and
// whether a working Python interpreter was found at all. Candidates are
// run-probed rather than merely resolved on PATH because dead launcher stubs
// (e.g. the Windows Store python3 alias) resolve but fail when executed.
func probeClaudeWrapper() (installed bool, pythonFound bool) {
	for _, interpreter := range pythonInterpreterCandidates {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		out, err := exec.CommandContext(ctx, interpreter, "-c", claudeWrapperProbe).Output()
		cancel()
		if err != nil {
			continue
		}
		switch strings.TrimSpace(string(out)) {
		case "ok":
			return true, true
		case "missing":
			return false, true
		}
	}
	return false, false
}

func harnessAuthStatus(envVars []string) string {
	for _, envVar := range envVars {
		if strings.TrimSpace(os.Getenv(envVar)) != "" {
			return "configured"
		}
	}
	return "unknown"
}

func findHarnessProviderSpec(name string) *harnessProviderSpec {
	for i := range harnessProviderSpecs {
		if harnessProviderSpecs[i].Name == name {
			return &harnessProviderSpecs[i]
		}
	}
	return nil
}

func printHarnessDoctorReports(cmd *cobra.Command, reports []HarnessProviderHealth) {
	for _, report := range reports {
		status := "ready"
		if !report.Usable {
			status = "unavailable"
		}
		fmt.Fprintf(cmd.OutOrStdout(), "%s: %s", report.Provider, status)
		if report.Version != "" {
			fmt.Fprintf(cmd.OutOrStdout(), " (%s)", report.Version)
		}
		fmt.Fprintln(cmd.OutOrStdout())
		if !report.Installed {
			fmt.Fprintf(cmd.OutOrStdout(), "  install: %s\n", report.InstallCommand)
		}
	}
}
