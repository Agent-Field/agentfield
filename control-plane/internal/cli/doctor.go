package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"time"

	"github.com/fatih/color"
	"github.com/spf13/cobra"
)

// DoctorReport is the JSON structure returned by `af doctor --json`.
// Coding agents (and skills like agentfield-multi-reasoner-builder) call this once
// to learn what's actually available in the environment instead of probing manually.
type DoctorReport struct {
	OS              string                  `json:"os"`
	Arch            string                  `json:"arch"`
	Python          ToolStatus              `json:"python"`
	Node            ToolStatus              `json:"node"`
	Docker          ToolStatus              `json:"docker"`
	HarnessProviders map[string]ToolStatus  `json:"harness_providers"`
	ProviderKeys    map[string]ProviderKey  `json:"provider_keys"`
	ControlPlane    ControlPlaneStatus      `json:"control_plane"`
	Recommendation  Recommendation          `json:"recommendation"`
}

// ToolStatus describes whether a CLI is available and, if so, where.
type ToolStatus struct {
	Available bool   `json:"available"`
	Path      string `json:"path,omitempty"`
	Version   string `json:"version,omitempty"`
}

// ProviderKey reports whether a provider's API key env var is set
// (without ever leaking the value).
type ProviderKey struct {
	EnvVar string `json:"env_var"`
	Set    bool   `json:"set"`
}

// ControlPlaneStatus reports whether a local control plane is reachable
// and whether the Docker image is locally available.
type ControlPlaneStatus struct {
	URL                string `json:"url"`
	Reachable          bool   `json:"reachable"`
	HealthStatus       string `json:"health_status,omitempty"`
	DockerImageName    string `json:"docker_image_name"`
	DockerImageLocal   bool   `json:"docker_image_local"`
}

// Recommendation tells the caller (a skill or a coding agent) what to default to,
// based on what's actually present in the environment.
type Recommendation struct {
	Provider           string   `json:"provider"`             // "openrouter" / "openai" / "anthropic" / "google" / "none"
	AIModel            string   `json:"ai_model"`             // suggested LiteLLM-style model string
	HarnessUsable      bool     `json:"harness_usable"`       // true only if at least one provider CLI is on PATH
	HarnessProviders   []string `json:"harness_providers"`    // available provider CLI names
	Notes              []string `json:"notes"`                // human-readable suggestions
}

// providerEnvVars maps provider name -> env var. Order matters for the recommendation.
var providerEnvVars = []struct {
	Name   string
	EnvVar string
	Model  string // suggested default model when this provider is the chosen one
}{
	{Name: "openrouter", EnvVar: "OPENROUTER_API_KEY", Model: "openrouter/google/gemini-2.5-flash"},
	{Name: "anthropic", EnvVar: "ANTHROPIC_API_KEY", Model: "claude-3-5-sonnet-20241022"},
	{Name: "openai", EnvVar: "OPENAI_API_KEY", Model: "gpt-4o"},
	{Name: "google", EnvVar: "GOOGLE_API_KEY", Model: "gemini-1.5-pro"},
}

// harnessProviders is the canonical list of CLIs `app.harness()` knows how to drive.
var harnessProviders = []struct {
	Name   string // value passed to provider= in app.harness()
	Binary string // executable name to look up on PATH
}{
	{Name: "claude-code", Binary: "claude"},
	{Name: "codex", Binary: "codex"},
	{Name: "gemini", Binary: "gemini"},
	{Name: "opencode", Binary: "opencode"},
}

// NewDoctorCommand builds the `af doctor` command.
func NewDoctorCommand() *cobra.Command {
	var jsonOut bool
	var controlPlaneURL string

	cmd := &cobra.Command{
		Use:   "doctor",
		Short: "Inspect the local environment for AgentField development capabilities",
		Long: `Doctor inspects the local environment and reports what's available for
building AgentField multi-reasoner systems:

  • Available harness provider CLIs (claude-code, codex, gemini, opencode)
  • Provider API keys set in the environment (without leaking values)
  • Docker availability and whether the control-plane image is locally cached
  • Whether a local control plane is reachable
  • A recommended default provider, model, and whether app.harness() is usable

Coding agents and skills (e.g. agentfield-multi-reasoner-builder) should call
this once at the start of a build to learn ground truth instead of probing
each tool by hand.

Examples:
  af doctor                  # Pretty human-readable output
  af doctor --json           # Machine-readable JSON for tooling/skills
  af doctor --json | jq      # Pipe to jq for filtering`,
		RunE: func(cmd *cobra.Command, args []string) error {
			report := buildDoctorReport(controlPlaneURL)

			if jsonOut {
				enc := json.NewEncoder(os.Stdout)
				enc.SetIndent("", "  ")
				return enc.Encode(report)
			}

			printDoctorReport(report)
			return nil
		},
	}

	cmd.Flags().BoolVar(&jsonOut, "json", false, "Output the report as JSON (recommended for tools and skills)")
	cmd.Flags().StringVar(&controlPlaneURL, "server", "http://localhost:8080", "Control plane URL to probe for /api/v1/health")

	return cmd
}

// buildDoctorReport collects the full environment snapshot.
func buildDoctorReport(controlPlaneURL string) DoctorReport {
	report := DoctorReport{
		OS:               runtime.GOOS,
		Arch:             runtime.GOARCH,
		Python:           checkTool("python3", "--version"),
		Node:             checkTool("node", "--version"),
		Docker:           checkTool("docker", "--version"),
		HarnessProviders: map[string]ToolStatus{},
		ProviderKeys:     map[string]ProviderKey{},
	}

	// Some systems use "python" instead of "python3"
	if !report.Python.Available {
		report.Python = checkTool("python", "--version")
	}

	// Harness CLIs
	availableHarness := []string{}
	for _, h := range harnessProviders {
		status := checkTool(h.Binary, "--version")
		report.HarnessProviders[h.Name] = status
		if status.Available {
			availableHarness = append(availableHarness, h.Name)
		}
	}

	// Provider keys
	chosenProvider := ""
	chosenModel := ""
	for _, p := range providerEnvVars {
		set := strings.TrimSpace(os.Getenv(p.EnvVar)) != ""
		report.ProviderKeys[p.Name] = ProviderKey{EnvVar: p.EnvVar, Set: set}
		if set && chosenProvider == "" {
			chosenProvider = p.Name
			chosenModel = p.Model
		}
	}

	// Control plane
	report.ControlPlane = checkControlPlane(controlPlaneURL)

	// Recommendation
	notes := []string{}
	if chosenProvider == "" {
		chosenProvider = "none"
		chosenModel = "openrouter/google/gemini-2.5-flash"
		notes = append(notes, "No provider API key detected. Set OPENROUTER_API_KEY (recommended) or OPENAI_API_KEY / ANTHROPIC_API_KEY before building.")
	} else {
		notes = append(notes, fmt.Sprintf("Provider key detected: %s. Default model: %s", chosenProvider, chosenModel))
	}

	if len(availableHarness) == 0 {
		notes = append(notes, "No harness provider CLIs available. Do NOT use app.harness() in scaffolds — use app.ai(tools=[...]) or chunked-loop reasoners instead.")
	} else {
		notes = append(notes, fmt.Sprintf("Harness providers available: %s. app.harness(provider=...) is usable.", strings.Join(availableHarness, ", ")))
	}

	if !report.Docker.Available {
		notes = append(notes, "Docker not available. Generated docker-compose.yml will validate with `docker compose config` but cannot be run locally.")
	}

	if !report.ControlPlane.DockerImageLocal && report.Docker.Available {
		notes = append(notes, fmt.Sprintf("Control plane image %s not present locally. First `docker compose up` will pull it.", report.ControlPlane.DockerImageName))
	}

	report.Recommendation = Recommendation{
		Provider:         chosenProvider,
		AIModel:          chosenModel,
		HarnessUsable:    len(availableHarness) > 0,
		HarnessProviders: availableHarness,
		Notes:            notes,
	}

	return report
}

// checkTool runs `<bin> <versionFlag>` and reports whether the binary is on PATH.
func checkTool(bin, versionFlag string) ToolStatus {
	path, err := exec.LookPath(bin)
	if err != nil {
		return ToolStatus{Available: false}
	}
	status := ToolStatus{Available: true, Path: path}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, bin, versionFlag)
	out, err := cmd.CombinedOutput()
	if err == nil {
		// First line of version output is usually enough; trim noise.
		first := strings.SplitN(strings.TrimSpace(string(out)), "\n", 2)[0]
		status.Version = first
	}
	return status
}

// checkControlPlane probes the control plane and checks for the docker image.
func checkControlPlane(url string) ControlPlaneStatus {
	status := ControlPlaneStatus{
		URL:             url,
		DockerImageName: "agentfield/control-plane:latest",
	}

	// Health check
	ctx, cancel := context.WithTimeout(context.Background(), 1500*time.Millisecond)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(url, "/")+"/api/v1/health", nil)
	resp, err := http.DefaultClient.Do(req)
	if err == nil {
		defer resp.Body.Close()
		status.Reachable = resp.StatusCode == 200
		status.HealthStatus = resp.Status
	}

	// Local docker image
	if _, err := exec.LookPath("docker"); err == nil {
		ctx2, cancel2 := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel2()
		out, err := exec.CommandContext(ctx2, "docker", "image", "inspect", status.DockerImageName).CombinedOutput()
		if err == nil && len(out) > 0 && !strings.Contains(string(out), "No such image") {
			status.DockerImageLocal = true
		}
	}

	return status
}

// printDoctorReport renders the report in human-readable form.
func printDoctorReport(r DoctorReport) {
	bold := color.New(color.Bold)
	green := color.New(color.FgGreen)
	red := color.New(color.FgRed)
	yellow := color.New(color.FgYellow)
	cyan := color.New(color.FgCyan)

	bold.Println("AgentField environment doctor")
	fmt.Printf("  os/arch: %s/%s\n\n", r.OS, r.Arch)

	bold.Println("Runtimes")
	printToolLine("python", r.Python, green, red)
	printToolLine("node", r.Node, green, red)
	printToolLine("docker", r.Docker, green, red)
	fmt.Println()

	bold.Println("Harness provider CLIs (for app.harness)")
	for _, h := range harnessProviders {
		printToolLine(h.Name, r.HarnessProviders[h.Name], green, red)
	}
	fmt.Println()

	bold.Println("Provider API keys")
	for _, p := range providerEnvVars {
		key := r.ProviderKeys[p.Name]
		mark := "✗"
		c := red
		if key.Set {
			mark = "✓"
			c = green
		}
		c.Printf("  %s %-12s  %s (%s)\n", mark, p.Name, key.EnvVar, ifThen(key.Set, "set", "unset"))
	}
	fmt.Println()

	bold.Println("Control plane")
	cyan.Printf("  url: %s\n", r.ControlPlane.URL)
	mark := "✗ unreachable"
	c := red
	if r.ControlPlane.Reachable {
		mark = "✓ reachable (" + r.ControlPlane.HealthStatus + ")"
		c = green
	}
	c.Printf("  %s\n", mark)
	mark = "✗ image not cached"
	c = yellow
	if r.ControlPlane.DockerImageLocal {
		mark = "✓ image cached locally"
		c = green
	}
	c.Printf("  %s (%s)\n", mark, r.ControlPlane.DockerImageName)
	fmt.Println()

	bold.Println("Recommendation")
	cyan.Printf("  provider:        %s\n", r.Recommendation.Provider)
	cyan.Printf("  AI_MODEL:        %s\n", r.Recommendation.AIModel)
	cyan.Printf("  harness usable:  %v\n", r.Recommendation.HarnessUsable)
	if len(r.Recommendation.HarnessProviders) > 0 {
		cyan.Printf("  harness providers: %s\n", strings.Join(r.Recommendation.HarnessProviders, ", "))
	}
	for _, note := range r.Recommendation.Notes {
		fmt.Printf("  • %s\n", note)
	}
	fmt.Println()
	fmt.Println("Tip: pipe to jq for tooling — `af doctor --json | jq`")
}

func printToolLine(name string, status ToolStatus, ok, fail *color.Color) {
	if status.Available {
		ok.Printf("  ✓ %-12s  %s", name, status.Path)
		if status.Version != "" {
			fmt.Printf("  (%s)", status.Version)
		}
		fmt.Println()
	} else {
		fail.Printf("  ✗ %-12s  not found\n", name)
	}
}

func ifThen(b bool, t, f string) string {
	if b {
		return t
	}
	return f
}
