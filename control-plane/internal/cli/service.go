package cli

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"runtime"
	"strconv"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/launchdsvc"
	"github.com/spf13/cobra"
)

// NewServiceCommand creates the `af service` command group, which manages the
// control plane running under launchd (installed by install.sh / the menu-bar
// app). It is registered on every platform so `af service` self-documents;
// off macOS each subcommand reports that launchd is macOS-only.
func NewServiceCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "service",
		Short: "Manage the background control plane (macOS launchd)",
		Long: `Manage the AgentField control plane that runs in the background.

install.sh registers the control plane as a launchd agent so it starts at login
and restarts if it crashes. Because it is supervised, killing the process does
not stop it — use these commands (or the menu-bar icon).

Examples:
  af service status     Show whether it is registered, healthy, and busy
  af service stop       Graceful shutdown (stays stopped)
  af service restart    Restart onto the currently installed binary
  af service uninstall  Deregister it and remove the menu-bar app`,
	}
	cmd.AddCommand(newServiceStatusCmd())
	cmd.AddCommand(newServiceStopCmd())
	cmd.AddCommand(newServiceRestartCmd())
	cmd.AddCommand(newServiceUninstallCmd())
	return cmd
}

// agentLoadedFn is the launchd registration probe, indirected so tests can
// assemble a status without shelling out to launchctl at all.
var agentLoadedFn = launchdsvc.AgentLoaded

func servicePort() int {
	if v := os.Getenv("AGENTFIELD_PORT"); v != "" {
		if p, err := strconv.Atoi(v); err == nil {
			return p
		}
	}
	return 8080
}

func serviceHome() string {
	h, _ := os.UserHomeDir()
	return h
}

func newServiceStatusCmd() *cobra.Command {
	var asJSON bool
	cmd := &cobra.Command{
		Use:   "status",
		Short: "Show the background control plane's registration, health, and load",
		RunE: func(cmd *cobra.Command, args []string) error {
			st := collectServiceStatus()
			if asJSON {
				out, err := json.MarshalIndent(st, "", "  ")
				if err != nil {
					return err
				}
				fmt.Println(string(out))
				return nil
			}
			printServiceStatus(st)
			return nil
		},
	}
	cmd.Flags().BoolVar(&asJSON, "json", false, "Emit machine-readable JSON")
	return cmd
}

// serviceStatus is the read-only view of the background control plane.
type serviceStatus struct {
	Supported        bool   `json:"supported"`
	Loaded           bool   `json:"loaded"`
	PlistPath        string `json:"plist_path"`
	Program          string `json:"program,omitempty"`
	Healthy          bool   `json:"healthy"`
	Port             int    `json:"port"`
	Version          string `json:"version,omitempty"`
	ActiveExecutions int    `json:"active_executions"`
	StaleExecutions  int    `json:"stale_executions"`
	ActiveKnown      bool   `json:"active_known"`
}

func collectServiceStatus() serviceStatus {
	port := servicePort()
	st := serviceStatus{
		Supported: launchdsvc.Supported(),
		Port:      port,
		PlistPath: launchdsvc.ServerPlistPath(serviceHome()),
	}
	if owner, ok := launchdsvc.ReadPlistOwner(st.PlistPath); ok {
		st.Program = owner.Program
	}
	if st.Supported {
		st.Loaded = agentLoadedFn(launchdsvc.ServerLabel)
	}
	st.Healthy = launchdsvc.ServerHealthy(port)
	if st.Healthy {
		st.Version = serviceServerVersion(port)
		st.ActiveExecutions, st.StaleExecutions, st.ActiveKnown =
			launchdsvc.ActiveExecutions(port, os.Getenv("AGENTFIELD_API_KEY"))
	}
	return st
}

// serviceServerVersion reads the version the running server reports. Best
// effort: an older server without the field simply reports nothing.
func serviceServerVersion(port int) string {
	client := &http.Client{Timeout: 1500 * time.Millisecond}
	resp, err := client.Get(fmt.Sprintf("http://localhost:%d/health", port))
	if err != nil {
		return ""
	}
	defer func() { _ = resp.Body.Close() }()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return ""
	}
	var parsed struct {
		Version string `json:"version"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return ""
	}
	return parsed.Version
}

func printServiceStatus(st serviceStatus) {
	if !st.Supported {
		fmt.Printf("Registration:  not applicable (launchd is macOS-only; this is %s)\n", runtime.GOOS)
	} else if st.Loaded {
		fmt.Println("Registration:  loaded (starts at login)")
	} else {
		fmt.Println("Registration:  not loaded")
	}
	if st.Program != "" {
		fmt.Printf("Binary:        %s\n", st.Program)
	}
	if st.Healthy {
		if st.Version != "" {
			fmt.Printf("Health:        responding on :%d (version %s)\n", st.Port, st.Version)
		} else {
			fmt.Printf("Health:        responding on :%d\n", st.Port)
		}
	} else {
		fmt.Printf("Health:        not responding on :%d\n", st.Port)
	}
	switch {
	case !st.Healthy:
	case st.ActiveKnown:
		if st.StaleExecutions > 0 {
			// Name the wedged runs explicitly: they are why the number here can
			// look lower than the dashboard's.
			fmt.Printf("In flight:     %d workflow(s) (plus %d stale, idle >%s)\n",
				st.ActiveExecutions, st.StaleExecutions, launchdsvc.ActiveWindow())
		} else {
			fmt.Printf("In flight:     %d workflow(s)\n", st.ActiveExecutions)
		}
	default:
		fmt.Println("In flight:     unknown (endpoint unreadable — set AGENTFIELD_API_KEY?)")
	}
}

func newServiceStopCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "stop",
		Short: "Stop the background control plane (graceful, stays stopped)",
		Long: `Send SIGTERM to the launchd-supervised control plane.

The agent is registered with KeepAlive={SuccessfulExit: false}, so a clean
shutdown is NOT relaunched — but a plain ` + "`kill`" + ` of the process looks like a
crash and launchd restarts it. That is why this command exists.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			return serviceStop()
		},
	}
}

func newServiceRestartCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "restart",
		Short: "Restart the background control plane onto the installed binary",
		RunE: func(cmd *cobra.Command, args []string) error {
			return serviceRestart()
		},
	}
}

func newServiceUninstallCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "uninstall",
		Short: "Deregister the control plane and remove the menu-bar app",
		RunE: func(cmd *cobra.Command, args []string) error {
			return serviceUninstall()
		},
	}
}
