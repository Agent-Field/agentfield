package cli

import (
	"encoding/json"
	"fmt"
	"os"
	"runtime"

	"github.com/Agent-Field/agentfield/control-plane/internal/application"
	"github.com/Agent-Field/agentfield/control-plane/internal/cli/commands"
	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var (
	cfgFile          string
	verbose          bool
	openBrowserFlag  bool
	uiDevFlag        bool
	backendOnlyFlag  bool
	portFlag         int
	noVCExecution    bool
	forceVCExecution bool
	storageModeFlag  string
	postgresURLFlag  string
	serverURL        string
	apiKey           string
	outputFormat     string
	requestTimeout   int
)

// NewRootCommand creates and returns the root Cobra command for the AgentField CLI.
func NewRootCommand(runServerFunc func(cmd *cobra.Command, args []string), versionInfo VersionInfo) *cobra.Command {
	RootCmd := &cobra.Command{
		Use:     "af",
		Aliases: []string{"agentfield"},
		Short:   "AgentField AI Agent Platform",
		Long: `AgentField is a comprehensive AI agent platform for building, managing, and deploying AI agent capabilities.

AI Agent? Run "af agent help" for structured JSON output optimized for programmatic use.`,
		PersistentPreRunE: func(cmd *cobra.Command, args []string) error {
			// Initialize logging based on verbose flag
			logger.InitLogger(verbose)
			if verbose {
				logger.Logger.Debug().Msg("Verbose logging enabled.")
			}
			return nil
		},
		// Default to server mode when no subcommand is provided (backward compatibility)
		Run: runServerFunc,
	}

	// Add --version flag
	var showVersion bool
	RootCmd.Flags().BoolVar(&showVersion, "version", false, "Print version information")

	// Override Run to check for version flag
	originalRun := RootCmd.Run
	RootCmd.Run = func(cmd *cobra.Command, args []string) {
		if showVersion {
			fmt.Printf("AgentField Control Plane\n")
			fmt.Printf("  Version:    %s\n", versionInfo.Version)
			fmt.Printf("  Commit:     %s\n", versionInfo.Commit)
			fmt.Printf("  Built:      %s\n", versionInfo.Date)
			fmt.Printf("  Go version: %s\n", runtime.Version())
			fmt.Printf("  OS/Arch:    %s/%s\n", runtime.GOOS, runtime.GOARCH)
			return
		}
		originalRun(cmd, args)
	}
	RootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "Path to configuration file (e.g., config/agentfield.yaml)")
	RootCmd.PersistentFlags().BoolVarP(&verbose, "verbose", "v", false, "Enable verbose logging")

	// Flags for the server command (moved from main.go)
	RootCmd.PersistentFlags().BoolVar(&openBrowserFlag, "open", true, "Open browser to UI (if UI is enabled)")
	RootCmd.PersistentFlags().BoolVar(&uiDevFlag, "ui-dev", false, "Run with UI development server (proxies to backend)")
	RootCmd.PersistentFlags().BoolVar(&backendOnlyFlag, "backend-only", false, "Run only backend APIs, UI served separately")
	RootCmd.PersistentFlags().IntVar(&portFlag, "port", 0, "Port for the af server (overrides config if set)")
	RootCmd.PersistentFlags().BoolVar(&noVCExecution, "no-vc-execution", false, "Disable generating verifiable credentials for executions")
	RootCmd.PersistentFlags().BoolVar(&forceVCExecution, "vc-execution", false, "Force-enable generating verifiable credentials for executions")
	RootCmd.PersistentFlags().StringVar(&storageModeFlag, "storage-mode", "", "Override the storage backend (local or postgres)")
	RootCmd.PersistentFlags().StringVar(&postgresURLFlag, "postgres-url", "", "PostgreSQL connection URL or DSN (implies --storage-mode=postgres)")
	RootCmd.PersistentFlags().StringVarP(&serverURL, "server", "s", "", "Control plane URL (env: AGENTFIELD_SERVER, default: http://localhost:8080)")
	RootCmd.PersistentFlags().StringVarP(&apiKey, "api-key", "k", "", "API key for authenticated endpoints (env: AGENTFIELD_API_KEY)")

	cobra.OnInitialize(initConfig)

	// Add init command
	RootCmd.AddCommand(NewInitCommand())

	// Add doctor command — environment introspection for skills/coding agents
	RootCmd.AddCommand(NewDoctorCommand())

	// Add skill command — install/manage AgentField skills across coding agents
	RootCmd.AddCommand(NewSkillCommand())

	// Create service container for framework commands
	cfg := &config.Config{} // Use default config for now
	services := application.CreateServiceContainer(cfg, getAgentFieldHomeDir())

	// Add framework-based commands (migrated commands)
	installCommand := commands.NewInstallCommand(services)
	RootCmd.AddCommand(installCommand.BuildCobraCommand())

	runCommand := commands.NewRunCommand(services)
	RootCmd.AddCommand(runCommand.BuildCobraCommand())

	devCommand := commands.NewDevCommand(services)
	RootCmd.AddCommand(devCommand.BuildCobraCommand())

	// Add remaining old commands (not yet migrated)
	RootCmd.AddCommand(NewUninstallCommand())
	RootCmd.AddCommand(NewListCommand())
	RootCmd.AddCommand(NewStopCommand())
	RootCmd.AddCommand(NewLogsCommand())
	RootCmd.AddCommand(NewConfigCommand())
RootCmd.AddCommand(NewVCCommand())
	RootCmd.AddCommand(NewNodesCommand())
	RootCmd.AddCommand(NewExecutionCommand())

	// Add version command
	RootCmd.AddCommand(NewVersionCommand(versionInfo))
	RootCmd.AddCommand(NewAgentCommand())

	// Add the server command
	serverCmd := &cobra.Command{
		Use:   "server",
		Short: "Run the AgentField AI Agent Platform server",
		Long:  `Starts the AgentField AI Agent Platform server, providing API endpoints and UI.`,
		Run:   runServerFunc,
	}
	RootCmd.AddCommand(serverCmd)

	return RootCmd
}

const AgentHint = `AI Agent? Run "af agent help" for structured JSON output.`

// AgentHintJSON returns a structured JSON hint on stderr for agents that ran a wrong root command.
func AgentHintJSON(errMsg string) string {
	hint := map[string]interface{}{
		"ok": false,
		"error": map[string]string{
			"code":    "invalid_command",
			"message": errMsg,
			"hint":    `Use "af agent <subcommand>" for machine-friendly JSON output. Run "af agent help" for the full command reference.`,
		},
	}
	b, _ := json.Marshal(hint)
	return string(b)
}

// initConfig reads in config file and ENV variables if set.
func initConfig() {
	if cfgFile != "" {
		// Use config file from the flag.
		viper.SetConfigFile(cfgFile)
	} else {
		// Search config in current directory and "config" directory
		viper.AddConfigPath(".")
		viper.AddConfigPath("./config")
		viper.SetConfigName("agentfield")
		viper.SetConfigType("yaml")
	}

	viper.AutomaticEnv() // read in environment variables that match

	if err := viper.ReadInConfig(); err == nil {
		fmt.Fprintln(os.Stderr, "Using config file:", viper.ConfigFileUsed())
	}
}

// Getters for flags
func GetConfigFilePath() string {
	return cfgFile
}

func GetOpenBrowserFlag() bool {
	return openBrowserFlag
}

func GetUIDevFlag() bool {
	return uiDevFlag
}

func GetBackendOnlyFlag() bool {
	return backendOnlyFlag
}

func GetPortFlag() int {
	return portFlag
}

func GetServerURL() string {
	if serverURL != "" {
		return serverURL
	}
	if env := os.Getenv("AGENTFIELD_SERVER"); env != "" {
		return env
	}
	if env := os.Getenv("AGENTFIELD_SERVER_URL"); env != "" {
		return env
	}
	return "http://localhost:8080"
}

func GetAPIKey() string {
	if apiKey != "" {
		return apiKey
	}
	return os.Getenv("AGENTFIELD_API_KEY")
}

func GetOutputFormat() string {
	return outputFormat
}

func GetRequestTimeout() int {
	return requestTimeout
}
