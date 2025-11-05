package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/cli"
	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/server"
	"github.com/Agent-Field/agentfield/control-plane/internal/utils"
	"github.com/Agent-Field/agentfield/control-plane/web/client"

	"github.com/spf13/cobra" // Import cobra
	"github.com/spf13/viper"
)

var (
	loadConfigFunc            = loadConfig
	newAgentFieldServerFunc   = server.NewAgentFieldServer
	buildUIFunc               = buildUI
	openBrowserFunc           = openBrowser
	sleepFunc                 = time.Sleep
	waitForShutdownFunc       = func() { select {} }
	commandRunner             = defaultCommandRunner
	browserLauncher           = defaultBrowserLauncher
	startAgentFieldServerFunc = defaultStartAgentFieldServer
)

// main function now acts as the entry point for the Cobra CLI.
func main() {
	rootCmd := cli.NewRootCommand(runServer) // Initialize RootCmd and add subcommands

	// Execute the root command
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// runServer contains the original server startup logic.
// This function will be called by the Cobra command's Run field.
func runServer(cmd *cobra.Command, args []string) {
	fmt.Println("AgentField server starting...")

	// Load configuration
	cfgFilePath, _ := cmd.Flags().GetString("config")
	cfg, err := loadConfigFunc(cfgFilePath)
	if err != nil {
		log.Fatalf("Failed to load configuration: %v", err)
	}

	// Override port from flag if provided
	if cmd.Flags().Lookup("port").Changed {
		port, _ := cmd.Flags().GetInt("port")
		cfg.AgentField.Port = port
	}

	// Override from environment variables
	if envPort := os.Getenv("AGENTFIELD_PORT"); envPort != "" {
		if port, err := strconv.Atoi(envPort); err == nil {
			cfg.AgentField.Port = port
		}
	}

	storageModeExplicit := false
	if flag := cmd.Flags().Lookup("storage-mode"); flag != nil && flag.Changed {
		if mode, err := cmd.Flags().GetString("storage-mode"); err == nil && mode != "" {
			cfg.Storage.Mode = mode
			storageModeExplicit = true
		}
	}

	if !storageModeExplicit {
		if envMode := os.Getenv("AGENTFIELD_STORAGE_MODE"); envMode != "" {
			cfg.Storage.Mode = envMode
		}
	}

	var postgresURL string
	if flag := cmd.Flags().Lookup("postgres-url"); flag != nil && flag.Changed {
		postgresURL, _ = cmd.Flags().GetString("postgres-url")
	}
	if postgresURL == "" {
		if env := os.Getenv("AGENTFIELD_POSTGRES_URL"); env != "" {
			postgresURL = env
		} else if env := os.Getenv("AGENTFIELD_STORAGE_POSTGRES_URL"); env != "" {
			postgresURL = env
		}
	}

	if postgresURL != "" {
		cfg.Storage.Postgres.DSN = postgresURL
		cfg.Storage.Postgres.URL = postgresURL
		if !storageModeExplicit {
			cfg.Storage.Mode = "postgres"
		}
	}

	if env := os.Getenv("AGENTFIELD_STORAGE_POSTGRES_HOST"); env != "" {
		cfg.Storage.Postgres.Host = env
	}
	if env := os.Getenv("AGENTFIELD_STORAGE_POSTGRES_PORT"); env != "" {
		if port, err := strconv.Atoi(env); err == nil {
			cfg.Storage.Postgres.Port = port
		}
	}
	if env := os.Getenv("AGENTFIELD_STORAGE_POSTGRES_DATABASE"); env != "" {
		cfg.Storage.Postgres.Database = env
	}
	if env := os.Getenv("AGENTFIELD_STORAGE_POSTGRES_USER"); env != "" {
		cfg.Storage.Postgres.User = env
	}
	if env := os.Getenv("AGENTFIELD_STORAGE_POSTGRES_PASSWORD"); env != "" {
		cfg.Storage.Postgres.Password = env
	}
	if env := os.Getenv("AGENTFIELD_STORAGE_POSTGRES_SSLMODE"); env != "" {
		cfg.Storage.Postgres.SSLMode = env
	}

	if cfg.Storage.Mode == "" {
		cfg.Storage.Mode = "local"
	}

	// Adjust config based on flags
	backendOnly, _ := cmd.Flags().GetBool("backend-only")
	if backendOnly {
		cfg.UI.Enabled = false
	}
	uiDev, _ := cmd.Flags().GetBool("ui-dev")
	if uiDev {
		cfg.UI.Mode = "dev" // Assuming "dev" mode implies UI is enabled
		cfg.UI.Enabled = true
	}

	// Disable execution VC generation if flag is set
	if noVC, err := cmd.Flags().GetBool("no-vc-execution"); err == nil && noVC {
		cfg.Features.DID.VCRequirements.RequireVCForExecution = false
		cfg.Features.DID.VCRequirements.PersistExecutionVC = false
		fmt.Println("⚠️  Execution VC generation disabled via --no-vc-execution flag")
	}

	// Build UI if in embedded mode and not in ui-dev mode and UI is not already embedded
	if cfg.UI.Enabled && cfg.UI.Mode == "embedded" && !uiDev && !client.IsUIEmbedded() {
		fmt.Println("Building UI for embedded mode...")
		if err := buildUIFunc(cfg); err != nil {
			log.Printf("Warning: Failed to build UI, UI might not be available: %v", err)
		} else {
			fmt.Println("UI build successful.")
		}
	} else if cfg.UI.Enabled && cfg.UI.Mode == "embedded" && client.IsUIEmbedded() {
		fmt.Println("UI is already embedded in binary, skipping build.")
	}

	// Create AgentField server instance
	agentfieldServer, err := newAgentFieldServerFunc(cfg)
	if err != nil {
		log.Fatalf("Failed to create AgentField server: %v", err)
	}

	// Start the server in a goroutine so we can open the browser
	go func() {
		fmt.Printf("AgentField server attempting to start on port %d...\n", cfg.AgentField.Port)
		if err := startAgentFieldServerFunc(agentfieldServer); err != nil {
			log.Fatalf("Failed to start AgentField server: %v", err)
		}
	}()

	// Wait a moment for the server to start before opening browser
	sleepFunc(1 * time.Second)

	openBrowserFlag, _ := cmd.Flags().GetBool("open")
	if cfg.UI.Enabled && openBrowserFlag && !backendOnly {
		uiTargetURL := fmt.Sprintf("http://localhost:%d", cfg.AgentField.Port)
		if cfg.UI.Mode == "dev" {
			// Use configured dev port or environment variable
			devPort := cfg.UI.DevPort
			if envDevPort := os.Getenv("VITE_DEV_PORT"); envDevPort != "" {
				if port, err := strconv.Atoi(envDevPort); err == nil {
					devPort = port
				}
			}
			if devPort == 0 {
				devPort = 5173 // Default Vite port
			}
			uiTargetURL = fmt.Sprintf("http://localhost:%d", devPort)
		}
		fmt.Printf("Opening browser to UI at %s...\n", uiTargetURL)
		openBrowserFunc(uiTargetURL)
	}

	fmt.Printf("AgentField server running. Press Ctrl+C to exit.\n")
	// Keep main goroutine alive
	waitForShutdownFunc()

	// TODO: Implement graceful shutdown
}

// loadConfig loads configuration from file and environment variables.
func loadConfig(configFile string) (*config.Config, error) {
	// Set environment variable prefixes
	viper.SetEnvPrefix("AGENTFIELD")
	viper.AutomaticEnv() // read in environment variables that match

	// Skip config file reading if explicitly set to /dev/null or empty
	if configFile != "/dev/null" && configFile != "" {
		viper.SetConfigFile(configFile)
		if err := viper.ReadInConfig(); err != nil {
			return nil, fmt.Errorf("failed to read config file: %w", err)
		}
	} else if configFile == "" {
		// Check for config file path from environment
		if envConfigFile := os.Getenv("AGENTFIELD_CONFIG_FILE"); envConfigFile != "" {
			viper.SetConfigFile(envConfigFile)
		} else {
			viper.SetConfigName("agentfield") // name of config file (without extension)
			viper.SetConfigType("yaml")       // type of the config file
			viper.AddConfigPath("./config")   // path to look for the config file in
			viper.AddConfigPath(".")          // optionally look for config in the working directory
		}

		if err := viper.ReadInConfig(); err != nil {
			if _, ok := err.(viper.ConfigFileNotFoundError); ok {
				fmt.Println("No config file found, using environment variables and defaults.")
			} else {
				return nil, fmt.Errorf("failed to read config file: %w", err)
			}
		}
	} else {
		fmt.Println("Skipping config file, using environment variables and defaults.")
	}

	var cfg config.Config
	if err := viper.Unmarshal(&cfg); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}

	// Apply defaults if not set
	if cfg.AgentField.Port == 0 {
		cfg.AgentField.Port = 8080 // Default port
	}
	if cfg.Storage.Mode == "" {
		cfg.Storage.Mode = "local" // Default storage mode
	}
	// Enable UI by default
	if !viper.IsSet("ui.enabled") {
		cfg.UI.Enabled = true // Default UI enabled
	}
	if cfg.UI.Mode == "" {
		cfg.UI.Mode = "embedded" // Default UI mode
	}
	// Enable DID by default - only disable if explicitly configured
	if !viper.IsSet("features.did.enabled") {
		cfg.Features.DID.Enabled = true // DEFAULT TO TRUE
	}
	// Apply DID sub-config defaults if DID is enabled
	if cfg.Features.DID.Enabled {
		if cfg.Features.DID.Method == "" {
			cfg.Features.DID.Method = "did:key"
		}
		if cfg.Features.DID.KeyAlgorithm == "" {
			cfg.Features.DID.KeyAlgorithm = "Ed25519"
		}
		if cfg.Features.DID.DerivationMethod == "" {
			cfg.Features.DID.DerivationMethod = "BIP32"
		}
		if cfg.Features.DID.KeyRotationDays == 0 {
			cfg.Features.DID.KeyRotationDays = 90
		}
		if cfg.Features.DID.Keystore.Type == "" {
			cfg.Features.DID.Keystore.Type = "local"
		}
		if cfg.Features.DID.Keystore.Path == "" {
			cfg.Features.DID.Keystore.Path = "./data/keys"
		}
		if cfg.Features.DID.Keystore.Encryption == "" {
			cfg.Features.DID.Keystore.Encryption = "AES-256-GCM"
		}
		if !viper.IsSet("features.did.keystore.backup_enabled") {
			cfg.Features.DID.Keystore.BackupEnabled = true
		}
		if cfg.Features.DID.Keystore.BackupInterval == "" {
			cfg.Features.DID.Keystore.BackupInterval = "24h"
		}
		// Apply VC requirements defaults
		if !viper.IsSet("features.did.vc_requirements.require_vc_registration") {
			cfg.Features.DID.VCRequirements.RequireVCForRegistration = true
		}
		if !viper.IsSet("features.did.vc_requirements.require_vc_execution") {
			cfg.Features.DID.VCRequirements.RequireVCForExecution = true
		}
		if !viper.IsSet("features.did.vc_requirements.require_vc_cross_agent") {
			cfg.Features.DID.VCRequirements.RequireVCForCrossAgent = true
		}
		if !viper.IsSet("features.did.vc_requirements.store_input_output") {
			cfg.Features.DID.VCRequirements.StoreInputOutput = false
		}
		if !viper.IsSet("features.did.vc_requirements.hash_sensitive_data") {
			cfg.Features.DID.VCRequirements.HashSensitiveData = true
		}
		if !viper.IsSet("features.did.vc_requirements.persist_execution_vc") {
			cfg.Features.DID.VCRequirements.PersistExecutionVC = true
		}
		if cfg.Features.DID.VCRequirements.StorageMode == "" {
			cfg.Features.DID.VCRequirements.StorageMode = "inline"
		}
	}
	if cfg.UI.Enabled && cfg.UI.DevPort == 0 {
		cfg.UI.DevPort = 5173 // Default Vite dev port
	}
	if cfg.UI.SourcePath == "" {
		// Get the executable path and find UI relative to it
		execPath, err := os.Executable()
		if err != nil {
			cfg.UI.SourcePath = filepath.Join("apps", "platform", "agentfield", "web", "client")
			if _, statErr := os.Stat(cfg.UI.SourcePath); os.IsNotExist(statErr) {
				cfg.UI.SourcePath = filepath.Join("web", "client")
			}
		} else {
			execDir := filepath.Dir(execPath)
			// Look for web/client relative to the executable directory
			// This assumes the binary is built in the agentfield/ directory
			cfg.UI.SourcePath = filepath.Join(execDir, "web", "client")

			// If that doesn't exist, try going up one level (if binary is in agentfield/)
			if _, err := os.Stat(cfg.UI.SourcePath); os.IsNotExist(err) {
				cfg.UI.SourcePath = filepath.Join(filepath.Dir(execDir), "apps", "platform", "agentfield", "web", "client")
			}

			// Final fallback to current working directory
			if _, err := os.Stat(cfg.UI.SourcePath); os.IsNotExist(err) {
				altPath := filepath.Join("apps", "platform", "agentfield", "web", "client")
				if _, altErr := os.Stat(altPath); altErr == nil {
					cfg.UI.SourcePath = altPath
				} else {
					cfg.UI.SourcePath = filepath.Join("web", "client")
				}
			}
		}
	}
	if cfg.UI.DistPath == "" {
		cfg.UI.DistPath = filepath.Join(cfg.UI.SourcePath, "dist") // Default UI dist path relative to source
	}
	// Set default storage paths for local mode using universal path management
	if cfg.Storage.Mode == "local" {
		// Use the universal path management system
		if cfg.Storage.Local.DatabasePath == "" {
			dbPath, err := utils.GetDatabasePath()
			if err != nil {
				return nil, fmt.Errorf("failed to get database path: %w", err)
			}
			cfg.Storage.Local.DatabasePath = dbPath
		}
		if cfg.Storage.Local.KVStorePath == "" {
			kvPath, err := utils.GetKVStorePath()
			if err != nil {
				return nil, fmt.Errorf("failed to get KV store path: %w", err)
			}
			cfg.Storage.Local.KVStorePath = kvPath
		}
		// Ensure all AgentField data directories exist
		if _, err := utils.EnsureDataDirectories(); err != nil {
			return nil, fmt.Errorf("failed to create AgentField data directories: %w", err)
		}
	}

	fmt.Printf("Loaded config - Storage mode: %s, AgentField Port: %d, UI Mode: %s, UI Enabled: %t, DID Enabled: %t\n",
		cfg.Storage.Mode, cfg.AgentField.Port, cfg.UI.Mode, cfg.UI.Enabled, cfg.Features.DID.Enabled)

	return &cfg, nil
}

func buildUI(cfg *config.Config) error {
	uiDir := cfg.UI.SourcePath
	if uiDir == "" {
		uiDir = "./web/client" // Default path
	}

	// Check if package.json exists
	if _, err := os.Stat(filepath.Join(uiDir, "package.json")); os.IsNotExist(err) {
		log.Printf("UI source path (%s) or package.json not found. Skipping UI build.", uiDir)
		return nil // Not an error if UI is optional or pre-built
	}

	fmt.Printf("Building UI in %s...\n", uiDir)

	// Set environment variables for the build process
	buildEnv := os.Environ()

	// Add Vite environment variables based on config
	if cfg.UI.DistPath != "" {
		buildEnv = append(buildEnv, fmt.Sprintf("VITE_BUILD_OUT_DIR=%s", filepath.Base(cfg.UI.DistPath)))
	}

	// Set API proxy target for development builds
	buildEnv = append(buildEnv, fmt.Sprintf("VITE_API_PROXY_TARGET=http://localhost:%d", cfg.AgentField.Port))

	// Install dependencies
	if err := commandRunner(uiDir, buildEnv, "npm", "install", "--force"); err != nil {
		return fmt.Errorf("failed to install UI dependencies: %w", err)
	}

	// Build for production
	if err := commandRunner(uiDir, buildEnv, "npm", "run", "build"); err != nil {
		return fmt.Errorf("failed to build UI: %w", err)
	}
	return nil
}

func defaultCommandRunner(dir string, env []string, name string, args ...string) error {
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	if len(env) > 0 {
		cmd.Env = env
	}
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func defaultBrowserLauncher(name string, args ...string) error {
	return exec.Command(name, args...).Start()
}

func defaultStartAgentFieldServer(s *server.AgentFieldServer) error {
	return s.Start()
}

func openBrowser(url string) {
	var err error
	switch runtime.GOOS {
	case "linux":
		err = browserLauncher("xdg-open", url)
	case "windows":
		err = browserLauncher("rundll32", "url.dll,FileProtocolHandler", url)
	case "darwin":
		err = browserLauncher("open", url)
	default:
		err = fmt.Errorf("unsupported platform")
	}
	if err != nil {
		log.Printf("Failed to open browser: %v. Please open manually: %s", err, url)
	}
}
