package main

import (
	"github.com/your-org/haxen/control-plane/internal/cli"
	"github.com/your-org/haxen/control-plane/internal/config"
	"github.com/your-org/haxen/control-plane/internal/logger"
	"github.com/your-org/haxen/control-plane/internal/server"
	"github.com/your-org/haxen/control-plane/internal/utils"
	"github.com/your-org/haxen/control-plane/web/client"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"time"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

func main() {
	// Create and execute the root command with actual server functionality
	rootCmd := cli.NewRootCommand(runServer)
	if err := rootCmd.Execute(); err != nil {
		logger.Logger.Error().Err(err).Msg("Error executing root command")
		os.Exit(1)
	}
}

// runServer contains the server startup logic for unified CLI
func runServer(cmd *cobra.Command, args []string) {
	logger.Logger.Debug().Msg("Haxen server starting...")

	// Load configuration with better defaults
	cfgFilePath, _ := cmd.Flags().GetString("config")
	cfg, err := loadConfig(cfgFilePath)
	if err != nil {
		logger.Logger.Fatal().Err(err).Msg("Failed to load configuration")
	}

	// Override port from flag if provided
	if cmd.Flags().Lookup("port").Changed {
		port, _ := cmd.Flags().GetInt("port")
		cfg.Haxen.Port = port
	}

	// Override from environment variables
	if envPort := os.Getenv("HAXEN_PORT"); envPort != "" {
		if port, err := strconv.Atoi(envPort); err == nil {
			cfg.Haxen.Port = port
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
		if envMode := os.Getenv("HAXEN_STORAGE_MODE"); envMode != "" {
			cfg.Storage.Mode = envMode
		}
	}

	var postgresURL string
	if flag := cmd.Flags().Lookup("postgres-url"); flag != nil && flag.Changed {
		postgresURL, _ = cmd.Flags().GetString("postgres-url")
	}
	if postgresURL == "" {
		if env := os.Getenv("HAXEN_POSTGRES_URL"); env != "" {
			postgresURL = env
		} else if env := os.Getenv("HAXEN_STORAGE_POSTGRES_URL"); env != "" {
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

	if env := os.Getenv("HAXEN_STORAGE_POSTGRES_HOST"); env != "" {
		cfg.Storage.Postgres.Host = env
	}
	if env := os.Getenv("HAXEN_STORAGE_POSTGRES_PORT"); env != "" {
		if port, err := strconv.Atoi(env); err == nil {
			cfg.Storage.Postgres.Port = port
		}
	}
	if env := os.Getenv("HAXEN_STORAGE_POSTGRES_DATABASE"); env != "" {
		cfg.Storage.Postgres.Database = env
	}
	if env := os.Getenv("HAXEN_STORAGE_POSTGRES_USER"); env != "" {
		cfg.Storage.Postgres.User = env
	}
	if env := os.Getenv("HAXEN_STORAGE_POSTGRES_PASSWORD"); env != "" {
		cfg.Storage.Postgres.Password = env
	}
	if env := os.Getenv("HAXEN_STORAGE_POSTGRES_SSLMODE"); env != "" {
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
		cfg.UI.Mode = "dev"
		cfg.UI.Enabled = true
	}

	// Disable execution VC generation if flag is set
	if noVC, err := cmd.Flags().GetBool("no-vc-execution"); err == nil && noVC {
		cfg.Features.DID.VCRequirements.RequireVCForExecution = false
		cfg.Features.DID.VCRequirements.PersistExecutionVC = false
		logger.Logger.Warn().Msg("Execution VC generation disabled via --no-vc-execution flag")
	}

	// Build UI if in embedded mode and not in ui-dev mode and UI is not already embedded
	if cfg.UI.Enabled && cfg.UI.Mode == "embedded" && !uiDev && !client.IsUIEmbedded() {
		fmt.Println("Building UI for embedded mode...")
		if err := buildUI(cfg); err != nil {
			log.Printf("Warning: Failed to build UI, UI might not be available: %v", err)
		} else {
			fmt.Println("UI build successful.")
		}
	} else if cfg.UI.Enabled && cfg.UI.Mode == "embedded" && client.IsUIEmbedded() {
		fmt.Println("UI is already embedded in binary, skipping build.")
	}

	// Create Haxen server instance
	haxenServer, err := server.NewHaxenServer(cfg)
	if err != nil {
		log.Fatalf("Failed to create Haxen server: %v", err)
	}

	// Start the server in a goroutine so we can open the browser
	go func() {
		fmt.Printf("Haxen server attempting to start on port %d...\n", cfg.Haxen.Port)
		if err := haxenServer.Start(); err != nil {
			log.Fatalf("Failed to start Haxen server: %v", err)
		}
	}()

	// Wait a moment for the server to start before opening browser
	time.Sleep(1 * time.Second)

	openBrowserFlag, _ := cmd.Flags().GetBool("open")
	if cfg.UI.Enabled && openBrowserFlag && !backendOnly {
		uiTargetURL := fmt.Sprintf("http://localhost:%d", cfg.Haxen.Port)
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
		openBrowser(uiTargetURL)
	}

	fmt.Printf("Haxen server running on http://localhost:%d\n", cfg.Haxen.Port)
	fmt.Printf("Press Ctrl+C to exit.\n")
	// Keep main goroutine alive
	select {}
}

// loadConfig loads configuration with sensible defaults for user experience
func loadConfig(configFile string) (*config.Config, error) {
	// Set environment variable prefixes
	viper.SetEnvPrefix("HAXEN")
	viper.AutomaticEnv()

	// Get the directory where the binary is located for UI paths
	execPath, err := os.Executable()
	if err != nil {
		log.Printf("Warning: Could not determine executable path: %v", err)
		execPath = "."
	}
	execDir := filepath.Dir(execPath)

	if configFile != "" {
		viper.SetConfigFile(configFile)
	} else {
		// Check for config file path from environment
		if envConfigFile := os.Getenv("HAXEN_CONFIG_FILE"); envConfigFile != "" {
			viper.SetConfigFile(envConfigFile)
		} else {
			// Look for config in user's home directory first, then relative to exec dir, then local
			homeDir, _ := os.UserHomeDir()
			viper.AddConfigPath(filepath.Join(homeDir, ".haxen"))
			viper.AddConfigPath(filepath.Join(execDir, "config"))
			viper.AddConfigPath("./config")
			viper.AddConfigPath(".")
			viper.SetConfigName("haxen")
			viper.SetConfigType("yaml")
		}
	}

	if err := viper.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); ok {
			fmt.Println("No config file found, using environment variables and defaults.")
		} else {
			return nil, fmt.Errorf("failed to read config file: %w", err)
		}
	}

	var cfg config.Config
	if err := viper.Unmarshal(&cfg); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}

	// Apply sensible defaults for user experience
	if cfg.Haxen.Port == 0 {
		cfg.Haxen.Port = 8080
	}
	// Enable UI by default unless explicitly disabled
	if cfg.UI.Mode == "" {
		cfg.UI.Mode = "embedded"
	}
	// Always enable UI by default unless explicitly disabled
	if !viper.IsSet("ui.enabled") {
		cfg.UI.Enabled = true
	}
	if cfg.UI.DevPort == 0 {
		cfg.UI.DevPort = 5173
	}
	if cfg.UI.SourcePath == "" {
		candidateSourcePaths := []string{
			filepath.Join(execDir, "web", "client"),
			filepath.Join(filepath.Dir(execDir), "apps", "platform", "haxen", "web", "client"),
			filepath.Join("apps", "platform", "haxen", "web", "client"),
			filepath.Join("web", "client"),
		}
		for _, candidate := range candidateSourcePaths {
			if _, err := os.Stat(candidate); err == nil {
				cfg.UI.SourcePath = candidate
				break
			}
		}
		if cfg.UI.SourcePath == "" {
			cfg.UI.SourcePath = filepath.Join("web", "client")
		}
	}
	if cfg.UI.DistPath == "" {
		candidateDistPaths := []string{
			filepath.Join(cfg.UI.SourcePath, "dist"),
			filepath.Join(execDir, "web", "client", "dist"),
			filepath.Join(filepath.Dir(execDir), "apps", "platform", "haxen", "web", "client", "dist"),
			filepath.Join("apps", "platform", "haxen", "web", "client", "dist"),
			filepath.Join("web", "client", "dist"),
		}
		for _, candidate := range candidateDistPaths {
			if _, err := os.Stat(candidate); err == nil {
				cfg.UI.DistPath = candidate
				break
			}
		}
		if cfg.UI.DistPath == "" {
			cfg.UI.DistPath = filepath.Join("web", "client", "dist")
		}
	}

	// Ensure VC generation/persistence defaults remain enabled unless explicitly disabled
	if cfg.Features.DID.Enabled {
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

	// Set default storage mode to local if not specified
	if cfg.Storage.Mode == "" {
		cfg.Storage.Mode = "local"
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
		// Ensure all Haxen data directories exist
		if _, err := utils.EnsureDataDirectories(); err != nil {
			return nil, fmt.Errorf("failed to create Haxen data directories: %w", err)
		}
	}

	fmt.Printf("Loaded config - Storage mode: %s, Haxen Port: %d, UI Mode: %s, UI Enabled: %t\n",
		cfg.Storage.Mode, cfg.Haxen.Port, cfg.UI.Mode, cfg.UI.Enabled)

	return &cfg, nil
}

func buildUI(cfg *config.Config) error {
	uiDir := cfg.UI.SourcePath
	if uiDir == "" {
		uiDir = "./web/client"
	}

	// Check if package.json exists
	if _, err := os.Stat(filepath.Join(uiDir, "package.json")); os.IsNotExist(err) {
		log.Printf("UI source path (%s) or package.json not found. Skipping UI build.", uiDir)
		return nil
	}

	fmt.Printf("Building UI in %s...\n", uiDir)

	// Set environment variables for the build process
	buildEnv := os.Environ()

	if cfg.UI.DistPath != "" {
		buildEnv = append(buildEnv, fmt.Sprintf("VITE_BUILD_OUT_DIR=%s", filepath.Base(cfg.UI.DistPath)))
	}

	buildEnv = append(buildEnv, fmt.Sprintf("VITE_API_PROXY_TARGET=http://localhost:%d", cfg.Haxen.Port))

	// Install dependencies
	cmdInstall := exec.Command("npm", "install", "--force")
	cmdInstall.Dir = uiDir
	cmdInstall.Env = buildEnv
	cmdInstall.Stdout = os.Stdout
	cmdInstall.Stderr = os.Stderr
	if err := cmdInstall.Run(); err != nil {
		return fmt.Errorf("failed to install UI dependencies: %w", err)
	}

	// Build for production
	cmdBuild := exec.Command("npm", "run", "build")
	cmdBuild.Dir = uiDir
	cmdBuild.Env = buildEnv
	cmdBuild.Stdout = os.Stdout
	cmdBuild.Stderr = os.Stderr
	if err := cmdBuild.Run(); err != nil {
		return fmt.Errorf("failed to build UI: %w", err)
	}
	return nil
}

func openBrowser(url string) {
	var err error
	switch runtime.GOOS {
	case "linux":
		err = exec.Command("xdg-open", url).Start()
	case "windows":
		err = exec.Command("rundll32", "url.dll,FileProtocolHandler", url).Start()
	case "darwin":
		err = exec.Command("open", url).Start()
	default:
		err = fmt.Errorf("unsupported platform")
	}
	if err != nil {
		log.Printf("Failed to open browser: %v. Please open manually: %s", err, url)
	}
}
