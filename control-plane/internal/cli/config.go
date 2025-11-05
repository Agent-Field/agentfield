package cli

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"syscall"

	"github.com/spf13/cobra"
	"golang.org/x/term"
	"gopkg.in/yaml.v3"

	"github.com/your-org/agentfield/control-plane/internal/packages"
)

var (
	configList  bool
	configSet   string
	configUnset string
)

// NewConfigCommand creates the config command
func NewConfigCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "config <package-name>",
		Short: "Configure environment variables for an installed AgentField agent package",
		Long: `Configure environment variables for an installed AgentField agent package.

This command allows you to set up required and optional environment variables
for installed packages. It will prompt for each variable and create a .env file
in the package directory that will be loaded when the agent runs.

Examples:
  af config my-agent                    # Interactive configuration
  af config my-agent --list            # List current configuration
  af config my-agent --set KEY=VALUE   # Set specific variable
  af config my-agent --unset KEY       # Remove variable`,
		Args: cobra.ExactArgs(1),
		Run:  runConfigCommand,
	}

	cmd.Flags().BoolVar(&configList, "list", false, "List current environment configuration")
	cmd.Flags().StringVar(&configSet, "set", "", "Set environment variable (KEY=VALUE)")
	cmd.Flags().StringVar(&configUnset, "unset", "", "Unset environment variable")

	return cmd
}

func runConfigCommand(cmd *cobra.Command, args []string) {
	packageName := args[0]
	agentfieldHome := getAgentFieldHomeDir()

	configManager := &PackageConfigManager{
		AgentFieldHome: agentfieldHome,
	}

	if configList {
		if err := configManager.ListConfig(packageName); err != nil {
			fmt.Printf("❌ Failed to list configuration: %v\n", err)
			os.Exit(1)
		}
		return
	}

	if configSet != "" {
		parts := strings.SplitN(configSet, "=", 2)
		if len(parts) != 2 {
			fmt.Printf("❌ Invalid format. Use KEY=VALUE\n")
			os.Exit(1)
		}
		if err := configManager.SetVariable(packageName, parts[0], parts[1]); err != nil {
			fmt.Printf("❌ Failed to set variable: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("✅ Set %s for package %s\n", parts[0], packageName)
		return
	}

	if configUnset != "" {
		if err := configManager.UnsetVariable(packageName, configUnset); err != nil {
			fmt.Printf("❌ Failed to unset variable: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("✅ Unset %s for package %s\n", configUnset, packageName)
		return
	}

	// Interactive configuration
	if err := configManager.InteractiveConfig(packageName); err != nil {
		fmt.Printf("❌ Configuration failed: %v\n", err)
		os.Exit(1)
	}
}

// PackageConfigManager handles environment configuration for packages
type PackageConfigManager struct {
	AgentFieldHome string
}

// InteractiveConfig runs interactive configuration for a package
func (pcm *PackageConfigManager) InteractiveConfig(packageName string) error {
	fmt.Printf("🔧 Configuring environment variables for: %s\n\n", packageName)

	// Load package metadata
	metadata, packagePath, err := pcm.loadPackageMetadata(packageName)
	if err != nil {
		return err
	}

	// Load existing environment variables
	envVars, err := pcm.loadEnvFile(packagePath)
	if err != nil {
		envVars = make(map[string]string)
	}

	// Configure required variables
	if len(metadata.UserEnvironment.Required) > 0 {
		fmt.Printf("📋 Required Environment Variables:\n")
		for _, envVar := range metadata.UserEnvironment.Required {
			value, err := pcm.promptForVariable(envVar, envVars[envVar.Name])
			if err != nil {
				return err
			}
			if value != "" {
				envVars[envVar.Name] = value
			}
		}
		fmt.Println()
	}

	// Configure optional variables
	if len(metadata.UserEnvironment.Optional) > 0 {
		fmt.Printf("📝 Optional Environment Variables (press Enter to skip):\n")
		for _, envVar := range metadata.UserEnvironment.Optional {
			currentValue := envVars[envVar.Name]
			if currentValue == "" {
				currentValue = envVar.Default
			}

			value, err := pcm.promptForVariable(envVar, currentValue)
			if err != nil {
				return err
			}
			if value != "" {
				envVars[envVar.Name] = value
			} else if envVar.Default != "" {
				envVars[envVar.Name] = envVar.Default
			}
		}
		fmt.Println()
	}

	// Save environment file
	if err := pcm.saveEnvFile(packagePath, envVars); err != nil {
		return fmt.Errorf("failed to save environment file: %w", err)
	}

	fmt.Printf("✅ Environment configuration saved to: %s/.env\n", packagePath)
	fmt.Printf("💡 Run 'af run %s' to start the agent with these settings\n", packageName)

	return nil
}

// promptForVariable prompts the user for a single environment variable
func (pcm *PackageConfigManager) promptForVariable(envVar packages.UserEnvironmentVar, currentValue string) (string, error) {
	// Show current value if it exists
	prompt := fmt.Sprintf("  %s", envVar.Name)
	if currentValue != "" {
		if envVar.Type == "secret" {
			prompt += fmt.Sprintf(" (current: %s)", maskSecret(currentValue))
		} else {
			prompt += fmt.Sprintf(" (current: %s)", currentValue)
		}
	}
	if envVar.Default != "" && currentValue == "" {
		prompt += fmt.Sprintf(" (default: %s)", envVar.Default)
	}
	prompt += fmt.Sprintf("\n    %s\n    > ", envVar.Description)

	fmt.Print(prompt)

	var value string
	var err error

	if envVar.Type == "secret" {
		// Hide input for secrets
		byteValue, err := term.ReadPassword(int(syscall.Stdin))
		if err != nil {
			return "", fmt.Errorf("failed to read secret: %w", err)
		}
		value = string(byteValue)
		fmt.Println() // Add newline after hidden input
	} else {
		// Regular input
		reader := bufio.NewReader(os.Stdin)
		value, err = reader.ReadString('\n')
		if err != nil {
			return "", fmt.Errorf("failed to read input: %w", err)
		}
		value = strings.TrimSpace(value)
	}

	// If empty and we have a current value, keep the current value
	if value == "" && currentValue != "" {
		return currentValue, nil
	}

	// If empty and we have a default, use the default
	if value == "" && envVar.Default != "" {
		return envVar.Default, nil
	}

	// Validate input if validation pattern is provided
	if value != "" && envVar.Validation != "" {
		matched, err := regexp.MatchString(envVar.Validation, value)
		if err != nil {
			return "", fmt.Errorf("invalid validation pattern: %w", err)
		}
		if !matched {
			fmt.Printf("    ❌ Invalid format. Please try again.\n")
			return pcm.promptForVariable(envVar, currentValue)
		}
	}

	return value, nil
}

// ListConfig lists current environment configuration for a package
func (pcm *PackageConfigManager) ListConfig(packageName string) error {
	fmt.Printf("🔧 Environment configuration for: %s\n\n", packageName)

	// Load package metadata
	metadata, packagePath, err := pcm.loadPackageMetadata(packageName)
	if err != nil {
		return err
	}

	// Load existing environment variables
	envVars, err := pcm.loadEnvFile(packagePath)
	if err != nil {
		fmt.Printf("📝 No environment file found (.env)\n\n")
		envVars = make(map[string]string)
	}

	// Show required variables
	if len(metadata.UserEnvironment.Required) > 0 {
		fmt.Printf("📋 Required Variables:\n")
		for _, envVar := range metadata.UserEnvironment.Required {
			value := envVars[envVar.Name]
			if value != "" {
				if envVar.Type == "secret" {
					fmt.Printf("  ✅ %s: %s\n", envVar.Name, maskSecret(value))
				} else {
					fmt.Printf("  ✅ %s: %s\n", envVar.Name, value)
				}
			} else {
				fmt.Printf("  ❌ %s: (not set)\n", envVar.Name)
			}
			fmt.Printf("      %s\n", envVar.Description)
		}
		fmt.Println()
	}

	// Show optional variables
	if len(metadata.UserEnvironment.Optional) > 0 {
		fmt.Printf("📝 Optional Variables:\n")
		for _, envVar := range metadata.UserEnvironment.Optional {
			value := envVars[envVar.Name]
			if value != "" {
				fmt.Printf("  ✅ %s: %s\n", envVar.Name, value)
			} else if envVar.Default != "" {
				fmt.Printf("  📄 %s: %s (default)\n", envVar.Name, envVar.Default)
			} else {
				fmt.Printf("  ⚪ %s: (not set)\n", envVar.Name)
			}
			fmt.Printf("      %s\n", envVar.Description)
		}
		fmt.Println()
	}

	return nil
}

// SetVariable sets a specific environment variable
func (pcm *PackageConfigManager) SetVariable(packageName, key, value string) error {
	// Load package metadata to validate the variable
	metadata, packagePath, err := pcm.loadPackageMetadata(packageName)
	if err != nil {
		return err
	}

	// Find the variable definition
	var envVar *packages.UserEnvironmentVar
	for _, v := range metadata.UserEnvironment.Required {
		if v.Name == key {
			envVar = &v
			break
		}
	}
	if envVar == nil {
		for _, v := range metadata.UserEnvironment.Optional {
			if v.Name == key {
				envVar = &v
				break
			}
		}
	}

	if envVar == nil {
		return fmt.Errorf("unknown environment variable: %s", key)
	}

	// Validate value
	if envVar.Validation != "" {
		matched, err := regexp.MatchString(envVar.Validation, value)
		if err != nil {
			return fmt.Errorf("invalid validation pattern: %w", err)
		}
		if !matched {
			return fmt.Errorf("value does not match required format")
		}
	}

	// Load existing environment variables
	envVars, err := pcm.loadEnvFile(packagePath)
	if err != nil {
		envVars = make(map[string]string)
	}

	// Set the variable
	envVars[key] = value

	// Save environment file
	return pcm.saveEnvFile(packagePath, envVars)
}

// UnsetVariable removes an environment variable
func (pcm *PackageConfigManager) UnsetVariable(packageName, key string) error {
	// Load package metadata
	_, packagePath, err := pcm.loadPackageMetadata(packageName)
	if err != nil {
		return err
	}

	// Load existing environment variables
	envVars, err := pcm.loadEnvFile(packagePath)
	if err != nil {
		return fmt.Errorf("no environment file found")
	}

	// Remove the variable
	delete(envVars, key)

	// Save environment file
	return pcm.saveEnvFile(packagePath, envVars)
}

// loadPackageMetadata loads package metadata and returns the package path
func (pcm *PackageConfigManager) loadPackageMetadata(packageName string) (*packages.PackageMetadata, string, error) {
	// Load registry to get package path
	registryPath := filepath.Join(pcm.AgentFieldHome, "installed.yaml")
	registry := &packages.InstallationRegistry{
		Installed: make(map[string]packages.InstalledPackage),
	}

	if data, err := os.ReadFile(registryPath); err == nil {
		if err := yaml.Unmarshal(data, registry); err != nil {
			return nil, "", fmt.Errorf("failed to parse registry: %w", err)
		}
	}

	installedPackage, exists := registry.Installed[packageName]
	if !exists {
		return nil, "", fmt.Errorf("package %s is not installed", packageName)
	}

	// Load package metadata
	metadataPath := filepath.Join(installedPackage.Path, "agentfield-package.yaml")
	data, err := os.ReadFile(metadataPath)
	if err != nil {
		return nil, "", fmt.Errorf("failed to read package metadata: %w", err)
	}

	var metadata packages.PackageMetadata
	if err := yaml.Unmarshal(data, &metadata); err != nil {
		return nil, "", fmt.Errorf("failed to parse package metadata: %w", err)
	}

	return &metadata, installedPackage.Path, nil
}

// loadEnvFile loads environment variables from .env file
func (pcm *PackageConfigManager) loadEnvFile(packagePath string) (map[string]string, error) {
	envPath := filepath.Join(packagePath, ".env")

	data, err := os.ReadFile(envPath)
	if err != nil {
		return nil, err
	}

	envVars := make(map[string]string)
	lines := strings.Split(string(data), "\n")

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		parts := strings.SplitN(line, "=", 2)
		if len(parts) == 2 {
			key := strings.TrimSpace(parts[0])
			value := strings.TrimSpace(parts[1])

			// Remove quotes if present
			if (strings.HasPrefix(value, "\"") && strings.HasSuffix(value, "\"")) ||
				(strings.HasPrefix(value, "'") && strings.HasSuffix(value, "'")) {
				value = value[1 : len(value)-1]
			}

			envVars[key] = value
		}
	}

	return envVars, nil
}

// saveEnvFile saves environment variables to .env file
func (pcm *PackageConfigManager) saveEnvFile(packagePath string, envVars map[string]string) error {
	envPath := filepath.Join(packagePath, ".env")

	var lines []string
	lines = append(lines, "# Environment variables for AgentField agent")
	lines = append(lines, "# Generated by 'af config' command")
	lines = append(lines, "")

	for key, value := range envVars {
		// Quote values that contain spaces or special characters
		if strings.ContainsAny(value, " \t\n\r\"'\\$") {
			value = fmt.Sprintf("\"%s\"", strings.ReplaceAll(value, "\"", "\\\""))
		}
		lines = append(lines, fmt.Sprintf("%s=%s", key, value))
	}

	content := strings.Join(lines, "\n") + "\n"
	return os.WriteFile(envPath, []byte(content), 0600) // Restrictive permissions for secrets
}

// maskSecret masks a secret value for display
func maskSecret(value string) string {
	if len(value) <= 8 {
		return strings.Repeat("*", len(value))
	}
	return value[:4] + strings.Repeat("*", len(value)-8) + value[len(value)-4:]
}
