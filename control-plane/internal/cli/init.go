package cli

import (
	"fmt"
	"net"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"

	"github.com/your-org/haxen/control-plane/internal/logger"
	"github.com/your-org/haxen/control-plane/internal/templates"
)

// NewInitCommand builds a fresh Cobra command for initializing a new agent project.
func NewInitCommand() *cobra.Command {
	var authorName string
	var authorEmail string

	cmd := &cobra.Command{
		Use:   "init [project-name]",
		Short: "Initialize a new Haxen agent project",
		Long: `Initialize a new Haxen agent project with a predefined
 directory structure and essential files.
 
 This command sets up a new project, including:
 - Python project structure (pyproject.toml, main.py, agent/...)
 - Basic agent implementation with example reasoner/skill
 - README.md, LICENSE, and .gitignore files
 - Configuration for connecting to the Haxen control plane
 
 Example:
   haxen init my-new-agent --author "John Doe" --email "john.doe@example.com"`,
		Args: cobra.ExactArgs(1),
		Run: func(cmd *cobra.Command, args []string) {
			projectName := args[0]

			if !isValidProjectName(projectName) {
				logger.Errorf("Error: Invalid project name '%s'. Project name must be lowercase, alphanumeric, and can contain hyphens or underscores.", projectName)
				os.Exit(1)
			}

			if authorName == "" {
				authorName = promptForInput("Enter author name (e.g., John Doe):")
			}
			if authorEmail == "" {
				authorEmail = promptForInput("Enter author email (e.g., john.doe@example.com):")
			}

			nodeID := generateNodeID(projectName)
			port, err := getFreePort()
			if err != nil {
				logger.Errorf("Error: Could not find a free port: %v", err)
				os.Exit(1)
			}

			data := templates.TemplateData{
				ProjectName: projectName,
				NodeID:      nodeID,
				Port:        port,
				CreatedAt:   time.Now().Format("2006-01-02 15:04:05 MST"),
				AuthorName:  authorName,
				AuthorEmail: authorEmail,
				CurrentYear: time.Now().Year(),
			}

			projectPath := filepath.Join(".", projectName)
			if err := os.MkdirAll(projectPath, 0o755); err != nil {
				logger.Errorf("Error creating project directory: %v", err)
				os.Exit(1)
			}

			fmt.Printf("Creating new agent project...\n")

			templateFiles, err := templates.GetTemplateFiles()
			if err != nil {
				PrintError(fmt.Sprintf("Error getting template files: %v", err))
				os.Exit(1)
			}

			spinner := NewSpinner("Generating project structure")
			spinner.Start()

			for tmplPath, destPath := range templateFiles {
				tmpl, err := templates.GetTemplate(tmplPath)
				if err != nil {
					spinner.Error("Failed to parse templates")
					PrintError(fmt.Sprintf("Error parsing template %s: %v", tmplPath, err))
					os.Exit(1)
				}

				var buf strings.Builder
				if err := tmpl.Execute(&buf, data); err != nil {
					spinner.Error("Failed to execute templates")
					PrintError(fmt.Sprintf("Error executing template %s: %v", tmplPath, err))
					os.Exit(1)
				}

				fullDestPath := filepath.Join(projectPath, destPath)
				if err := os.MkdirAll(filepath.Dir(fullDestPath), 0o755); err != nil {
					spinner.Error("Failed to create directories")
					PrintError(fmt.Sprintf("Error creating directory for %s: %v", fullDestPath, err))
					os.Exit(1)
				}
				if err := os.WriteFile(fullDestPath, []byte(buf.String()), 0o644); err != nil {
					spinner.Error("Failed to write files")
					PrintError(fmt.Sprintf("Error writing file %s: %v", fullDestPath, err))
					os.Exit(1)
				}
			}

			spinner.Success("Project structure created")
			PrintSuccess(fmt.Sprintf("Created %s in ./%s", projectName, projectName))

			fmt.Printf("\nNext steps:\n")
			PrintInfo(fmt.Sprintf("cd %s", projectName))
			PrintInfo("haxen install .")
			PrintInfo(fmt.Sprintf("haxen run %s", projectName))
		},
	}

	cmd.Flags().StringVarP(&authorName, "author", "a", "", "Author name for the project")
	cmd.Flags().StringVarP(&authorEmail, "email", "e", "", "Author email for the project")

	viper.BindPFlag("author.name", cmd.Flags().Lookup("author"))
	viper.BindPFlag("author.email", cmd.Flags().Lookup("email"))

	return cmd
}

// isValidProjectName checks if the project name is valid (lowercase, alphanumeric, hyphens/underscores).
func isValidProjectName(name string) bool {
	match, _ := regexp.MatchString("^[a-z0-9_-]+$", name)
	return match
}

// promptForInput prompts the user for input if a value is not provided via flag.
func promptForInput(prompt string) string {
	fmt.Print(prompt + " ")
	var input string
	fmt.Scanln(&input)
	return strings.TrimSpace(input)
}

// generateNodeID generates a unique node ID based on the project name.
func generateNodeID(projectName string) string {
	name := strings.ToLower(projectName)
	name = strings.ReplaceAll(name, "_", "-")
	collapse := regexp.MustCompile("-+")
	name = collapse.ReplaceAllString(name, "-")
	return strings.Trim(name, "-")
}

// getFreePort finds an available TCP port.
func getFreePort() (int, error) {
	addr, err := net.ResolveTCPAddr("tcp", "localhost:0")
	if err != nil {
		return 0, err
	}
	l, err := net.ListenTCP("tcp", addr)
	if err != nil {
		return 0, err
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port, nil
}
