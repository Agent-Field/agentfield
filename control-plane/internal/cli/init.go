package cli

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/fatih/color"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"

	"github.com/your-org/agentfield/control-plane/internal/templates"
)

var (
	// Styles for Bubble Tea UI
	titleStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("205")).
			MarginBottom(1)

	promptStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("86"))

	selectedStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("205")).
			Bold(true)

	helpStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("241"))

	errorStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("196")).
			Bold(true)

	successStyle = lipgloss.NewStyle(). //nolint:unused // Reserved for future use
			Foreground(lipgloss.Color("42")).
			Bold(true)
)

// initModel represents the state of the interactive init flow
type initModel struct {
	step           int
	projectName    string
	language       string
	authorName     string
	authorEmail    string
	cursor         int
	choices        []string
	textInput      string
	err            error
	done           bool
	nonInteractive bool //nolint:unused // Reserved for future use
}

func (m initModel) Init() tea.Cmd {
	return nil
}

func (m initModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c", "q":
			m.done = true
			return m, tea.Quit

		case "enter":
			return m.handleEnter()

		case "up", "k":
			if m.step == 1 && len(m.choices) > 0 {
				m.cursor--
				if m.cursor < 0 {
					m.cursor = len(m.choices) - 1
				}
			}

		case "down", "j":
			if m.step == 1 && len(m.choices) > 0 {
				m.cursor++
				if m.cursor >= len(m.choices) {
					m.cursor = 0
				}
			}

		case "backspace":
			if len(m.textInput) > 0 {
				m.textInput = m.textInput[:len(m.textInput)-1]
			}

		default:
			// Handle text input for project name, author name and email
			if m.step == 0 || m.step == 2 || m.step == 3 {
				m.textInput += msg.String()
			}
		}
	}

	return m, nil
}

func (m initModel) handleEnter() (tea.Model, tea.Cmd) {
	switch m.step {
	case 0: // Project name
		if m.textInput != "" && isValidProjectName(m.textInput) {
			m.projectName = m.textInput
			m.step++
			m.textInput = ""
			m.err = nil
		} else if m.textInput != "" {
			m.err = fmt.Errorf("invalid project name (use lowercase, letters, numbers, hyphens, underscores)")
		}

	case 1: // Language selection
		if m.cursor >= 0 && m.cursor < len(m.choices) {
			m.language = strings.ToLower(m.choices[m.cursor])
			m.step++
			m.textInput = ""
		}

	case 2: // Author name
		if m.textInput != "" {
			m.authorName = m.textInput
			m.step++
			m.textInput = ""
		}

	case 3: // Author email
		if m.textInput != "" && isValidEmail(m.textInput) {
			m.authorEmail = m.textInput
			m.done = true
			return m, tea.Quit
		} else if m.textInput != "" {
			m.err = fmt.Errorf("invalid email format")
		}
	}

	return m, nil
}

func (m initModel) View() string {
	if m.done {
		return ""
	}

	var s strings.Builder

	// Title
	s.WriteString(titleStyle.Render("🎯 Creating AgentField Agent") + "\n\n")

	switch m.step {
	case 0: // Project name
		s.WriteString(promptStyle.Render("? Project name: ") + m.textInput + "█\n")
		if m.err != nil {
			s.WriteString("\n" + errorStyle.Render("✗ "+m.err.Error()) + "\n")
		}
		s.WriteString("\n" + helpStyle.Render("Use lowercase, letters, numbers, hyphens, underscores"))

	case 1: // Language selection
		s.WriteString(promptStyle.Render("? Select language:") + "\n\n")
		for i, choice := range m.choices {
			cursor := " "
			if m.cursor == i {
				cursor = "❯"
				s.WriteString(selectedStyle.Render(fmt.Sprintf("%s %s\n", cursor, choice)))
			} else {
				s.WriteString(fmt.Sprintf("%s %s\n", cursor, choice))
			}
		}
		s.WriteString("\n" + helpStyle.Render("Use ↑/↓ to navigate, Enter to select"))

	case 2: // Author name
		s.WriteString(promptStyle.Render("? Author name: ") + m.textInput + "█\n\n")
		s.WriteString(helpStyle.Render("Press Enter to continue"))

	case 3: // Author email
		s.WriteString(promptStyle.Render("? Author email: ") + m.textInput + "█\n")
		if m.err != nil {
			s.WriteString("\n" + errorStyle.Render("✗ "+m.err.Error()) + "\n")
		}
		s.WriteString("\n" + helpStyle.Render("Press Enter to continue"))
	}

	return s.String()
}

// NewInitCommand builds a fresh Cobra command for initializing a new agent project.
func NewInitCommand() *cobra.Command {
	var authorName string
	var authorEmail string
	var language string
	var nonInteractive bool

	cmd := &cobra.Command{
		Use:   "init [project-name]",
		Short: "Initialize a new AgentField agent project",
		Long: `Initialize a new AgentField agent project with a predefined
directory structure and essential files.

This command sets up a new project, including:
- Language-specific project structure (Python or Go)
- Basic agent implementation with example reasoner
- README.md and .gitignore files
- Configuration for connecting to the AgentField control plane

Example:
  af init                    # Interactive mode
  af init my-new-agent       # With project name
  af init my-agent --language python
  af init my-agent -l go --author "John Doe" --email "john@example.com"`,
		Args: cobra.MaximumNArgs(1),
		Run: func(cmd *cobra.Command, args []string) {
			var projectName string

			// Get project name from args or interactively
			if len(args) > 0 {
				projectName = args[0]

				// Validate project name
				if !isValidProjectName(projectName) {
					printError("Invalid project name '%s'", projectName)
					fmt.Println("\nProject names must:")
					fmt.Println("  • Be lowercase")
					fmt.Println("  • Use letters, numbers, hyphens, or underscores")
					fmt.Println("  • Start with a letter")
					fmt.Println("  • Not contain spaces or special characters")
					fmt.Println("\nExamples:")
					fmt.Println("  ✓ my-agent")
					fmt.Println("  ✓ user_analytics")
					fmt.Println("  ✓ support-team")
					os.Exit(1)
				}
			}

			// Start interactive mode if project name not provided or other fields missing
			if projectName == "" || (language == "" && !nonInteractive) {
				startStep := 0
				if projectName != "" {
					startStep = 1 // Skip project name prompt if already provided
				}

				model := initModel{
					step:        startStep,
					projectName: projectName,
					choices:     templates.GetSupportedLanguages(),
					cursor:      0,
				}

				p := tea.NewProgram(model)
				finalModel, err := p.Run()
				if err != nil {
					printError("Error running interactive prompt: %v", err)
					os.Exit(1)
				}

				m := finalModel.(initModel)
				if projectName == "" {
					projectName = m.projectName
				}
				language = m.language
				authorName = m.authorName
				authorEmail = m.authorEmail
			} else if language == "" {
				language = "python" // Default to Python
			}

			// Validate language
			supportedLangs := templates.GetSupportedLanguages()
			isSupported := false
			for _, lang := range supportedLangs {
				if lang == language {
					isSupported = true
					break
				}
			}
			if !isSupported {
				printError("Unsupported language: %s", language)
				fmt.Printf("Supported languages: %s\n", strings.Join(supportedLangs, ", "))
				os.Exit(1)
			}

			// Get author info from flags or git config
			if authorName == "" {
				authorName = getGitConfig("user.name")
				if authorName == "" {
					authorName = "Your Name"
				}
			}
			if authorEmail == "" {
				authorEmail = getGitConfig("user.email")
				if authorEmail == "" {
					authorEmail = "you@example.com"
				}
			}

			// Generate node ID (same as project name)
			nodeID := generateNodeID(projectName)

			// Prepare template data
			data := templates.TemplateData{
				ProjectName: projectName,
				NodeID:      nodeID,
				GoModule:    projectName, // Use project name as Go module
				AuthorName:  authorName,
				AuthorEmail: authorEmail,
				CurrentYear: time.Now().Year(),
				CreatedAt:   time.Now().Format("2006-01-02 15:04:05 MST"),
				Language:    language,
			}

			// Create project directory
			projectPath := filepath.Join(".", projectName)
			if err := os.MkdirAll(projectPath, 0o755); err != nil {
				printError("Error creating project directory: %v", err)
				os.Exit(1)
			}

			// Get template files for the selected language
			templateFiles, err := templates.GetTemplateFiles(language)
			if err != nil {
				printError("Error getting template files: %v", err)
				os.Exit(1)
			}

			// Generate files
			printInfo("✨ Creating project structure...")
			for tmplPath, destPath := range templateFiles {
				tmpl, err := templates.GetTemplate(tmplPath)
				if err != nil {
					printError("Error parsing template %s: %v", tmplPath, err)
					os.Exit(1)
				}

				var buf strings.Builder
				if err := tmpl.Execute(&buf, data); err != nil {
					printError("Error executing template %s: %v", tmplPath, err)
					os.Exit(1)
				}

				fullDestPath := filepath.Join(projectPath, destPath)
				if err := os.MkdirAll(filepath.Dir(fullDestPath), 0o755); err != nil {
					printError("Error creating directory for %s: %v", fullDestPath, err)
					os.Exit(1)
				}

				if err := os.WriteFile(fullDestPath, []byte(buf.String()), 0o644); err != nil {
					printError("Error writing file %s: %v", fullDestPath, err)
					os.Exit(1)
				}

				printSuccess("  ✓ %s", destPath)
			}

			// Print success message
			fmt.Println()
			printSuccess("🚀 Agent '%s' created successfully!", projectName)
			fmt.Println()
			printInfo("📁 Location: ./%s", projectName)
			fmt.Println()
			fmt.Println("Next steps:")
			fmt.Println("  1. cd " + projectName)

			if language == "python" {
				fmt.Println("  2. pip install -r requirements.txt")
			} else if language == "go" {
				fmt.Println("  2. go mod download")
			}

			fmt.Println("  3. af server                    # Start AgentField server")

			if language == "python" {
				fmt.Println("  4. python main.py                  # Start your agent")
			} else if language == "go" {
				fmt.Println("  4. go run .                        # Start your agent")
			}

			fmt.Println()
			fmt.Println("Test it:")
			fmt.Printf("  curl -X POST http://localhost:8080/api/v1/execute/%s.demo_echo \\\n", nodeID)
			fmt.Println("    -H \"Content-Type: application/json\" \\")
			fmt.Println("    -d '{\"input\": {\"message\": \"Hello!\"}}'")
			fmt.Println()
			fmt.Println("Enable AI:")
			fmt.Println("  1. Uncomment ai_config in main." + getFileExtension(language))

			if language == "python" {
				fmt.Println("  2. Set API key: export OPENAI_API_KEY=sk-...")
				fmt.Println("     (or ANTHROPIC_API_KEY, GOOGLE_API_KEY, etc.)")
				fmt.Println("  3. Uncomment analyze_sentiment in reasoners.py")
			} else if language == "go" {
				fmt.Println("  2. Set API key: export OPENAI_API_KEY=sk-...")
				fmt.Println("     (or OPENROUTER_API_KEY for OpenRouter)")
				fmt.Println("  3. Uncomment analyze_sentiment in reasoners.go")
			}

			fmt.Println("  4. Restart your agent")
			fmt.Println()
			printInfo("Learn more: https://docs.agentfield.ai")
			fmt.Println()
			printSuccess("Happy building! 🎉")
		},
	}

	cmd.Flags().StringVarP(&language, "language", "l", "", "Language for the agent (python or go)")
	cmd.Flags().StringVarP(&authorName, "author", "a", "", "Author name for the project")
	cmd.Flags().StringVarP(&authorEmail, "email", "e", "", "Author email for the project")
	cmd.Flags().BoolVar(&nonInteractive, "non-interactive", false, "Run in non-interactive mode (use defaults)")

	if err := viper.BindPFlag("language", cmd.Flags().Lookup("language")); err != nil {
		printError("failed to bind language flag: %v", err)
	}
	if err := viper.BindPFlag("author.name", cmd.Flags().Lookup("author")); err != nil {
		printError("failed to bind author flag: %v", err)
	}
	if err := viper.BindPFlag("author.email", cmd.Flags().Lookup("email")); err != nil {
		printError("failed to bind email flag: %v", err)
	}

	return cmd
}

// isValidProjectName checks if the project name is valid (lowercase, alphanumeric, hyphens/underscores).
func isValidProjectName(name string) bool {
	match, _ := regexp.MatchString("^[a-z][a-z0-9_-]*$", name)
	return match
}

// isValidEmail checks if the email is valid.
func isValidEmail(email string) bool {
	match, _ := regexp.MatchString(`^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`, email)
	return match
}

// generateNodeID generates a unique node ID based on the project name.
func generateNodeID(projectName string) string {
	name := strings.ToLower(projectName)
	name = strings.ReplaceAll(name, "_", "-")
	collapse := regexp.MustCompile("-+")
	name = collapse.ReplaceAllString(name, "-")
	return strings.Trim(name, "-")
}

// getGitConfig retrieves a git config value.
func getGitConfig(key string) string {
	cmd := exec.Command("git", "config", "--get", key)
	output, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(output))
}

// getFileExtension returns the file extension for the language.
func getFileExtension(language string) string {
	switch language {
	case "python":
		return "py"
	case "go":
		return "go"
	default:
		return "txt"
	}
}

// printSuccess prints a success message in green.
func printSuccess(format string, args ...interface{}) {
	green := color.New(color.FgGreen, color.Bold)
	green.Printf(format+"\n", args...)
}

// printInfo prints an info message in cyan.
func printInfo(format string, args ...interface{}) {
	cyan := color.New(color.FgCyan)
	cyan.Printf(format+"\n", args...)
}

// printError prints an error message in red.
func printError(format string, args ...interface{}) {
	red := color.New(color.FgRed, color.Bold)
	red.Printf("❌ Error: "+format+"\n", args...)
}
