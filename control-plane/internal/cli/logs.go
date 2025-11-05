package cli

import (
	"errors"
	"fmt"
	"os"
	"os/exec" // Added missing import
	"path/filepath"

	"github.com/spf13/cobra"
	"github.com/your-org/agentfield/control-plane/internal/logger"
	"github.com/your-org/agentfield/control-plane/internal/packages"
	"gopkg.in/yaml.v3"
)

var (
	logsFollow bool
	logsTail   int
)

// NewLogsCommand creates the logs command
func NewLogsCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "logs <agent-node-name>",
		Short: "View logs for a AgentField agent node",
		Long: `Display logs for an installed AgentField agent node package.

Shows the most recent log entries from the agent node's log file.

Examples:
  af logs email-helper
  af logs data-analyzer --follow`,
		Args: cobra.ExactArgs(1),
		Run:  runLogsCommand,
	}

	cmd.Flags().BoolVarP(&logsFollow, "follow", "f", false, "Follow log output")
	cmd.Flags().IntVarP(&logsTail, "tail", "n", 50, "Number of lines to show from the end")

	return cmd
}

func runLogsCommand(cmd *cobra.Command, args []string) {
	agentNodeName := args[0]

	logViewer := &LogViewer{
		AgentFieldHome: getAgentFieldHomeDir(),
		Follow:         logsFollow,
		Tail:           logsTail,
	}

	if err := logViewer.ViewLogs(agentNodeName); err != nil {
		logger.Logger.Error().Err(err).Msg("Failed to view logs")
		os.Exit(1)
	}
}

// LogViewer handles viewing agent node logs
type LogViewer struct {
	AgentFieldHome string
	Follow         bool
	Tail           int
}

// ViewLogs displays logs for an agent node
func (lv *LogViewer) ViewLogs(agentNodeName string) error {
	// Load registry to get log file path
	registryPath := filepath.Join(lv.AgentFieldHome, "installed.yaml")
	registry := &packages.InstallationRegistry{
		Installed: make(map[string]packages.InstalledPackage),
	}

	if data, err := os.ReadFile(registryPath); err == nil {
		if err := yaml.Unmarshal(data, registry); err != nil {
			return fmt.Errorf("failed to parse registry: %w", err)
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("failed to read registry: %w", err)
	}

	agentNode, exists := registry.Installed[agentNodeName]
	if !exists {
		return fmt.Errorf("agent node %s not installed", agentNodeName)
	}

	logFile := agentNode.Runtime.LogFile
	if _, err := os.Stat(logFile); os.IsNotExist(err) {
		logger.Logger.Info().Msgf("📝 No logs found for %s", agentNodeName)
		logger.Logger.Info().Msg("💡 Logs will appear here when the agent node is running")
		return nil
	}

	logger.Logger.Info().Msgf("📝 Logs for %s:", agentNodeName)
	logger.Logger.Info().Msgf("📁 %s\n", logFile)

	if lv.Follow {
		return lv.followLogs(logFile)
	} else {
		return lv.tailLogs(logFile, lv.Tail)
	}
}

// tailLogs shows the last N lines of the log file
func (lv *LogViewer) tailLogs(logFile string, lines int) error {
	cmd := exec.Command("tail", "-n", fmt.Sprintf("%d", lines), logFile)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// followLogs follows the log file in real-time
func (lv *LogViewer) followLogs(logFile string) error {
	cmd := exec.Command("tail", "-f", logFile)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}
