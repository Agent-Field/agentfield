package mcp

import (
	"fmt"
	"path/filepath"
	"strconv"
	"strings"
)

// TemplateProcessor handles template variable processing for MCP commands
type TemplateProcessor struct {
	projectDir string
	dataDir    string
	verbose    bool
}

// TemplateVars holds all available template variables
type TemplateVars struct {
	Port       int    `json:"port"`
	DataDir    string `json:"data_dir"`
	ConfigFile string `json:"config_file"`
	LogFile    string `json:"log_file"`
	ServerDir  string `json:"server_dir"`
	ProjectDir string `json:"project_dir"`
	Alias      string `json:"alias"`
}

// NewTemplateProcessor creates a new template processor
func NewTemplateProcessor(projectDir string, verbose bool) *TemplateProcessor {
	dataDir := filepath.Join(projectDir, "packages", "mcp")
	return &TemplateProcessor{
		projectDir: projectDir,
		dataDir:    dataDir,
		verbose:    verbose,
	}
}

// ProcessCommand processes template variables in a single command string
func (tp *TemplateProcessor) ProcessCommand(command string, vars TemplateVars) (string, error) {
	if command == "" {
		return "", nil
	}

	processed := command

	// Replace all template variables
	processed = strings.ReplaceAll(processed, "{{port}}", strconv.Itoa(vars.Port))
	processed = strings.ReplaceAll(processed, "{{data_dir}}", vars.DataDir)
	processed = strings.ReplaceAll(processed, "{{config_file}}", vars.ConfigFile)
	processed = strings.ReplaceAll(processed, "{{log_file}}", vars.LogFile)
	processed = strings.ReplaceAll(processed, "{{server_dir}}", vars.ServerDir)
	processed = strings.ReplaceAll(processed, "{{project_dir}}", vars.ProjectDir)
	processed = strings.ReplaceAll(processed, "{{alias}}", vars.Alias)

	if tp.verbose {
		fmt.Printf("Template processing: %s -> %s\n", command, processed)
	}

	return processed, nil
}

// ProcessCommands processes template variables in multiple command strings
func (tp *TemplateProcessor) ProcessCommands(commands []string, vars TemplateVars) ([]string, error) {
	if len(commands) == 0 {
		return nil, nil
	}

	processed := make([]string, len(commands))
	for i, command := range commands {
		processedCmd, err := tp.ProcessCommand(command, vars)
		if err != nil {
			return nil, fmt.Errorf("error processing command %d: %w", i, err)
		}
		processed[i] = processedCmd
	}

	return processed, nil
}

// CreateTemplateVars creates template variables for the given configuration and port
func (tp *TemplateProcessor) CreateTemplateVars(config MCPServerConfig, port int) TemplateVars {
	serverDir := filepath.Join(tp.dataDir, config.Alias)

	return TemplateVars{
		Port:       port,
		DataDir:    tp.dataDir,
		ConfigFile: filepath.Join(serverDir, "config.json"),
		LogFile:    filepath.Join(serverDir, fmt.Sprintf("%s.log", config.Alias)),
		ServerDir:  serverDir,
		ProjectDir: tp.projectDir,
		Alias:      config.Alias,
	}
}

// ProcessEnvironment processes template variables in environment variable map
func (tp *TemplateProcessor) ProcessEnvironment(env map[string]string, vars TemplateVars) (map[string]string, error) {
	if len(env) == 0 {
		return nil, nil
	}

	processed := make(map[string]string, len(env))
	for key, value := range env {
		processedValue, err := tp.ProcessCommand(value, vars)
		if err != nil {
			return nil, fmt.Errorf("error processing environment variable %s: %w", key, err)
		}
		processed[key] = processedValue
	}

	return processed, nil
}
