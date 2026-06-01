package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

type cliContextKey struct{}

// CLIError represents an error encountered while handling CLI input.
// Code follows common CLI semantics: 0 success, 1 runtime error, 2 usage error.
type CLIError struct {
	Code int
	Err  error
}

const (
	ansiReset = "\033[0m"
	ansiBold  = "\033[1m"
	ansiCyan  = "\033[36m"
)

func (e *CLIError) Error() string {
	if e == nil || e.Err == nil {
		return ""
	}
	return e.Err.Error()
}

func (e *CLIError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

func (e *CLIError) ExitCode() int {
	if e == nil || e.Code == 0 {
		return 0
	}
	return e.Code
}

func colorText(enabled bool, code string, text string) string {
	if !enabled {
		return text
	}
	return code + text + ansiReset
}

type cliInvocation struct {
	command      string
	outputFormat string
	input        map[string]any
	setValues    map[string]string
	help         bool
	helpTarget   string
	version      bool
	useColor     bool
}

type cliContext struct {
	args         map[string]string
	command      string
	outputFormat string
	useColor     bool
}

// IsCLIMode returns true if the current execution is in CLI mode.
func IsCLIMode(ctx context.Context) bool {
	_, ok := ctx.Value(cliContextKey{}).(cliContext)
	return ok
}

// GetCLIArgs returns parsed CLI arguments stored on the context.
// Keys include user-supplied --set flags plus metadata keys: __command, __output, __color.
func GetCLIArgs(ctx context.Context) map[string]string {
	cliCtx, ok := ctx.Value(cliContextKey{}).(cliContext)
	if !ok {
		return nil
	}
	out := make(map[string]string, len(cliCtx.args))
	for k, v := range cliCtx.args {
		out[k] = v
	}
	return out
}

func (a *Agent) runCLI(ctx context.Context, args []string) error {
	if !a.hasCLIReasoners() {
		return &CLIError{Code: 2, Err: errors.New("no CLI reasoners registered; add agent.WithCLI() to a reasoner")}
	}

	inv, err := a.parseCLIArgs(args)
	if err != nil {
		a.printHelp("", inv.useColor)
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		return &CLIError{Code: 2, Err: err}
	}

	switch {
	case inv.version || inv.command == "version":
		a.printVersion()
		return nil
	case inv.command == "list":
		a.printList(inv.useColor)
		return nil
	case inv.command == "help" || inv.help:
		a.printHelp(inv.helpTarget, inv.useColor)
		return nil
	}

	reasonerName := inv.command
	if reasonerName == "" {
		reasonerName = a.defaultCLIReasoner
	}
	if reasonerName == "" {
		a.printHelp("", inv.useColor)
		return &CLIError{Code: 2, Err: errors.New("no default CLI reasoner configured")}
	}

	reasoner, ok := a.reasoners[reasonerName]
	if !ok || !reasoner.CLIEnabled {
		return &CLIError{Code: 2, Err: fmt.Errorf("reasoner %q is not available for CLI use", reasonerName)}
	}

	ctx = withCLIContext(ctx, cliContext{
		args:         buildCLIArgMap(inv),
		command:      reasonerName,
		outputFormat: inv.outputFormat,
		useColor:     inv.useColor,
	})

	result, execErr := a.Execute(ctx, reasonerName, inv.input)

	formatter := reasoner.CLIFormatter
	if formatter == nil {
		formatter = defaultFormatter(inv.outputFormat, inv.useColor)
	}

	formatter(ctx, result, execErr)
	if execErr != nil {
		return &CLIError{Code: 1, Err: execErr}
	}

	return nil
}

func (a *Agent) parseCLIArgs(args []string) (cliInvocation, error) {
	inv := cliInvocation{
		setValues:    make(map[string]string),
		useColor:     a.cfg.CLIConfig == nil || !a.cfg.CLIConfig.DisableColors,
		outputFormat: "pretty",
	}
	if cfg := a.cfg.CLIConfig; cfg != nil && strings.TrimSpace(cfg.DefaultOutputFormat) != "" {
		inv.outputFormat = strings.ToLower(strings.TrimSpace(cfg.DefaultOutputFormat))
	}

	var rawInput string
	var inputFile string
	for i := 0; i < len(args); i++ {
		arg := args[i]
		switch {
		case arg == "-h" || arg == "--help":
			inv.help = true
		case arg == "--version":
			inv.version = true
		case strings.HasPrefix(arg, "--set="):
			if err := applySet(inv.setValues, strings.TrimPrefix(arg, "--set=")); err != nil {
				return inv, err
			}
		case arg == "--set":
			if i+1 >= len(args) {
				return inv, errors.New("missing key=value after --set")
			}
			i++
			if err := applySet(inv.setValues, args[i]); err != nil {
				return inv, err
			}
		case strings.HasPrefix(arg, "--input="):
			rawInput = strings.TrimPrefix(arg, "--input=")
		case arg == "--input":
			if i+1 >= len(args) {
				return inv, errors.New("missing value for --input")
			}
			i++
			rawInput = args[i]
		case strings.HasPrefix(arg, "--input-file="):
			inputFile = strings.TrimPrefix(arg, "--input-file=")
		case arg == "--input-file":
			if i+1 >= len(args) {
				return inv, errors.New("missing value for --input-file")
			}
			i++
			inputFile = args[i]
		case strings.HasPrefix(arg, "--output="):
			inv.outputFormat = strings.ToLower(strings.TrimSpace(strings.TrimPrefix(arg, "--output=")))
		case arg == "--output":
			if i+1 >= len(args) {
				return inv, errors.New("missing value for --output")
			}
			i++
			inv.outputFormat = strings.ToLower(strings.TrimSpace(args[i]))
		case arg == "--no-color":
			inv.useColor = false
		default:
			if strings.HasPrefix(arg, "-") {
				return inv, fmt.Errorf("unknown flag %s", arg)
			}
			if inv.command == "" {
				inv.command = arg
			} else if (inv.command == "help" || inv.help) && inv.helpTarget == "" {
				inv.helpTarget = arg
			} else {
				return inv, fmt.Errorf("unexpected argument %s", arg)
			}
		}
	}

	if inv.outputFormat == "" {
		inv.outputFormat = "pretty"
	}
	if !isSupportedOutput(inv.outputFormat) {
		return inv, fmt.Errorf("unsupported output format %q", inv.outputFormat)
	}

	stdinInput, err := parseJSONFromStdin()
	if err != nil {
		return inv, err
	}
	fileInput, err := parseJSONFromFile(inputFile)
	if err != nil {
		return inv, err
	}
	inputFromFlag, err := decodeJSONInput(rawInput)
	if err != nil {
		return inv, err
	}

	finalInput := mergeInput(stdinInput, fileInput, inputFromFlag, inv.setValues)
	inv.input = finalInput

	return inv, nil
}

func buildCLIArgMap(inv cliInvocation) map[string]string {
	args := make(map[string]string, len(inv.setValues)+3)
	for k, v := range inv.setValues {
		args[k] = v
	}
	args["__command"] = inv.command
	args["__output"] = inv.outputFormat
	if inv.useColor {
		args["__color"] = "true"
	} else {
		args["__color"] = "false"
	}
	return args
}

func isSupportedOutput(format string) bool {
	switch strings.ToLower(format) {
	case "json", "pretty", "yaml":
		return true
	default:
		return false
	}
}

func applySet(target map[string]string, value string) error {
	if value == "" {
		return errors.New("empty --set value")
	}

	parts := strings.SplitN(value, "=", 2)
	if len(parts) != 2 {
		return fmt.Errorf("expected key=value, got %q", value)
	}
	key := strings.TrimSpace(parts[0])
	val := parts[1]
	if key == "" {
		return fmt.Errorf("missing key in %q", value)
	}
	target[key] = val
	return nil
}

func parseJSONFromStdin() (map[string]any, error) {
	info, err := os.Stdin.Stat()
	if err != nil {
		return nil, err
	}
	if info.Mode()&os.ModeCharDevice != 0 {
		return nil, nil
	}

	data, err := io.ReadAll(os.Stdin)
	if err != nil {
		return nil, err
	}
	return decodeJSONInput(string(data))
}

func parseJSONFromFile(path string) (map[string]any, error) {
	if strings.TrimSpace(path) == "" {
		return nil, nil
	}
	content, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read input file: %w", err)
	}
	return decodeJSONInput(string(content))
}

func decodeJSONInput(raw string) (map[string]any, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, nil
	}

	var parsed map[string]any
	if err := json.Unmarshal([]byte(raw), &parsed); err != nil {
		return nil, fmt.Errorf("parse JSON input: %w", err)
	}
	return parsed, nil
}

func mergeInput(stdin, file, flag map[string]any, setValues map[string]string) map[string]any {
	merged := make(map[string]any)

	for _, source := range []map[string]any{stdin, file, flag} {
		for k, v := range source {
			merged[k] = v
		}
	}

	for k, v := range setValues {
		merged[k] = parseScalar(v)
	}

	return merged
}

func parseScalar(raw string) any {
	if raw == "" {
		return ""
	}

	var decoded any
	if err := json.Unmarshal([]byte(raw), &decoded); err == nil {
		return decoded
	}

	// fallback: treat as string
	return raw
}

func defaultFormatter(format string, useColor bool) func(context.Context, any, error) {
	return func(_ context.Context, result any, err error) {
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			return
		}
		if result == nil {
			return
		}

		switch strings.ToLower(format) {
		case "json":
			data, encErr := json.Marshal(result)
			if encErr != nil {
				fmt.Fprintf(os.Stderr, "Error encoding JSON: %v\n", encErr)
				return
			}
			fmt.Println(string(data))
		case "pretty":
			data, encErr := json.MarshalIndent(result, "", "  ")
			if encErr != nil {
				fmt.Fprintf(os.Stderr, "Error encoding JSON: %v\n", encErr)
				return
			}
			fmt.Println(string(data))
		case "yaml":
			data, encErr := yaml.Marshal(result)
			if encErr != nil {
				fmt.Fprintf(os.Stderr, "Error encoding YAML: %v\n", encErr)
				return
			}
			fmt.Print(string(data))
		default:
			fmt.Fprintf(os.Stderr, "Unknown output format %s\n", format)
		}
	}
}

func (a *Agent) printList(useColor bool) {
	reasoners := make([]*Reasoner, 0, len(a.reasoners))
	for _, r := range a.reasoners {
		if r.CLIEnabled {
			reasoners = append(reasoners, r)
		}
	}
	sort.Slice(reasoners, func(i, j int) bool { return reasoners[i].Name < reasoners[j].Name })

	if len(reasoners) == 0 {
		fmt.Println("No CLI reasoners registered.")
		return
	}

	fmt.Println(colorText(useColor, ansiBold, "Available reasoners:"))
	for _, r := range reasoners {
		label := r.Name
		if r.DefaultCLI || a.defaultCLIReasoner == r.Name {
			label += " (default)"
		}
		label = colorText(useColor, ansiCyan, label)
		if desc := strings.TrimSpace(r.Description); desc != "" {
			fmt.Printf("  %s - %s\n", label, desc)
		} else {
			fmt.Printf("  %s\n", label)
		}
	}
}

func (a *Agent) printHelp(reasonerName string, useColor bool) {
	cfg := a.cfg.CLIConfig
	appName := strings.TrimSpace(filepath.Base(os.Args[0]))
	if cfg != nil && strings.TrimSpace(cfg.AppName) != "" {
		appName = strings.TrimSpace(cfg.AppName)
	}
	appDesc := ""
	if cfg != nil {
		appDesc = strings.TrimSpace(cfg.AppDescription)
	}

	title := appName
	if appDesc != "" {
		title = fmt.Sprintf("%s - %s", appName, appDesc)
	}
	fmt.Println(colorText(useColor, ansiBold, title))
	fmt.Println()

	if cfg != nil && strings.TrimSpace(cfg.HelpPreamble) != "" {
		fmt.Println(strings.TrimSpace(cfg.HelpPreamble))
		fmt.Println()
	}

	fmt.Println(colorText(useColor, ansiBold, "Usage:"))
	fmt.Printf("  %s [command] [flags]\n\n", appName)

	if reasonerName == "" {
		fmt.Println(colorText(useColor, ansiBold, "Available Commands:"))
		fmt.Println("  serve          Start agent server")
		fmt.Println("  list           List available reasoners")
		fmt.Println("  help [command] Show help information")
		fmt.Println("  version        Display version information")

		reasoners := make([]*Reasoner, 0, len(a.reasoners))
		for _, r := range a.reasoners {
			if r.CLIEnabled {
				reasoners = append(reasoners, r)
			}
		}
		sort.Slice(reasoners, func(i, j int) bool { return reasoners[i].Name < reasoners[j].Name })
		if len(reasoners) > 0 {
			fmt.Println()
			fmt.Println(colorText(useColor, ansiBold, "Reasoners:"))
			for _, r := range reasoners {
				name := r.Name
				if r.DefaultCLI || a.defaultCLIReasoner == r.Name {
					name += " (default)"
				}
				if r.Description != "" {
					fmt.Printf("  %s  %s\n", name, r.Description)
				} else {
					fmt.Printf("  %s\n", name)
				}
			}
		}
	} else {
		r, ok := a.reasoners[reasonerName]
		if !ok {
			fmt.Printf("\nUnknown reasoner %q\n", reasonerName)
		} else {
			fmt.Printf("\nReasoner: %s\n", reasonerName)
			if strings.TrimSpace(r.Description) != "" {
				fmt.Printf("  %s\n", strings.TrimSpace(r.Description))
			}
		}
	}

	fmt.Println()
	fmt.Println(colorText(useColor, ansiBold, "Flags:"))
	fmt.Println("  --set key=value   Set individual input parameters (repeatable)")
	fmt.Println("  --input <json>    Provide input as JSON string")
	fmt.Println("  --input-file <p>  Load input from JSON file")
	fmt.Println("  --output <fmt>    Output format: json, pretty, yaml")
	fmt.Println("  --no-color        Disable colorized output")
	fmt.Println("  --help            Show help information")

	if cfg != nil && len(cfg.EnvironmentVars) > 0 {
		fmt.Println()
		fmt.Println(colorText(useColor, ansiBold, "Environment Variables:"))
		for _, env := range cfg.EnvironmentVars {
			fmt.Printf("  %s\n", strings.TrimSpace(env))
		}
	}

	if cfg != nil && strings.TrimSpace(cfg.HelpEpilog) != "" {
		fmt.Println()
		fmt.Println(strings.TrimSpace(cfg.HelpEpilog))
	}
}

func (a *Agent) printVersion() {
	fmt.Printf("AgentField SDK: v%s\n", sdkVersion)
	fmt.Printf("Agent: %s v%s\n", a.cfg.NodeID, a.cfg.Version)
	fmt.Printf("Go: %s\n", runtime.Version())
}

func withCLIContext(ctx context.Context, cliCtx cliContext) context.Context {
	return context.WithValue(ctx, cliContextKey{}, cliCtx)
}

func (a *Agent) hasCLIReasoners() bool {
	for _, r := range a.reasoners {
		if r.CLIEnabled {
			return true
		}
	}
	return false
}
