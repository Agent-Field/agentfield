package framework

import (
	"fmt"
	"github.com/fatih/color"
)

// OutputFormatter provides consistent output formatting for all commands
type OutputFormatter struct {
	verbose bool
}

// NewOutputFormatter creates a new output formatter
func NewOutputFormatter(verbose bool) *OutputFormatter {
	return &OutputFormatter{verbose: verbose}
}

// PrintSuccess prints a success message with green checkmark
func (o *OutputFormatter) PrintSuccess(message string) {
	fmt.Printf("✅ %s\n", color.GreenString(message))
}

// PrintError prints an error message with red X
func (o *OutputFormatter) PrintError(message string) {
	fmt.Printf("❌ %s\n", color.RedString(message))
}

// PrintInfo prints an info message with blue info icon
func (o *OutputFormatter) PrintInfo(message string) {
	fmt.Printf("ℹ️  %s\n", color.BlueString(message))
}

// PrintWarning prints a warning message with yellow warning icon
func (o *OutputFormatter) PrintWarning(message string) {
	fmt.Printf("⚠️  %s\n", color.YellowString(message))
}

// PrintHeader prints a header message with haxen emoji and bold text
func (o *OutputFormatter) PrintHeader(message string) {
	fmt.Printf("\n%s %s\n", color.CyanString("🧠"), color.New(color.Bold).Sprint(message))
}

// PrintVerbose prints a verbose message only if verbose mode is enabled
func (o *OutputFormatter) PrintVerbose(message string) {
	if o.verbose {
		fmt.Printf("🔍 %s\n", color.New(color.Faint).Sprint(message))
	}
}

// PrintProgress prints a progress message with spinner-like icon
func (o *OutputFormatter) PrintProgress(message string) {
	fmt.Printf("⏳ %s\n", color.YellowString(message))
}

// SetVerbose updates the verbose setting
func (o *OutputFormatter) SetVerbose(verbose bool) {
	o.verbose = verbose
}

// IsVerbose returns whether verbose mode is enabled
func (o *OutputFormatter) IsVerbose() bool {
	return o.verbose
}
