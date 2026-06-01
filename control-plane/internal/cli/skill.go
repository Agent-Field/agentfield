package cli

import (
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/fatih/color"
	"github.com/spf13/cobra"

	"github.com/Agent-Field/agentfield/control-plane/internal/skillkit"
)

// NewSkillCommand builds the `af skill` command tree. The skill subsystem
// embeds skill content into the af binary, installs it into multiple
// coding-agent integrations (Claude Code, Codex, Gemini, OpenCode, Aider,
// Windsurf, Cursor), and tracks state in ~/.agentfield/skills/.state.json.
//
// Mirrors the plandb installer pattern but lives inside the binary so that
// existing af users can run `af skill install` directly without re-running
// the install.sh shell bootstrapper.
func NewSkillCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "skill",
		Short: "Install and manage AgentField skills across coding agents",
		Long: `Manage AgentField skills bundled with the af binary.

A skill is a self-contained instruction packet (a SKILL.md file plus
reference markdown) that teaches a coding agent (Claude Code, Codex,
Gemini, etc.) how to use AgentField properly. The af binary ships with
the agentfield-multi-reasoner-builder skill embedded — install it once
into every agent you use and they will know how to architect, scaffold,
and ship multi-reasoner systems on AgentField.

Examples:
  af skill install                       # Interactive picker (default)
  af skill install --all                 # All detected agents, no prompt
  af skill install --all-targets         # Every registered agent, even undetected
  af skill install --target claude-code  # Just one agent
  af skill list                          # Show what is installed where
  af skill update                        # Re-install at the binary's embedded version
  af skill uninstall                     # Remove from all targets
  af skill print                         # Print SKILL.md to stdout (pipe to clipboard)
  af skill path                          # Print canonical store location`,
	}

	cmd.AddCommand(newSkillInstallCommand())
	cmd.AddCommand(newSkillListCommand())
	cmd.AddCommand(newSkillUpdateCommand())
	cmd.AddCommand(newSkillUninstallCommand())
	cmd.AddCommand(newSkillPrintCommand())
	cmd.AddCommand(newSkillPathCommand())
	cmd.AddCommand(newSkillCatalogCommand())

	return cmd
}

// ── install ──────────────────────────────────────────────────────────────

func newSkillInstallCommand() *cobra.Command {
	var (
		skillName    string
		version      string
		targets      []string
		allDetected  bool
		allTargets   bool
		force        bool
		dryRun       bool
		nonInteractive bool
	)

	cmd := &cobra.Command{
		Use:   "install [skill-name]",
		Short: "Install a skill into one or more coding-agent integrations",
		Args:  cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			if len(args) == 1 {
				skillName = args[0]
			}

			// If no targets explicitly chosen and not in --all/--all-targets mode,
			// run the interactive picker.
			if len(targets) == 0 && !allDetected && !allTargets && !nonInteractive {
				picked, err := runInteractivePicker()
				if err != nil {
					return err
				}
				if len(picked) == 0 {
					printInfo("Skill install skipped — no targets selected")
					return nil
				}
				targets = picked
			}

			report, err := skillkit.Install(skillkit.InstallOptions{
				SkillName:     skillName,
				Version:       version,
				Targets:       targets,
				AllDetected:   allDetected,
				AllRegistered: allTargets,
				Force:         force,
				DryRun:        dryRun,
			})
			if err != nil {
				return err
			}
			printInstallReport(report, dryRun)
			return nil
		},
	}

	cmd.Flags().StringVar(&skillName, "skill", "", "Skill name to install (defaults to the first/only skill in the catalog)")
	cmd.Flags().StringVar(&version, "version", "", "Specific skill version to install (defaults to the version embedded in the binary)")
	cmd.Flags().StringSliceVar(&targets, "target", nil, "Specific target(s) to install into (claude-code, codex, gemini, opencode, aider, windsurf, cursor). Repeatable.")
	cmd.Flags().BoolVar(&allDetected, "all", false, "Install into every detected target without prompting")
	cmd.Flags().BoolVar(&allTargets, "all-targets", false, "Install into every registered target even if not detected on this machine")
	cmd.Flags().BoolVar(&force, "force", false, "Reinstall even if the same version is already present in state")
	cmd.Flags().BoolVar(&dryRun, "dry-run", false, "Print the planned operations without writing")
	cmd.Flags().BoolVar(&nonInteractive, "non-interactive", false, "Skip the interactive picker; default to detected targets")

	return cmd
}

// ── list ─────────────────────────────────────────────────────────────────

func newSkillListCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "list",
		Short: "List installed skills and their target integrations",
		RunE: func(cmd *cobra.Command, args []string) error {
			state, err := skillkit.ListInstalled()
			if err != nil {
				return err
			}
			printSkillList(state)
			return nil
		},
	}
}

// ── update ───────────────────────────────────────────────────────────────

func newSkillUpdateCommand() *cobra.Command {
	var skillName string
	cmd := &cobra.Command{
		Use:   "update [skill-name]",
		Short: "Re-install a skill at the binary's embedded version into every target it is currently installed at",
		Args:  cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			if len(args) == 1 {
				skillName = args[0]
			}
			report, err := skillkit.Update(skillName)
			if err != nil {
				return err
			}
			printInstallReport(report, false)
			return nil
		},
	}
	cmd.Flags().StringVar(&skillName, "skill", "", "Skill name to update")
	return cmd
}

// ── uninstall ────────────────────────────────────────────────────────────

func newSkillUninstallCommand() *cobra.Command {
	var (
		skillName       string
		targets         []string
		removeCanonical bool
	)
	cmd := &cobra.Command{
		Use:   "uninstall [skill-name]",
		Short: "Remove a skill from one or more targets",
		Args:  cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			if len(args) == 1 {
				skillName = args[0]
			}
			err := skillkit.Uninstall(skillkit.UninstallOptions{
				SkillName:       skillName,
				Targets:         targets,
				RemoveCanonical: removeCanonical,
			})
			if err != nil {
				return err
			}
			printSuccess("Skill uninstalled")
			return nil
		},
	}
	cmd.Flags().StringVar(&skillName, "skill", "", "Skill name to uninstall")
	cmd.Flags().StringSliceVar(&targets, "target", nil, "Specific target(s) to uninstall from (default: all installed targets)")
	cmd.Flags().BoolVar(&removeCanonical, "remove-canonical", false, "Also delete the canonical ~/.agentfield/skills/<name>/ directory")
	return cmd
}

// ── print ────────────────────────────────────────────────────────────────

func newSkillPrintCommand() *cobra.Command {
	var skillName string
	cmd := &cobra.Command{
		Use:   "print [skill-name]",
		Short: "Print SKILL.md for the named skill to stdout",
		Args:  cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			if len(args) == 1 {
				skillName = args[0]
			}
			if skillName == "" {
				skillName = skillkit.Catalog[0].Name
			}
			skill, err := skillkit.CatalogByName(skillName)
			if err != nil {
				return err
			}
			content, err := skill.EntryContent()
			if err != nil {
				return err
			}
			_, err = os.Stdout.Write(content)
			return err
		},
	}
	cmd.Flags().StringVar(&skillName, "skill", "", "Skill name (defaults to the first in the catalog)")
	return cmd
}

// ── path ─────────────────────────────────────────────────────────────────

func newSkillPathCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "path",
		Short: "Print the canonical skill store location (~/.agentfield/skills)",
		RunE: func(cmd *cobra.Command, args []string) error {
			root, err := skillkit.CanonicalRoot()
			if err != nil {
				return err
			}
			fmt.Println(root)
			return nil
		},
	}
}

// ── catalog ──────────────────────────────────────────────────────────────

func newSkillCatalogCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "catalog",
		Short: "List skills bundled with this af binary",
		RunE: func(cmd *cobra.Command, args []string) error {
			bold := color.New(color.Bold)
			bold.Println("Skills shipped with this af binary")
			fmt.Println()
			for _, s := range skillkit.Catalog {
				bold.Printf("  %s ", s.Name)
				color.New(color.FgCyan).Printf("v%s\n", s.Version)
				fmt.Printf("    %s\n\n", s.Description)
			}
			fmt.Println("Install with:")
			fmt.Println("  af skill install                # interactive picker")
			fmt.Println("  af skill install --all          # all detected agents")
			fmt.Println("  af skill install --target codex # one specific agent")
			return nil
		},
	}
}

// ── interactive picker ───────────────────────────────────────────────────

func runInteractivePicker() ([]string, error) {
	bold := color.New(color.Bold)
	cyan := color.New(color.FgCyan)
	dim := color.New(color.Faint)
	green := color.New(color.FgGreen)

	bold.Println("\nInstall agentfield-multi-reasoner-builder skill")
	fmt.Println()
	fmt.Println("  This skill teaches any coding agent how to architect and ship")
	fmt.Println("  multi-reasoner systems on AgentField. It uses composite-")
	fmt.Println("  intelligence patterns: parallel reasoner hunters, dynamic")
	fmt.Println("  routing, deep DAG composition, safe-default fallbacks, and")
	fmt.Println("  the canonical scaffold-to-curl workflow.")
	fmt.Println()
	bold.Println("  Targets")
	fmt.Println()

	targets := skillkit.AllTargets()
	for i, t := range targets {
		marker := dim.Sprint("○ ")
		suffix := ""
		if t.Detected() {
			marker = green.Sprint("● ")
			suffix = dim.Sprint(" (detected)")
		}
		fmt.Printf("    %s%d. %s%s\n", marker, i+1, t.DisplayName(), suffix)
	}

	fmt.Println()
	bold.Println("  Options")
	fmt.Println()
	fmt.Printf("    %s   Install into all detected targets\n", cyan.Sprint("a"))
	fmt.Printf("    %s   Install into ALL targets (even undetected)\n", cyan.Sprint("A"))
	fmt.Printf("    %s   Skip skill install\n", cyan.Sprint("n"))
	fmt.Printf("    %s   Toggle individual targets (comma-separated)\n", cyan.Sprint("1-7"))
	fmt.Println()
	fmt.Printf("  Choice [%s]: ", cyan.Sprint("a"))

	var choice string
	if _, err := fmt.Scanln(&choice); err != nil {
		// blank input → default
		choice = "a"
	}
	choice = strings.TrimSpace(choice)
	if choice == "" {
		choice = "a"
	}

	switch choice {
	case "a":
		var picked []string
		for _, t := range targets {
			if t.Detected() {
				picked = append(picked, t.Name())
			}
		}
		return picked, nil
	case "A":
		var picked []string
		for _, t := range targets {
			picked = append(picked, t.Name())
		}
		return picked, nil
	case "n", "N":
		return nil, nil
	default:
		var picked []string
		for _, num := range strings.Split(choice, ",") {
			num = strings.TrimSpace(num)
			var idx int
			if _, err := fmt.Sscanf(num, "%d", &idx); err == nil {
				if idx >= 1 && idx <= len(targets) {
					picked = append(picked, targets[idx-1].Name())
				}
			}
		}
		return picked, nil
	}
}

// ── output rendering ─────────────────────────────────────────────────────

func printInstallReport(report *skillkit.InstallReport, dryRun bool) {
	bold := color.New(color.Bold)
	green := color.New(color.FgGreen)
	yellow := color.New(color.FgYellow)
	red := color.New(color.FgRed)
	cyan := color.New(color.FgCyan)

	fmt.Println()
	if dryRun {
		bold.Println("Skill install — DRY RUN")
	} else {
		bold.Println("Skill install")
	}
	fmt.Println()

	cyan.Printf("  Skill:        %s v%s\n", report.Skill.Name, report.Skill.Version)
	cyan.Printf("  Canonical:    %s\n", report.CanonicalDir)
	cyan.Printf("  Current link: %s\n", report.CurrentLink)
	fmt.Println()

	if len(report.TargetsInstalled) > 0 {
		bold.Println("  Installed")
		for _, t := range report.TargetsInstalled {
			green.Printf("    ✓ %-12s ", t.TargetName)
			fmt.Printf("(%s) %s\n", t.Method, t.Path)
		}
		fmt.Println()
	}

	if len(report.TargetsSkipped) > 0 {
		bold.Println("  Skipped")
		for _, s := range report.TargetsSkipped {
			yellow.Printf("    ○ %-12s ", s.TargetName)
			fmt.Printf("(%s)\n", s.Reason)
		}
		fmt.Println()
	}

	if len(report.TargetsFailed) > 0 {
		bold.Println("  Failed")
		for _, e := range report.TargetsFailed {
			red.Printf("    ✗ %-12s ", e.TargetName)
			fmt.Printf("%s\n", e.Err)
		}
		fmt.Println()
	}

	bold.Println("  Verify")
	fmt.Println("    af skill list")
	fmt.Println()
	bold.Println("  Use")
	fmt.Println("    Open Claude Code / Codex / etc. and ask:")
	fmt.Println(`    "Build me a multi-reasoner agent on AgentField that..."`)
	fmt.Println("    The skill will fire automatically.")
	fmt.Println()
}

func printSkillList(state *skillkit.State) {
	bold := color.New(color.Bold)
	dim := color.New(color.Faint)
	green := color.New(color.FgGreen)
	cyan := color.New(color.FgCyan)
	yellow := color.New(color.FgYellow)

	fmt.Println()
	bold.Println("Installed skills")
	fmt.Println()

	if len(state.Skills) == 0 {
		dim.Println("  No skills installed yet.")
		fmt.Println()
		fmt.Println("  Install with:")
		fmt.Println("    af skill install            # interactive picker")
		fmt.Println("    af skill install --all      # all detected agents")
		fmt.Println()
		return
	}

	skillNames := make([]string, 0, len(state.Skills))
	for name := range state.Skills {
		skillNames = append(skillNames, name)
	}
	sort.Strings(skillNames)

	for _, name := range skillNames {
		s := state.Skills[name]
		bold.Printf("  %s ", name)
		cyan.Printf("v%s\n", s.CurrentVersion)
		dim.Printf("    Installed: %s\n", s.InstalledAt.Format("2006-01-02 15:04:05 MST"))
		dim.Printf("    Versions available locally: %s\n", strings.Join(s.AvailableVersions, ", "))

		if len(s.Targets) == 0 {
			yellow.Println("    (no active target integrations)")
			fmt.Println()
			continue
		}

		fmt.Println("    Targets:")
		for _, tname := range s.SortedTargetNames() {
			t := s.Targets[tname]
			green.Printf("      ✓ %-12s ", tname)
			fmt.Printf("v%s  %s  %s\n", t.Version, t.Method, t.Path)
		}
		fmt.Println()
	}
}
