package cli

import (
	"encoding/json"
	"fmt"

	"github.com/spf13/cobra"

	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
)

// requirementVar is the JSON/text shape of one configurable variable.
type requirementVar struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Type        string `json:"type,omitempty"`
	Default     string `json:"default,omitempty"`
	Validation  string `json:"validation,omitempty"`
	Scope       string `json:"scope,omitempty"`
}

// requirementGroup mirrors a require_one_of group.
type requirementGroup struct {
	ID          string           `json:"id,omitempty"`
	Description string           `json:"description,omitempty"`
	Options     []requirementVar `json:"options"`
}

// requirementsReport is the full machine-readable answer for one source.
type requirementsReport struct {
	Node         string             `json:"node"`
	Version      string             `json:"version,omitempty"`
	Description  string             `json:"description,omitempty"`
	Language     string             `json:"language,omitempty"`
	Source       string             `json:"source"`
	Required     []requirementVar   `json:"required"`
	Optional     []requirementVar   `json:"optional"`
	RequireOneOf []requirementGroup `json:"require_one_of"`
}

// NewShowRequirementsCommand returns `af show-requirements`, which reports the
// environment a node needs BEFORE installing it — the gap that previously forced
// users to install first just to discover what to configure.
func NewShowRequirementsCommand() *cobra.Command {
	var output string
	cmd := &cobra.Command{
		Use:   "show-requirements <path-or-git-url>",
		Short: "Show the environment variables a node needs, without installing it",
		Long: `Inspect an agent node's agentfield-package.yaml and print the environment it
needs — required variables, optional variables with their defaults, and
require_one_of groups — WITHOUT installing anything.

The source can be a local directory path or a Git URL (with an optional @ref and
//subdir selector). A Git source is shallow-cloned into a temporary directory
that is removed afterwards; nothing is written under ~/.agentfield.

Examples:
  af show-requirements ./my-agent
  af show-requirements https://github.com/Agent-Field/pr-af
  af show-requirements https://github.com/Agent-Field/SWE-AF//go
  af show-requirements ./my-agent -o json`,
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			if output != "" && output != "text" && output != "json" {
				return fmt.Errorf("invalid --output %q (want \"text\" or \"json\")", output)
			}

			metadata, err := packages.InspectSource(args[0])
			if err != nil {
				return err
			}

			report := buildRequirementsReport(args[0], metadata)
			if output == "json" {
				return printRequirementsJSON(report)
			}
			printRequirementsText(report)
			return nil
		},
	}
	cmd.Flags().StringVarP(&output, "output", "o", "text", "output format: text or json")
	return cmd
}

func toRequirementVar(v packages.UserEnvironmentVar) requirementVar {
	return requirementVar{
		Name:        v.Name,
		Description: v.Description,
		Type:        v.Type,
		Default:     v.Default,
		Validation:  v.Validation,
		Scope:       v.Scope,
	}
}

func buildRequirementsReport(source string, m *packages.PackageMetadata) requirementsReport {
	report := requirementsReport{
		Node:         m.Name,
		Version:      m.Version,
		Description:  m.Description,
		Language:     m.Language,
		Source:       source,
		Required:     []requirementVar{},
		Optional:     []requirementVar{},
		RequireOneOf: []requirementGroup{},
	}
	for _, v := range m.UserEnvironment.Required {
		report.Required = append(report.Required, toRequirementVar(v))
	}
	for _, v := range m.UserEnvironment.Optional {
		report.Optional = append(report.Optional, toRequirementVar(v))
	}
	for _, g := range m.UserEnvironment.RequireOneOf {
		grp := requirementGroup{ID: g.ID, Description: g.Description, Options: []requirementVar{}}
		for _, o := range g.Options {
			grp.Options = append(grp.Options, toRequirementVar(o))
		}
		report.RequireOneOf = append(report.RequireOneOf, grp)
	}
	return report
}

func printRequirementsJSON(report requirementsReport) error {
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return err
	}
	fmt.Println(string(data))
	return nil
}

func printRequirementsText(report requirementsReport) {
	header := packages.Bold(report.Node)
	if report.Version != "" {
		header += " " + packages.Gray("v"+report.Version)
	}
	fmt.Printf("%s\n", header)
	if report.Description != "" {
		fmt.Printf("  %s\n", packages.Gray(report.Description))
	}

	if len(report.Required) == 0 && len(report.RequireOneOf) == 0 && len(report.Optional) == 0 {
		fmt.Printf("\n%s\n", packages.Gray("This node needs no user configuration."))
		fmt.Printf("\n%s %s\n", packages.Blue("→"), packages.Bold("Install: af install "+report.Source))
		return
	}

	if len(report.Required) > 0 {
		fmt.Printf("\n%s\n", packages.Bold("Required environment variables:"))
		for _, v := range report.Required {
			printReqVar(v, report.Node)
		}
	}

	for _, g := range report.RequireOneOf {
		label := g.Description
		if label == "" {
			label = "one of these"
		}
		fmt.Printf("\n%s (%s):\n", packages.Bold("At least one of"), label)
		for _, v := range g.Options {
			printReqVar(v, report.Node)
		}
	}

	if len(report.Optional) > 0 {
		fmt.Printf("\n%s\n", packages.Gray("Optional environment variables (with defaults):"))
		for _, v := range report.Optional {
			line := "  " + packages.Bold(v.Name)
			if v.Default != "" {
				line += " = " + v.Default + " " + packages.Gray("(default)")
			}
			fmt.Println(line)
			if v.Description != "" {
				fmt.Printf("      %s\n", packages.Gray(v.Description))
			}
		}
	}

	fmt.Printf("\n%s %s\n", packages.Blue("→"), packages.Bold("Install: af install "+report.Source))
}

// printReqVar renders one required/group variable with the exact command that
// supplies it, so the reader can configure it before (or after) installing.
func printReqVar(v requirementVar, node string) {
	line := "  " + packages.Bold(v.Name)
	if v.Type != "" {
		line += " " + packages.Gray("("+v.Type+")")
	}
	fmt.Println(line)
	if v.Description != "" {
		fmt.Printf("      %s\n", packages.Gray(v.Description))
	}
	fmt.Printf("      %s\n", packages.Cyan(fmt.Sprintf("af secrets set %s --node %s", v.Name, node)))
}
