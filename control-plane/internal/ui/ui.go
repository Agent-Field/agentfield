// Package ui provides a small set of shared, lipgloss-based rendering helpers so
// the AgentField CLI presents a consistent, styled look — bordered panels,
// column tables, and status badges — across commands. The palette matches the
// existing `af init` flow (magenta titles, cyan accents, green/red status).
package ui

import (
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/lipgloss/table"
)

// Palette. 256-color codes chosen to match the existing `af init` styling.
var (
	colTitle   = lipgloss.Color("205") // magenta — titles
	colAccent  = lipgloss.Color("86")  // cyan — headers / accents
	colSuccess = lipgloss.Color("42")  // green — running / ok
	colError   = lipgloss.Color("196") // red — error
	colMuted   = lipgloss.Color("241") // gray — secondary text
)

var (
	titleStyle  = lipgloss.NewStyle().Bold(true).Foreground(colTitle)
	headerStyle = lipgloss.NewStyle().Bold(true).Foreground(colAccent)
	mutedStyle  = lipgloss.NewStyle().Foreground(colMuted)
	cellStyle   = lipgloss.NewStyle().Padding(0, 1)
	panelStyle  = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(colTitle).
			Padding(0, 1)
)

// Title renders bold magenta title text (no newline).
func Title(s string) string { return titleStyle.Render(s) }

// Muted renders secondary gray text.
func Muted(s string) string { return mutedStyle.Render(s) }

// Panel wraps body in a rounded border with an optional title line above it.
func Panel(title, body string) string {
	box := panelStyle.Render(body)
	if title == "" {
		return box
	}
	return titleStyle.Render(title) + "\n" + box
}

// Table renders a bordered table with a styled header row. An optional title is
// printed above the table. Columns are auto-sized by lipgloss.
func Table(title string, headers []string, rows [][]string) string {
	t := table.New().
		Border(lipgloss.RoundedBorder()).
		BorderStyle(lipgloss.NewStyle().Foreground(colTitle)).
		Headers(headers...).
		Rows(rows...).
		StyleFunc(func(row, _ int) lipgloss.Style {
			if row == table.HeaderRow {
				return headerStyle.Padding(0, 1)
			}
			return cellStyle
		})
	out := t.Render()
	if title != "" {
		return titleStyle.Render(title) + "\n" + out
	}
	return out
}

// StatusBadge returns a colored ●/○ badge for a node status string.
func StatusBadge(status string) string {
	switch status {
	case "running":
		return lipgloss.NewStyle().Foreground(colSuccess).Render("● running")
	case "error":
		return lipgloss.NewStyle().Foreground(colError).Render("● error")
	case "":
		return mutedStyle.Render("○ —")
	default:
		return mutedStyle.Render("○ " + status)
	}
}

// SuccessPanel renders a green-bordered panel — for completion summaries.
func SuccessPanel(title, body string) string {
	box := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(colSuccess).
		Padding(0, 1).
		Render(body)
	if title == "" {
		return box
	}
	return lipgloss.NewStyle().Bold(true).Foreground(colSuccess).Render(title) + "\n" + box
}

// KV renders aligned "key  value" lines with muted keys, for detail blocks.
func KV(pairs [][2]string) string {
	width := 0
	for _, p := range pairs {
		if len(p[0]) > width {
			width = len(p[0])
		}
	}
	var b strings.Builder
	for i, p := range pairs {
		if i > 0 {
			b.WriteByte('\n')
		}
		key := mutedStyle.Render(p[0] + strings.Repeat(" ", width-len(p[0])))
		b.WriteString(key + "  " + p[1])
	}
	return b.String()
}
