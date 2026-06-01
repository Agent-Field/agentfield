package framework

import (
	"testing"

	"github.com/spf13/cobra"
)

type testCommand struct {
	name        string
	description string
	cobraCmd    *cobra.Command
}

func (c *testCommand) BuildCobraCommand() *cobra.Command {
	return c.cobraCmd
}

func (c *testCommand) GetName() string {
	return c.name
}

func (c *testCommand) GetDescription() string {
	return c.description
}

func TestNewCommandRegistry(t *testing.T) {
	registry := NewCommandRegistry()
	if registry == nil {
		t.Fatal("expected registry")
	}
	if registry.commands == nil {
		t.Fatal("expected commands slice to be initialized")
	}
	if len(registry.GetCommands()) != 0 {
		t.Fatalf("expected empty registry, got %d commands", len(registry.GetCommands()))
	}
}

func TestCommandRegistryRegisterAndGetCommands(t *testing.T) {
	registry := NewCommandRegistry()
	first := &testCommand{name: "first", description: "first command", cobraCmd: &cobra.Command{Use: "first"}}
	second := &testCommand{name: "second", description: "second command", cobraCmd: &cobra.Command{Use: "second"}}

	registry.Register(first)
	registry.Register(second)

	got := registry.GetCommands()
	if len(got) != 2 {
		t.Fatalf("expected 2 commands, got %d", len(got))
	}
	if got[0] != first || got[1] != second {
		t.Fatalf("unexpected command order: %#v", got)
	}
	if got[0].GetName() != "first" || got[1].GetDescription() != "second command" {
		t.Fatalf("unexpected command metadata: %q %q", got[0].GetName(), got[1].GetDescription())
	}
}

func TestCommandRegistryBuildCobraCommands(t *testing.T) {
	tests := []struct {
		name      string
		commands   []Command
		wantUses   []string
	}{
		{
			name:    "empty registry",
			wantUses: nil,
		},
		{
			name: "multiple commands",
			commands: []Command{
				&testCommand{cobraCmd: &cobra.Command{Use: "alpha"}},
				&testCommand{cobraCmd: &cobra.Command{Use: "beta"}},
			},
			wantUses: []string{"alpha", "beta"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			registry := NewCommandRegistry()
			for _, cmd := range tt.commands {
				registry.Register(cmd)
			}

			got := registry.BuildCobraCommands()
			if len(got) != len(tt.wantUses) {
				t.Fatalf("expected %d cobra commands, got %d", len(tt.wantUses), len(got))
			}
			for i, use := range tt.wantUses {
				if got[i].Use != use {
					t.Fatalf("expected command %d use %q, got %q", i, use, got[i].Use)
				}
			}
		})
	}
}
