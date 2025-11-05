package framework

import (
	"github.com/spf13/cobra"
	"github.com/your-org/haxen/control-plane/internal/core/interfaces"
	"github.com/your-org/haxen/control-plane/internal/services"
	"github.com/your-org/haxen/control-plane/internal/storage"
)

// Command represents a CLI command that can be built into a Cobra command
type Command interface {
	BuildCobraCommand() *cobra.Command
	GetName() string
	GetDescription() string
}

// ServiceContainer holds all the services that commands might need
type ServiceContainer struct {
	PackageService   interfaces.PackageService
	AgentService     interfaces.AgentService
	DevService       interfaces.DevService
	DIDService       *services.DIDService
	VCService        *services.VCService
	KeystoreService  *services.KeystoreService
	DIDRegistry      *services.DIDRegistry
	StorageProvider  storage.StorageProvider
}

// BaseCommand provides common functionality for all commands
type BaseCommand struct {
	Services *ServiceContainer
}

// CommandRegistry manages registration and building of commands
type CommandRegistry struct {
	commands []Command
}

// NewCommandRegistry creates a new command registry
func NewCommandRegistry() *CommandRegistry {
	return &CommandRegistry{
		commands: make([]Command, 0),
	}
}

// Register adds a command to the registry
func (r *CommandRegistry) Register(cmd Command) {
	r.commands = append(r.commands, cmd)
}

// BuildCobraCommands converts all registered commands to Cobra commands
func (r *CommandRegistry) BuildCobraCommands() []*cobra.Command {
	var cobraCommands []*cobra.Command
	for _, cmd := range r.commands {
		cobraCommands = append(cobraCommands, cmd.BuildCobraCommand())
	}
	return cobraCommands
}

// GetCommands returns all registered commands
func (r *CommandRegistry) GetCommands() []Command {
	return r.commands
}
