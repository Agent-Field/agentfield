package application

import (
	"path/filepath"

	"github.com/Agent-Field/agentfield/control-plane/internal/cli/framework"
	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/core/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/encryption"
	"github.com/Agent-Field/agentfield/control-plane/internal/infrastructure/process"
	"github.com/Agent-Field/agentfield/control-plane/internal/infrastructure/storage"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	didServices "github.com/Agent-Field/agentfield/control-plane/internal/services"
	storageInterface "github.com/Agent-Field/agentfield/control-plane/internal/storage"
)

// CreateServiceContainer creates and wires up all services for the CLI commands
func CreateServiceContainer(cfg *config.Config, agentfieldHome string) *framework.ServiceContainer {
	// Create infrastructure components
	fileSystem := storage.NewFileSystemAdapter()
	registryPath := filepath.Join(agentfieldHome, "installed.json")
	registryStorage := storage.NewLocalRegistryStorage(fileSystem, registryPath)
	processManager := process.NewProcessManager()
	portManager := process.NewPortManager()

	// Create storage provider based on configuration
	storageFactory := &storageInterface.StorageFactory{}
	storageProvider, _, err := storageFactory.CreateStorage(cfg.Storage)
	if err != nil {
		// Log error - database storage initialization failed
		// In production, this should be handled more gracefully
		storageProvider = nil
	}

	// Create services
	packageService := services.NewPackageService(registryStorage, fileSystem, agentfieldHome)
	agentService := services.NewAgentService(processManager, portManager, registryStorage, nil, agentfieldHome) // nil agentClient for now
	devService := services.NewDevService(processManager, portManager, fileSystem)

	// Create DID services if enabled
	var didService *didServices.DIDService
	var vcService *didServices.VCService
	var keystoreService *didServices.KeystoreService
	var didRegistry *didServices.DIDRegistry

	if cfg.Features.DID.Enabled {
		// Create keystore service
		keystoreService, err = didServices.NewKeystoreService(&cfg.Features.DID.Keystore)
		if err != nil {
			// Log error but continue - DID system will be disabled
			keystoreService = nil
		}

		// Create DID registry with database storage (required)
		if storageProvider != nil {
			didRegistry = didServices.NewDIDRegistryWithStorage(storageProvider)
			if passphrase := cfg.Features.DID.Keystore.EncryptionPassphrase; passphrase != "" {
				didRegistry.SetEncryptionService(encryption.NewEncryptionService(passphrase))
			}
		} else {
			// DID registry requires database storage, skip if not available
			didRegistry = nil
		}

		if didRegistry != nil {
			if err := didRegistry.Initialize(); err != nil {
				// Log error but continue
				didRegistry = nil
			}
		}

		// Create DID service
		if keystoreService != nil && didRegistry != nil {
			didService = didServices.NewDIDService(&cfg.Features.DID, keystoreService, didRegistry)

			// Pick the stored server identity this control plane owns. Moving
			// AGENTFIELD_HOME must not silently create a new identity.
			agentfieldServerID, err := didServices.SelectAgentFieldServerID(
				didRegistry, cfg.Features.DID.ServerID, generateAgentFieldServerID(agentfieldHome))
			if err == nil {
				err = didService.Initialize(agentfieldServerID)
			}
			if err != nil {
				logger.Logger.Warn().Err(err).Msg("failed to initialize DID service")
				didService = nil
			} else {
				// Create VC service with database storage (required)
				if storageProvider != nil {
					vcService = didServices.NewVCService(&cfg.Features.DID, didService, storageProvider)
				}

				if vcService != nil {
					if err := vcService.Initialize(); err != nil {
						logger.Logger.Warn().Err(err).Msg("failed to initialize VC service")
						vcService = nil
					}
				}
			}
		}
	}

	return &framework.ServiceContainer{
		PackageService:  packageService,
		AgentService:    agentService,
		DevService:      devService,
		DIDService:      didService,
		VCService:       vcService,
		KeystoreService: keystoreService,
		DIDRegistry:     didRegistry,
		StorageProvider: storageProvider,
	}
}

// CreateServiceContainerWithDefaults creates a service container with default configuration
func CreateServiceContainerWithDefaults(agentfieldHome string) *framework.ServiceContainer {
	// Use default config for now
	cfg := &config.Config{} // This will be enhanced when config is properly structured
	return CreateServiceContainer(cfg, agentfieldHome)
}

var absPathForAgentFieldServerID = filepath.Abs

// generateAgentFieldServerID returns the af server ID derived from the
// agentfield home directory. It names a new registry and finds one created at
// the same home path; SelectAgentFieldServerID decides which stored registry
// is used.
func generateAgentFieldServerID(agentfieldHome string) string {
	return didServices.LegacyAgentFieldServerID(agentfieldHome, absPathForAgentFieldServerID)
}
