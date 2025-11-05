package application

import (
	"crypto/sha256"
	"encoding/hex"
	"github.com/your-org/agentfield/control-plane/internal/cli/framework"
	"github.com/your-org/agentfield/control-plane/internal/config"
	"github.com/your-org/agentfield/control-plane/internal/core/services"
	"github.com/your-org/agentfield/control-plane/internal/infrastructure/process"
	"github.com/your-org/agentfield/control-plane/internal/infrastructure/storage"
	didServices "github.com/your-org/agentfield/control-plane/internal/services"
	storageInterface "github.com/your-org/agentfield/control-plane/internal/storage"
	"path/filepath"
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

			// Generate af server ID based on agentfield home directory
			// This ensures each agentfield instance has a unique ID while being deterministic
			agentfieldServerID := generateAgentFieldServerID(agentfieldHome)
			didService.Initialize(agentfieldServerID)

			// Create VC service with database storage (required)
			if storageProvider != nil {
				vcService = didServices.NewVCService(&cfg.Features.DID, didService, storageProvider)
			}

			if vcService != nil {
				vcService.Initialize()
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

// generateAgentFieldServerID creates a deterministic af server ID based on the agentfield home directory.
// This ensures each agentfield instance has a unique ID while being deterministic for the same installation.
func generateAgentFieldServerID(agentfieldHome string) string {
	// Use the absolute path of agentfield home to generate a deterministic ID
	absPath, err := filepath.Abs(agentfieldHome)
	if err != nil {
		// Fallback to the original path if absolute path fails
		absPath = agentfieldHome
	}

	// Create a hash of the agentfield home path to generate a unique but deterministic ID
	hash := sha256.Sum256([]byte(absPath))

	// Use first 16 characters of the hex hash as the af server ID
	// This provides uniqueness while keeping the ID manageable
	agentfieldServerID := hex.EncodeToString(hash[:])[:16]

	return agentfieldServerID
}
