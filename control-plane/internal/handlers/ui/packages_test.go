//go:build integration
// +build integration

package ui

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/your-org/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

// setupRouterAndServices is a helper function to set up the router and mock services for package tests
func setupRouterAndServices() (*gin.Engine, *MockStorageProvider, *MockAgentServiceForLifecycle) {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	mockStorage := &MockStorageProvider{}
	mockAgentService := &MockAgentServiceForLifecycle{} // Assuming this is the correct mock service

	packageHandler := NewPackageHandler(mockStorage)
	// lifecycleHandler := NewLifecycleHandler(mockStorage, mockAgentService) // If needed for other routes

	v1 := router.Group("/api/ui/v1")
	{
		packages := v1.Group("/agents/packages")
		{
			packages.GET("", packageHandler.ListPackagesHandler)
			packages.GET("/:packageId/details", packageHandler.GetPackageDetailsHandler)
		}
		// Example for lifecycle routes if needed in the same test file or for consistency
		// agents := v1.Group("/agents")
		// {
		// 	agents.POST("/:agentId/start", lifecycleHandler.StartAgentHandler)
		// }
	}
	return router, mockStorage, mockAgentService
}

func TestListPackagesHandler(t *testing.T) {
	// Test data
	description := "Test package description"
	author := "Test Author"
	packages := []*types.AgentPackage{
		{
			ID:                  "test-package-1",
			Name:                "Test Package 1",
			Version:             "1.0.0",
			Description:         &description,
			Author:              &author,
			InstallPath:         "/path/to/package1",
			ConfigurationSchema: json.RawMessage(`{"required": {"api_key": {"type": "secret"}}}`),
		},
		{
			ID:                  "test-package-2",
			Name:                "Test Package 2",
			Version:             "2.0.0",
			Description:         nil,
			Author:              nil,
			InstallPath:         "/path/to/package2",
			ConfigurationSchema: json.RawMessage(`{}`),
		},
	}

	t.Run("successful list packages", func(t *testing.T) {
		router, mockStorage, _ := setupRouterAndServices()
		// Setup mocks
		mockStorage.On("QueryAgentPackages", mock.AnythingOfType("context.Context"), mock.AnythingOfType("types.PackageFilters")).Return(packages, nil).Once()
		mockStorage.On("GetAgentConfiguration", mock.AnythingOfType("context.Context"), "test-package-1", "test-package-1").Return(nil, assert.AnError).Once()
		mockStorage.On("GetAgentConfiguration", mock.AnythingOfType("context.Context"), "test-package-2", "test-package-2").Return(nil, assert.AnError).Once()

		// Make request
		req, _ := http.NewRequest("GET", "/api/ui/v1/agents/packages", nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		// Assertions
		assert.Equal(t, http.StatusOK, w.Code)

		var response PackageListResponse
		err := json.Unmarshal(w.Body.Bytes(), &response)
		assert.NoError(t, err)
		assert.Equal(t, 2, response.Total)
		assert.Len(t, response.Packages, 2)

		// Check first package
		pkg1 := response.Packages[0]
		assert.Equal(t, "test-package-1", pkg1.ID)
		assert.Equal(t, "Test Package 1", pkg1.Name)
		assert.Equal(t, "1.0.0", pkg1.Version)
		assert.Equal(t, "Test package description", pkg1.Description)
		assert.Equal(t, "Test Author", pkg1.Author)
		assert.True(t, pkg1.ConfigurationRequired)
		assert.False(t, pkg1.ConfigurationComplete)

		// Check second package
		pkg2 := response.Packages[1]
		assert.Equal(t, "test-package-2", pkg2.ID)
		assert.Equal(t, "", pkg2.Description) // nil pointer should become empty string
		assert.Equal(t, "", pkg2.Author)      // nil pointer should become empty string
		assert.False(t, pkg2.ConfigurationRequired)
		assert.True(t, pkg2.ConfigurationComplete)

		mockStorage.AssertExpectations(t)
	})

	t.Run("storage error", func(t *testing.T) {
		router, mockStorage, _ := setupRouterAndServices()
		// Setup mocks
		mockStorage.On("QueryAgentPackages", mock.AnythingOfType("context.Context"), mock.AnythingOfType("types.PackageFilters")).Return([]*types.AgentPackage{}, assert.AnError).Once()

		// Make request
		req, _ := http.NewRequest("GET", "/api/ui/v1/agents/packages", nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		// Assertions
		assert.Equal(t, http.StatusInternalServerError, w.Code)

		var response ErrorResponse
		err := json.Unmarshal(w.Body.Bytes(), &response)
		assert.NoError(t, err)
		assert.Equal(t, "failed to list packages", response.Error)

		mockStorage.AssertExpectations(t)
	})
}

func TestGetPackageDetailsHandler(t *testing.T) {
	// Test data
	description := "Test package description"
	author := "Test Author"
	configSchema := json.RawMessage(`{
		"required": {
			"api_key": {
				"type": "secret",
				"description": "API key for authentication"
			}
		},
		"optional": {
			"timeout": {
				"type": "number",
				"description": "Request timeout in seconds",
				"default": 30
			}
		}
	}`)

	pkg := &types.AgentPackage{
		ID:                  "test-package",
		Name:                "Test Package",
		Version:             "1.0.0",
		Description:         &description,
		Author:              &author,
		InstallPath:         "/path/to/package",
		ConfigurationSchema: configSchema,
	}

	t.Run("successful get package details", func(t *testing.T) {
		router, mockStorage, _ := setupRouterAndServices()
		// Setup mocks
		mockStorage.On("GetAgentPackage", mock.AnythingOfType("context.Context"), "test-package").Return(pkg, nil).Once()
		mockStorage.On("GetAgentConfiguration", mock.AnythingOfType("context.Context"), "test-package", "test-package").Return(nil, assert.AnError).Once()

		// Make request
		req, _ := http.NewRequest("GET", "/api/ui/v1/agents/packages/test-package/details", nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		// Assertions
		assert.Equal(t, http.StatusOK, w.Code)

		var response PackageDetailsResponse
		err := json.Unmarshal(w.Body.Bytes(), &response)
		assert.NoError(t, err)

		assert.Equal(t, "test-package", response.ID)
		assert.Equal(t, "Test Package", response.Name)
		assert.Equal(t, "1.0.0", response.Version)
		assert.Equal(t, "Test package description", response.Description)
		assert.Equal(t, "Test Author", response.Author)
		assert.Equal(t, "/path/to/package", response.InstallPath)
		assert.Equal(t, "not_configured", response.Status)

		// Check configuration
		assert.True(t, response.Configuration.Required)
		assert.False(t, response.Configuration.Complete)
		assert.NotNil(t, response.Configuration.Schema)
		assert.NotNil(t, response.Configuration.Current)

		mockStorage.AssertExpectations(t)
	})

	t.Run("package not found", func(t *testing.T) {
		router, mockStorage, _ := setupRouterAndServices()
		// Setup mocks
		mockStorage.On("GetAgentPackage", mock.AnythingOfType("context.Context"), "nonexistent").Return((*types.AgentPackage)(nil), assert.AnError).Once()

		// Make request
		req, _ := http.NewRequest("GET", "/api/ui/v1/agents/packages/nonexistent/details", nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		// Assertions
		assert.Equal(t, http.StatusNotFound, w.Code)

		var response ErrorResponse
		err := json.Unmarshal(w.Body.Bytes(), &response)
		assert.NoError(t, err)
		assert.Equal(t, "package not found", response.Error)

		mockStorage.AssertExpectations(t)
	})
}

func TestSafeStringValue(t *testing.T) {
	handler := NewPackageHandler(nil) // No storage needed for this helper

	t.Run("non-nil string pointer", func(t *testing.T) {
		str := "test string"
		result := handler.safeStringValue(&str)
		assert.Equal(t, "test string", result)
	})

	t.Run("nil string pointer", func(t *testing.T) {
		result := handler.safeStringValue(nil)
		assert.Equal(t, "", result)
	})
}

func TestMatchesSearch(t *testing.T) {
	handler := &PackageHandler{}

	description := "A test package for testing"
	author := "Test Author"
	pkg := &types.AgentPackage{
		ID:          "test-package-id",
		Name:        "Test Package Name",
		Description: &description,
		Author:      &author,
	}

	t.Run("matches name", func(t *testing.T) {
		result := handler.matchesSearch(pkg, "Test Package")
		assert.True(t, result)
	})

	t.Run("matches description", func(t *testing.T) {
		result := handler.matchesSearch(pkg, "testing")
		assert.True(t, result)
	})

	t.Run("matches author", func(t *testing.T) {
		result := handler.matchesSearch(pkg, "Test Author")
		assert.True(t, result)
	})

	t.Run("matches ID", func(t *testing.T) {
		result := handler.matchesSearch(pkg, "test-package-id")
		assert.True(t, result)
	})

	t.Run("case insensitive", func(t *testing.T) {
		result := handler.matchesSearch(pkg, "TEST PACKAGE")
		assert.True(t, result)
	})

	t.Run("no match", func(t *testing.T) {
		result := handler.matchesSearch(pkg, "nonexistent")
		assert.False(t, result)
	})

	t.Run("nil description and author", func(t *testing.T) {
		pkgWithNils := &types.AgentPackage{
			ID:          "test-id",
			Name:        "Test Name",
			Description: nil,
			Author:      nil,
		}

		result := handler.matchesSearch(pkgWithNils, "Test Name")
		assert.True(t, result)

		result = handler.matchesSearch(pkgWithNils, "nonexistent")
		assert.False(t, result)
	})
}
