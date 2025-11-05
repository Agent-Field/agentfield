// UI embedding and route registration for Haxen

package client

import (
	"embed"
	"fmt"
	"io/fs"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
)

//go:embed dist/* dist/**
var UIFiles embed.FS

// RegisterUIRoutes registers the UI routes with the Gin engine.
func RegisterUIRoutes(router *gin.Engine) {
	fmt.Println("Registering embedded UI routes...")

	// Create a sub-filesystem that strips the "dist" prefix
	uiFS, err := fs.Sub(UIFiles, "dist")
	if err != nil {
		panic("Failed to create UI filesystem: " + err.Error())
	}

	fileServer := http.FileServer(http.FS(uiFS))

	router.GET("/ui/*filepath", func(c *gin.Context) {
		path := c.Param("filepath")
		
		// If accessing root UI path or a directory, serve index.html
		if path == "/" || path == "" || strings.HasSuffix(path, "/") {
			indexHTML, err := UIFiles.ReadFile("dist/index.html")
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{
					"error": "Failed to load UI index",
				})
				return
			}
			c.Header("Content-Type", "text/html; charset=utf-8")
			c.String(http.StatusOK, string(indexHTML))
			return
		}
		
		// Check if it's a static asset by looking for common web asset file extensions
		// This prevents reasoner IDs with dots (like "deepresearchagent.meta_research_methodology_reasoner")
		// from being treated as static assets
		pathLower := strings.ToLower(path)
		isStaticAsset := strings.HasSuffix(pathLower, ".js") ||
			strings.HasSuffix(pathLower, ".css") ||
			strings.HasSuffix(pathLower, ".html") ||
			strings.HasSuffix(pathLower, ".ico") ||
			strings.HasSuffix(pathLower, ".png") ||
			strings.HasSuffix(pathLower, ".jpg") ||
			strings.HasSuffix(pathLower, ".jpeg") ||
			strings.HasSuffix(pathLower, ".gif") ||
			strings.HasSuffix(pathLower, ".svg") ||
			strings.HasSuffix(pathLower, ".woff") ||
			strings.HasSuffix(pathLower, ".woff2") ||
			strings.HasSuffix(pathLower, ".ttf") ||
			strings.HasSuffix(pathLower, ".eot") ||
			strings.HasSuffix(pathLower, ".map") ||
			strings.HasSuffix(pathLower, ".json") ||
			strings.HasSuffix(pathLower, ".xml") ||
			strings.HasSuffix(pathLower, ".txt")

		if isStaticAsset {
			// Try to serve the static file
			http.StripPrefix("/ui", fileServer).ServeHTTP(c.Writer, c.Request)
			return
		}
		
		// For all other paths (SPA routes), serve index.html
		indexHTML, err := UIFiles.ReadFile("dist/index.html")
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "Failed to load UI index",
			})
			return
		}
		c.Header("Content-Type", "text/html; charset=utf-8")
		c.String(http.StatusOK, string(indexHTML))
	})

	// Root redirect to embedded UI
	router.GET("/", func(c *gin.Context) {
		c.Redirect(http.StatusMovedPermanently, "/ui/")
	})

	// SPA fallback - serve index.html for all /ui/* routes that don't match static files
	router.NoRoute(func(c *gin.Context) {
		if strings.HasPrefix(c.Request.URL.Path, "/ui/") {
			indexHTML, err := UIFiles.ReadFile("dist/index.html")
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{
					"error": "Failed to load UI",
				})
				return
			}
			c.Header("Content-Type", "text/html; charset=utf-8")
			c.String(http.StatusOK, string(indexHTML))
		} else {
			// For non-UI paths, return 404
			c.JSON(http.StatusNotFound, gin.H{"error": "endpoint not found"})
		}
	})
}

// IsUIEmbedded checks if UI files are embedded in the binary.
func IsUIEmbedded() bool {
	// Try to read a file that should exist in the embedded UI
	_, err := UIFiles.ReadFile("dist/index.html")
	return err == nil
}
