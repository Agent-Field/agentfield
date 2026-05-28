//go:build uiembed
// +build uiembed

package embedded

import (
	"embed"
	"io/fs"
	"net/http"
	"path/filepath"
	"strings"

	"github.com/gin-gonic/gin"
)

// UIFiles embeds the entire web/client/dist directory into the binary
//
//go:embed web/client/dist
var UIFiles embed.FS

// EmbeddedUIHandler creates a Gin handler that serves embedded UI files
func EmbeddedUIHandler() gin.HandlerFunc {
	// Create a sub-filesystem that strips the "web/client/dist" prefix
	uiFS, err := fs.Sub(UIFiles, "web/client/dist")
	if err != nil {
		panic("Failed to create UI filesystem: " + err.Error())
	}

	fileServer := http.FileServer(http.FS(uiFS))

	return gin.WrapH(http.StripPrefix("/ui", fileServer))
}

// EmbeddedSPAHandler serves the index.html for SPA routing
func EmbeddedSPAHandler() gin.HandlerFunc {
	return func(c *gin.Context) {
		// Read index.html from embedded files
		indexHTML, err := UIFiles.ReadFile("web/client/dist/index.html")
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "Failed to load UI",
			})
			return
		}

		c.Header("Content-Type", "text/html; charset=utf-8")
		c.String(http.StatusOK, string(indexHTML))
	}
}

// IsUIEmbedded checks if UI files are embedded in the binary
func IsUIEmbedded() bool {
	// Try to read a file that should exist in the embedded UI
	_, err := UIFiles.ReadFile("web/client/dist/index.html")
	return err == nil
}

// GetEmbeddedFile reads a specific file from the embedded UI
func GetEmbeddedFile(path string) ([]byte, error) {
	// Ensure path starts with the correct prefix
	if !strings.HasPrefix(path, "web/client/dist/") {
		path = filepath.Join("web/client/dist", path)
	}
	return UIFiles.ReadFile(path)
}

// ListEmbeddedFiles returns a list of all embedded UI files
func ListEmbeddedFiles() ([]string, error) {
	var files []string
	err := fs.WalkDir(UIFiles, "web/client/dist", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() {
			// Strip the prefix to get relative path
			relativePath := strings.TrimPrefix(path, "web/client/dist/")
			files = append(files, relativePath)
		}
		return nil
	})
	return files, err
}
