//go:build !embedded

package client

import "github.com/gin-gonic/gin"

// RegisterUIRoutes is a no-op when the UI is not embedded.
// The server will serve UI assets from the filesystem when configured.
func RegisterUIRoutes(_ *gin.Engine) {}

// IsUIEmbedded reports whether UI assets are embedded in the binary.
func IsUIEmbedded() bool { return false }

