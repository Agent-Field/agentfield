package middleware

import (
	"net/http"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/gin-gonic/gin"
)

// ConnectorCapabilityCheck enforces that a specific capability is enabled for the
// connector token and respects read_only mode by rejecting write HTTP methods.
// This is the CP-side security boundary — even if the connector is compromised,
// requests for disabled or read-only capabilities are rejected here.
func ConnectorCapabilityCheck(capName string, capabilities map[string]config.ConnectorCapability) gin.HandlerFunc {
	return func(c *gin.Context) {
		cap, exists := capabilities[capName]
		if !exists || !cap.Enabled {
			c.AbortWithStatusJSON(http.StatusForbidden, gin.H{
				"error":   "capability_disabled",
				"message": "capability " + capName + " is not enabled for this connector",
			})
			return
		}

		if cap.ReadOnly {
			switch c.Request.Method {
			case http.MethodPost, http.MethodPut, http.MethodDelete, http.MethodPatch:
				c.AbortWithStatusJSON(http.StatusForbidden, gin.H{
					"error":   "read_only",
					"message": "capability " + capName + " is read-only; write operations are not permitted",
				})
				return
			}
		}

		c.Next()
	}
}
