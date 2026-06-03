package middleware

import (
	"crypto/subtle"
	"net/http"

	"github.com/gin-gonic/gin"
)

// ConnectorTokenAuth enforces the connector token for connector routes.
// It validates the X-Connector-Token header and injects audit correlation
// metadata (X-Command-ID, X-Command-Source) into the gin context.
func ConnectorTokenAuth(connectorToken string) gin.HandlerFunc {
	return func(c *gin.Context) {
		if connectorToken == "" {
			c.AbortWithStatusJSON(http.StatusForbidden, gin.H{
				"error":   "forbidden",
				"message": "connector is not configured (no token set)",
			})
			return
		}

		token := c.GetHeader("X-Connector-Token")

		if subtle.ConstantTimeCompare([]byte(token), []byte(connectorToken)) != 1 {
			c.AbortWithStatusJSON(http.StatusForbidden, gin.H{
				"error":   "forbidden",
				"message": "connector token required for this operation (use X-Connector-Token header)",
			})
			return
		}

		// Inject audit correlation metadata from the connector
		if cmdID := c.GetHeader("X-Command-ID"); cmdID != "" {
			c.Set("connector_command_id", cmdID)
		}
		if cmdSource := c.GetHeader("X-Command-Source"); cmdSource != "" {
			c.Set("connector_command_source", cmdSource)
		}

		c.Next()
	}
}
