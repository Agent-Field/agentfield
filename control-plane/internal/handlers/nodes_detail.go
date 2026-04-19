package handlers

import (
	"net/http"

	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
)

// GetNodeHandler handles getting a specific node by ID
func GetNodeHandler(storageProvider storage.StorageProvider) gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx := c.Request.Context()
		nodeID := c.Param("node_id")
		if nodeID == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "node_id is required"})
			return
		}

		var node *types.AgentNode
		var err error
		if version := c.Query("version"); version != "" {
			node, err = storageProvider.GetAgentVersion(ctx, nodeID, version)
		} else {
			node, err = storageProvider.GetAgent(ctx, nodeID)
		}
		if err != nil || node == nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "node not found"})
			return
		}

		c.JSON(http.StatusOK, node)
	}
}
