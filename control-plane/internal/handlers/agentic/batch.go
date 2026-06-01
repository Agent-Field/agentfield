package agentic

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"

	"github.com/gin-gonic/gin"
)

const maxBatchSize = 20

// BatchOperation represents a single operation in a batch.
type BatchOperation struct {
	ID     string          `json:"id"`
	Method string          `json:"method" binding:"required"`
	Path   string          `json:"path" binding:"required"`
	Body   json.RawMessage `json:"body,omitempty"`
}

// BatchRequest is the top-level batch request.
type BatchRequest struct {
	Operations []BatchOperation `json:"operations" binding:"required"`
}

// BatchResult represents the result of a single batch operation.
type BatchResult struct {
	ID     string          `json:"id"`
	Status int             `json:"status"`
	Body   json.RawMessage `json:"body"`
}

// BatchHandler executes multiple API operations concurrently.
func BatchHandler(router *gin.Engine) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req BatchRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			respondError(c, http.StatusBadRequest, "invalid_request", err.Error())
			return
		}

		if len(req.Operations) == 0 {
			respondError(c, http.StatusBadRequest, "empty_batch", "at least one operation is required")
			return
		}

		if len(req.Operations) > maxBatchSize {
			respondError(c, http.StatusBadRequest, "batch_too_large",
				"maximum batch size is 20 operations")
			return
		}

		results := make([]BatchResult, len(req.Operations))
		var wg sync.WaitGroup

		for i, op := range req.Operations {
			wg.Add(1)
			go func(idx int, op BatchOperation) {
				defer wg.Done()

				var bodyReader io.Reader
				if op.Body != nil {
					bodyReader = bytes.NewReader(op.Body)
				}

				// Create a sub-request
				httpReq, err := http.NewRequestWithContext(c.Request.Context(), op.Method, op.Path, bodyReader)
				if err != nil {
					results[idx] = BatchResult{
						ID:     op.ID,
						Status: http.StatusBadRequest,
						Body:   json.RawMessage(`{"error":"invalid operation"}`),
					}
					return
				}

				// Copy auth headers
				httpReq.Header.Set("Content-Type", "application/json")
				if apiKey := c.GetHeader("X-API-Key"); apiKey != "" {
					httpReq.Header.Set("X-API-Key", apiKey)
				}
				if auth := c.GetHeader("Authorization"); auth != "" {
					httpReq.Header.Set("Authorization", auth)
				}

				// Execute against the router
				w := httptest.NewRecorder()
				router.ServeHTTP(w, httpReq)

				results[idx] = BatchResult{
					ID:     op.ID,
					Status: w.Code,
					Body:   json.RawMessage(w.Body.Bytes()),
				}
			}(i, op)
		}

		wg.Wait()

		respondOK(c, gin.H{
			"results": results,
			"total":   len(results),
		})
	}
}
