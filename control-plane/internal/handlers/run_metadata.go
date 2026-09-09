package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
)

// maxRunMetadataRequestBytes is large enough for every documented field at
// its maximum size, including JSON escaping, while preventing an unbounded
// chunked body from being decoded into RawMessages and slices. Execute has its
// own larger payload limit; this endpoint only accepts short identity fields.
const maxRunMetadataRequestBytes int64 = 128 << 10

// RunMetadataInput preserves absent, null and value as distinct patch states.
type RunMetadataInput struct {
	DisplayName json.RawMessage `json:"display_name,omitempty"`
	Labels      json.RawMessage `json:"labels,omitempty"`
	Links       json.RawMessage `json:"links,omitempty"`
}

type workflowRunMetadataWriter interface {
	UpdateWorkflowRunMetadata(context.Context, string, func(map[string]json.RawMessage) error) error
}

// SetRunMetadataHandler handles POST /api/v1/runs/:run_id/metadata.
func SetRunMetadataHandler(store ExecutionStore) gin.HandlerFunc {
	return func(c *gin.Context) {
		runID := strings.TrimSpace(c.Param("run_id"))
		if runID == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "run_id is required"})
			return
		}
		c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxRunMetadataRequestBytes)
		var input RunMetadataInput
		if err := c.ShouldBindJSON(&input); err != nil {
			var maxBytesErr *http.MaxBytesError
			if errors.As(err, &maxBytesErr) {
				c.JSON(http.StatusRequestEntityTooLarge, gin.H{"error": "request body too large"})
				return
			}
			c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request body: " + err.Error()})
			return
		}
		if _, err := applyRunMetadataInput(types.RunMetadata{}, input); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		executions, err := store.QueryExecutionRecords(c.Request.Context(), types.ExecutionFilter{RunID: &runID, Limit: 1})
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		if len(executions) == 0 {
			c.JSON(http.StatusNotFound, gin.H{"error": "workflow run not found"})
			return
		}
		writer, ok := store.(workflowRunMetadataWriter)
		if !ok {
			c.JSON(http.StatusNotImplemented, gin.H{"error": "run metadata storage is not supported"})
			return
		}
		actor, err := normalizeRunMetadataActor(c.GetHeader("X-Actor-ID"))
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		var merged types.RunMetadata
		err = writer.UpdateWorkflowRunMetadata(c.Request.Context(), runID, func(namespaces map[string]json.RawMessage) error {
			current := types.RunMetadata{}
			if raw, exists := namespaces[types.RunMetadataNamespace]; exists {
				_ = json.Unmarshal(raw, &current)
			}
			var err error
			merged, err = applyRunMetadataInput(current, input)
			if err != nil {
				return err
			}
			merged.SetBy = actor
			merged.UpdatedAt = time.Now().UTC().Format(time.RFC3339)
			if merged.DisplayName == "" && len(merged.Labels) == 0 && len(merged.Links) == 0 {
				delete(namespaces, types.RunMetadataNamespace)
				return nil
			}
			namespaces[types.RunMetadataNamespace], err = json.Marshal(merged)
			return err
		})
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, merged)
	}
}

func normalizeRunMetadataActor(raw string) (string, error) {
	actor := strings.TrimSpace(raw)
	if actor == "" {
		return "api", nil
	}
	if !utf8.ValidString(actor) {
		return "", fmt.Errorf("X-Actor-ID must be valid UTF-8")
	}
	if utf8.RuneCountInString(actor) > types.MaxRunMetadataSetByRunes {
		return "", fmt.Errorf("X-Actor-ID exceeds %d runes", types.MaxRunMetadataSetByRunes)
	}
	return actor, nil
}

// applyRunMetadataInput merges a public API patch. Unlike the UI-private golden
// route, which truncates its fixed button input, this public endpoint rejects
// values outside the documented bounds.
func applyRunMetadataInput(current types.RunMetadata, input RunMetadataInput) (types.RunMetadata, error) {
	if input.DisplayName != nil {
		if string(input.DisplayName) == "null" {
			current.DisplayName = ""
		} else {
			var value string
			if err := json.Unmarshal(input.DisplayName, &value); err != nil {
				return current, fmt.Errorf("display_name must be a string")
			}
			value = strings.TrimSpace(value)
			if utf8.RuneCountInString(value) > types.MaxRunDisplayNameRunes {
				return current, fmt.Errorf("display_name exceeds %d runes", types.MaxRunDisplayNameRunes)
			}
			current.DisplayName = value
		}
	}
	if input.Labels != nil {
		if string(input.Labels) == "null" {
			current.Labels = nil
		} else {
			var values []string
			if err := json.Unmarshal(input.Labels, &values); err != nil {
				return current, fmt.Errorf("labels must be an array of strings")
			}
			if len(values) > types.MaxRunLabels {
				return current, fmt.Errorf("labels exceeds %d items", types.MaxRunLabels)
			}
			seen := make(map[string]struct{}, len(values))
			current.Labels = nil
			for _, value := range values {
				value = strings.TrimSpace(value)
				if utf8.RuneCountInString(value) > types.MaxRunLabelRunes {
					return current, fmt.Errorf("label exceeds %d runes", types.MaxRunLabelRunes)
				}
				if value == "" {
					continue
				}
				if _, exists := seen[value]; exists {
					continue
				}
				seen[value] = struct{}{}
				current.Labels = append(current.Labels, value)
			}
		}
	}
	if input.Links != nil {
		if string(input.Links) == "null" {
			current.Links = nil
		} else {
			var links []types.RunMetadataLink
			if err := json.Unmarshal(input.Links, &links); err != nil {
				return current, fmt.Errorf("links must be an array of links")
			}
			if len(links) > types.MaxRunLinks {
				return current, fmt.Errorf("links exceeds %d items", types.MaxRunLinks)
			}
			for i := range links {
				links[i].Label = strings.TrimSpace(links[i].Label)
				if utf8.RuneCountInString(links[i].Label) > types.MaxRunLinkLabelRunes {
					return current, fmt.Errorf("link label exceeds %d runes", types.MaxRunLinkLabelRunes)
				}
				if err := types.ValidateRunLinkURL(links[i].URL); err != nil {
					return current, fmt.Errorf("invalid link url: %w", err)
				}
			}
			current.Links = links
		}
	}
	return current, nil
}
