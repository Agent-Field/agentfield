package handlers

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
)

type StartSessionRequest struct {
	Provider  string                 `json:"provider"`
	Transport string                 `json:"transport"`
	Model     string                 `json:"model,omitempty"`
	Voice     string                 `json:"voice,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

type ToolSessionRequest struct {
	Target string                 `json:"target,omitempty"`
	Input  map[string]interface{} `json:"input,omitempty"`
}

func StartSessionHandler(store storage.StorageProvider) gin.HandlerFunc {
	return func(c *gin.Context) {
		nodeID, sessionName, ok := splitSessionTarget(c.Param("target"))
		if !ok {
			c.JSON(http.StatusBadRequest, gin.H{"error": "session target must be <node>.<session>"})
			return
		}

		definition, found := lookupSessionDefinition(c, store, nodeID, sessionName)
		if !found {
			return
		}

		var req StartSessionRequest
		if c.Request.Body != nil {
			_ = c.ShouldBindJSON(&req)
		}

		provider := firstNonEmptySession(req.Provider, definition.Provider)
		transport := firstNonEmptySession(req.Transport, definition.Transport)
		capability, err := types.ValidateSessionTransport(provider, transport)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"error":     err.Error(),
				"provider":  provider,
				"transport": transport,
			})
			return
		}
		if capability.Provider != definition.Provider || capability.Transport != definition.Transport {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": fmt.Sprintf(
					"session %s.%s is registered for provider=%s transport=%s; requested provider=%s transport=%s",
					nodeID,
					sessionName,
					definition.Provider,
					definition.Transport,
					capability.Provider,
					capability.Transport,
				),
			})
			return
		}

		sessionID := "sess_" + time.Now().UTC().Format("20060102_150405") + "_" + shortRandom()
		model := firstNonEmptySession(req.Model, definition.Model)
		voice := firstNonEmptySession(req.Voice, definition.Voice)
		c.JSON(http.StatusCreated, gin.H{
			"session_id":   sessionID,
			"target":       nodeID + "." + sessionName,
			"provider":     capability.Provider,
			"transport":    capability.Transport,
			"model":        model,
			"voice":        voice,
			"modalities":   definition.Modalities,
			"tags":         definition.ApprovedTags,
			"tool_targets": sessionToolTargets(nodeID, definition.Tools),
			"offer_url":    fmt.Sprintf("/api/v1/session-instances/%s/realtime-offer", url.PathEscape(sessionID)),
			"tool_url":     fmt.Sprintf("/api/v1/session-instances/%s/tools/{tool}", url.PathEscape(sessionID)),
			"created_at":   time.Now().UTC().Format(time.RFC3339Nano),
		})
	}
}

func SessionRealtimeOfferHandler(store storage.StorageProvider) gin.HandlerFunc {
	return func(c *gin.Context) {
		_ = store
		provider := strings.TrimSpace(c.Query("provider"))
		transport := strings.TrimSpace(c.Query("transport"))
		if provider == "" || transport == "" {
			var req StartSessionRequest
			_ = c.ShouldBindJSON(&req)
			provider = req.Provider
			transport = req.Transport
		}
		if _, err := types.ValidateSessionTransport(provider, transport); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		if types.NormalizeSessionTransportValue(transport) != "webrtc" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "realtime-offer requires transport=webrtc"})
			return
		}
		if types.NormalizeSessionTransportValue(provider) != "openai" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "webrtc realtime offers currently require provider=openai"})
			return
		}
		if strings.TrimSpace(os.Getenv("OPENAI_API_KEY")) == "" {
			c.JSON(http.StatusBadGateway, gin.H{"error": "OPENAI_API_KEY is required for provider=openai transport=webrtc"})
			return
		}
		sdp, err := io.ReadAll(c.Request.Body)
		if err != nil || strings.TrimSpace(string(sdp)) == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "SDP offer body is required"})
			return
		}
		answer, err := createOpenAIRealtimeCall(
			c.Request.Context(),
			sessionPathID(c),
			string(sdp),
			firstNonEmptySession(c.Query("model"), "gpt-realtime-2"),
			firstNonEmptySession(c.Query("voice"), "marin"),
		)
		if err != nil {
			c.JSON(http.StatusBadGateway, gin.H{
				"error":    err.Error(),
				"boundary": "control-plane -> realtime provider",
				"provider": "openai",
			})
			return
		}
		c.Data(http.StatusOK, "application/sdp", []byte(answer))
	}
}

func SessionToolHandler(store storage.StorageProvider, timeout time.Duration, internalToken string) gin.HandlerFunc {
	return func(c *gin.Context) {
		_ = store
		sessionID := sessionPathID(c)
		toolName := strings.TrimSpace(c.Param("tool"))
		var req ToolSessionRequest
		if err := c.ShouldBindJSON(&req); err != nil && err != io.EOF {
			c.JSON(http.StatusBadRequest, gin.H{"error": "invalid JSON body"})
			return
		}
		target := firstNonEmptySession(req.Target, toolName)
		if !strings.Contains(target, ".") {
			c.JSON(http.StatusBadRequest, gin.H{"error": "session tool target must be <node>.<reasoner>"})
			return
		}

		body, _ := json.Marshal(gin.H{"input": req.Input})
		scheme := "http"
		if c.Request.TLS != nil {
			scheme = "https"
		}
		executeURL := fmt.Sprintf("%s://%s/api/v1/execute/async/%s", scheme, c.Request.Host, url.PathEscape(target))
		forwardReq, err := http.NewRequestWithContext(c.Request.Context(), http.MethodPost, executeURL, strings.NewReader(string(body)))
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		forwardReq.Header.Set("Content-Type", "application/json")
		forwardReq.Header.Set("X-Session-ID", sessionID)
		copyForwardedSessionAuthHeaders(c, forwardReq)
		if internalToken != "" {
			forwardReq.Header.Set("Authorization", "Bearer "+internalToken)
		}

		client := &http.Client{Timeout: timeout}
		resp, err := client.Do(forwardReq)
		if err != nil {
			c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
			return
		}
		defer resp.Body.Close()
		respBody, _ := io.ReadAll(resp.Body)
		c.Data(resp.StatusCode, resp.Header.Get("Content-Type"), respBody)
	}
}

func sessionPathID(c *gin.Context) string {
	return firstNonEmptySession(c.Param("session_id"), c.Param("target"))
}

func lookupSessionDefinition(c *gin.Context, store storage.StorageProvider, nodeID string, sessionName string) (types.SessionDefinition, bool) {
	agent, err := store.GetAgent(c.Request.Context(), nodeID)
	if err != nil || agent == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "agent not found"})
		return types.SessionDefinition{}, false
	}
	types.HydrateAgentSessions(agent)
	if len(agent.Sessions) == 0 {
		c.JSON(http.StatusNotFound, gin.H{"error": "agent has no registered sessions"})
		return types.SessionDefinition{}, false
	}
	for _, definition := range agent.Sessions {
		if definition.Name == sessionName {
			return definition, true
		}
	}
	c.JSON(http.StatusNotFound, gin.H{"error": "session not registered"})
	return types.SessionDefinition{}, false
}

func copyForwardedSessionAuthHeaders(c *gin.Context, req *http.Request) {
	for _, header := range []string{
		"X-Caller-Agent-ID",
		"X-Caller-DID",
		"X-Actor-ID",
		"X-API-Key",
		"X-Run-ID",
		"X-Parent-Execution-ID",
		"X-Parent-VC-ID",
	} {
		if value := strings.TrimSpace(c.GetHeader(header)); value != "" {
			req.Header.Set(header, value)
		}
	}
}

func splitSessionTarget(target string) (string, string, bool) {
	target = strings.TrimSpace(target)
	parts := strings.SplitN(target, ".", 2)
	if len(parts) != 2 || strings.TrimSpace(parts[0]) == "" || strings.TrimSpace(parts[1]) == "" {
		return "", "", false
	}
	return strings.TrimSpace(parts[0]), strings.TrimSpace(parts[1]), true
}

func firstNonEmptySession(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func sessionToolTargets(nodeID string, tools []string) map[string]string {
	targets := map[string]string{}
	for _, tool := range tools {
		tool = strings.TrimSpace(tool)
		if tool == "" {
			continue
		}
		if strings.Contains(tool, ".") {
			parts := strings.SplitN(tool, ".", 2)
			targets[parts[1]] = tool
		} else {
			targets[tool] = nodeID + "." + tool
		}
	}
	return targets
}

func createOpenAIRealtimeCall(ctx context.Context, sessionID string, sdp string, model string, voice string) (string, error) {
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	if err := writer.WriteField("sdp", sdp); err != nil {
		return "", err
	}
	sessionConfig := map[string]interface{}{
		"type":         "realtime",
		"model":        model,
		"instructions": "You are a realtime voice front end for an AgentField session. Use registered tools to route agent work through the AgentField control plane.",
		"audio":        map[string]interface{}{"output": map[string]interface{}{"voice": voice}},
		"tool_choice":  "auto",
	}
	sessionBytes, _ := json.Marshal(sessionConfig)
	if err := writer.WriteField("session", string(sessionBytes)); err != nil {
		return "", err
	}
	if err := writer.Close(); err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.openai.com/v1/realtime/calls", &body)
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+strings.TrimSpace(os.Getenv("OPENAI_API_KEY")))
	req.Header.Set("Content-Type", writer.FormDataContentType())
	hash := sha256.Sum256([]byte(sessionID))
	req.Header.Set("OpenAI-Safety-Identifier", hex.EncodeToString(hash[:])[:32])

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	answer, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= http.StatusBadRequest {
		return "", fmt.Errorf("OpenAI realtime call failed (%d): %s", resp.StatusCode, strings.TrimSpace(string(answer)))
	}
	return string(answer), nil
}

func shortRandom() string {
	return fmt.Sprintf("%08x", time.Now().UnixNano()&0xffffffff)
}
