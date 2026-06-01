package ui

import (
	"bufio"
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/gin-gonic/gin"
)

const agentProcessLogsPath = "/agentfield/v1/logs"

// NodeLogsProxyHandler proxies UI requests to an agent node's NDJSON log API.
type NodeLogsProxyHandler struct {
	Storage storage.StorageProvider
	// Snapshot returns effective proxy limits and the internal bearer token for upstream calls.
	Snapshot func() (proxy config.NodeLogProxyConfig, internalToken string)
}

// ProxyNodeLogsHandler is GET /api/ui/v1/nodes/:nodeId/logs
func (h *NodeLogsProxyHandler) ProxyNodeLogsHandler(c *gin.Context) {
	if h.Snapshot == nil || h.Storage == nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "proxy_not_configured"})
		return
	}
	nodeID := c.Param("nodeId")
	if nodeID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "nodeId is required"})
		return
	}
	ctx := c.Request.Context()
	agent, err := h.Storage.GetAgent(ctx, nodeID)
	if err != nil || agent == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "node not found"})
		return
	}
	base := strings.TrimSpace(agent.BaseURL)
	if base == "" {
		c.JSON(http.StatusBadGateway, gin.H{"error": "agent_unreachable", "message": "node has no base_url"})
		return
	}
	u, err := url.Parse(base)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid_base_url"})
		return
	}

	proxyCfg, token := h.Snapshot()
	q := c.Request.URL.Query()
	if tl := q.Get("tail_lines"); tl != "" {
		n, err := strconv.Atoi(tl)
		if err != nil || n < 0 {
			c.JSON(http.StatusBadRequest, gin.H{"error": "invalid_tail_lines"})
			return
		}
		if n > proxyCfg.MaxTailLines {
			c.JSON(http.StatusBadRequest, gin.H{
				"error":       "tail_too_large",
				"message":     "tail_lines exceeds server maximum",
				"max_allowed": proxyCfg.MaxTailLines,
			})
			return
		}
	}

	upstream := strings.TrimSuffix(base, "/") + agentProcessLogsPath
	if raw := c.Request.URL.RawQuery; raw != "" {
		upstream += "?" + raw
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, upstream, nil)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "upstream_build_failed"})
		return
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	req.Header.Set("Accept", "application/x-ndjson, application/json")

	dialer := &net.Dialer{Timeout: proxyCfg.ConnectTimeout}
	transport := &http.Transport{
		DialContext: func(dctx context.Context, network, addr string) (net.Conn, error) {
			dctx2, cancel := context.WithTimeout(dctx, proxyCfg.ConnectTimeout)
			defer cancel()
			return dialer.DialContext(dctx2, network, addr)
		},
	}
	client := &http.Client{Transport: transport, Timeout: 0}

	streamCtx := ctx
	var cancel context.CancelFunc
	if strings.EqualFold(q.Get("follow"), "1") || strings.EqualFold(q.Get("follow"), "true") || q.Get("follow") == "yes" {
		streamCtx, cancel = context.WithTimeout(ctx, proxyCfg.MaxStreamDuration)
		defer cancel()
		req = req.WithContext(streamCtx)
	}

	resp, err := client.Do(req)
	if err != nil {
		var netErr net.Error
		if errors.As(err, &netErr) && netErr.Timeout() {
			c.JSON(http.StatusGatewayTimeout, gin.H{"error": "agent_timeout"})
			return
		}
		if errors.Is(err, context.DeadlineExceeded) {
			c.JSON(http.StatusGatewayTimeout, gin.H{"error": "agent_timeout"})
			return
		}
		logger.Logger.Debug().Err(err).Str("node_id", nodeID).Msg("node logs proxy upstream error")
		c.JSON(http.StatusBadGateway, gin.H{"error": "agent_unreachable", "message": err.Error()})
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusNotFound || resp.StatusCode == http.StatusRequestEntityTooLarge {
		c.Status(resp.StatusCode)
		_, _ = io.Copy(c.Writer, resp.Body)
		return
	}
	if resp.StatusCode != http.StatusOK {
		c.JSON(http.StatusBadGateway, gin.H{"error": "agent_bad_response", "status": resp.StatusCode})
		return
	}

	c.Header("Content-Type", "application/x-ndjson")
	c.Header("Cache-Control", "no-store")
	c.Header("X-Content-Type-Options", "nosniff")
	c.Status(http.StatusOK)

	flusher, _ := c.Writer.(http.Flusher)
	// Idle timeout between chunks while streaming
	idle := proxyCfg.StreamIdleTimeout
	if idle <= 0 {
		idle = 60 * time.Second
	}

	if strings.EqualFold(q.Get("follow"), "1") || strings.EqualFold(q.Get("follow"), "true") || q.Get("follow") == "yes" {
		err = streamNDJSONWithIdle(c.Writer, resp.Body, idle, flusher, streamCtx)
	} else {
		_, err = io.Copy(c.Writer, resp.Body)
		if flusher != nil {
			flusher.Flush()
		}
	}
	if err != nil && !errors.Is(err, context.Canceled) {
		logger.Logger.Debug().Err(err).Str("node_id", nodeID).Msg("node logs proxy copy ended")
	}
}

func streamNDJSONWithIdle(w io.Writer, r io.Reader, idle time.Duration, flusher http.Flusher, ctx context.Context) error {
	type result struct {
		line []byte
		err  error
	}
	ch := make(chan result, 8)
	go func() {
		defer close(ch)
		sc := bufio.NewScanner(r)
		const maxLine = 2 << 20 // 2 MiB per line safety cap
		sc.Buffer(make([]byte, 0, 64*1024), maxLine)
		for sc.Scan() {
			b := append([]byte(nil), sc.Bytes()...)
			b = append(b, '\n')
			select {
			case ch <- result{line: b}:
			case <-ctx.Done():
				return
			}
		}
		if err := sc.Err(); err != nil {
			select {
			case ch <- result{err: err}:
			case <-ctx.Done():
			}
		}
	}()

	timer := time.NewTimer(idle)
	defer timer.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-timer.C:
			return context.DeadlineExceeded
		case res, ok := <-ch:
			if !ok {
				return nil
			}
			if res.err != nil {
				return res.err
			}
			if len(res.line) == 0 {
				continue
			}
			if _, err := w.Write(res.line); err != nil {
				return err
			}
			if flusher != nil {
				flusher.Flush()
			}
			if !timer.Stop() {
				select {
				case <-timer.C:
				default:
				}
			}
			timer.Reset(idle)
		}
	}
}
