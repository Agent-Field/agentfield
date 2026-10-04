package observability

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

const usageSafeInteger int64 = 9007199254740991
const usageOutboxLimit = 10000
const usageRetention = 30 * 24 * time.Hour

func usageProvider(provider string) string {
	switch strings.ToLower(strings.TrimSpace(provider)) {
	case "openrouter", "openai", "anthropic", "google", "bedrock", "azure", "ollama":
		return strings.ToLower(strings.TrimSpace(provider))
	case "", "unknown":
		return "unknown"
	default:
		return "other"
	}
}

func usageModelFamily(model string) string {
	model = strings.ToLower(strings.TrimSpace(model))
	if model == "" || model == "unknown" {
		return "unknown"
	}
	// Response models may have one vendor prefix; never emit that raw prefix.
	if i := strings.LastIndex(model, "/"); i >= 0 {
		model = model[i+1:]
	}
	for _, family := range strings.Fields("deepseek claude gpt gemini llama qwen kimi glm") {
		suffix := strings.TrimPrefix(model, family)
		if model == family || (suffix != model && suffix != "" && (suffix[0] == '-' || suffix[0] == '.' || (suffix[0] >= '0' && suffix[0] <= '9'))) {
			return family
		}
	}
	return "other"
}

func (s *TelemetryService) recordUsage(executionID string, index int, row *types.ExecutionUsage) {
	if s == nil || row == nil || executionID == "" || s.usageOutbox == "" || row.Source != "llm" {
		return
	}
	props := s.withUsageContext(map[string]interface{}{
		"routing_provider": usageProvider(row.RoutingProvider), "model_family": usageModelFamily(row.Model),
		"usage_status": "reported", "accounting_source": "control_plane",
	})
	if row.UsageStatus == "missing" {
		props["usage_status"] = "missing"
	} else {
		if row.InputTokens < 0 || row.OutputTokens < 0 || row.TotalTokens < 0 || row.InputTokens > usageSafeInteger || row.OutputTokens > usageSafeInteger || row.TotalTokens > usageSafeInteger {
			return
		}
		props["input_tokens"] = row.InputTokens
		props["output_tokens"] = row.OutputTokens
		if row.InputTokens > usageSafeInteger-row.OutputTokens {
			logger.Logger.Warn().Msg("anonymous usage token sum exceeds safe integer; skipping event")
			return
		}
		props["total_tokens"] = row.InputTokens + row.OutputTokens
	}
	event := TelemetryEvent{SchemaVersion: telemetrySchemaVersion, EventID: s.eventIdentity("usage_delta", fmt.Sprintf("%s\x00%d", executionID, index)), EventName: "usage_delta", AnonymousInstallIDHash: s.installHash, EventTime: time.Now().UTC().Format(time.RFC3339), Component: "control-plane", AgentFieldVersion: s.version, Runtime: s.runtimeName, StorageMode: s.storageMode, Properties: props}
	if err := s.storeUsage(event); err != nil {
		logger.Logger.Warn().Err(err).Msg("anonymous usage outbox failed; usage remains in local execution accounting")
		return
	}
	select {
	case s.usageWake <- struct{}{}:
	default:
	}
}

// Pending files survive restart. Acknowledgement tombstones prevent callback
// replay from re-counting a call; the hosted relay also deduplicates EventID.
// Both pending events and tombstones expire after 30 days, with an explicit
// warning for expired pending data. Only sanitized metadata is written here.
func (s *TelemetryService) storeUsage(event TelemetryEvent) error {
	s.usageMu.Lock()
	defer s.usageMu.Unlock()
	if err := os.MkdirAll(s.usageOutbox, 0700); err != nil {
		return err
	}
	path := filepath.Join(s.usageOutbox, event.EventID)
	for _, file := range []string{path + ".json", filepath.Join(s.usageOutbox, "ack", event.EventID+".sent")} {
		if _, err := os.Stat(file); err == nil {
			return nil
		}
	}
	if !s.usagePendingInitialized {
		files, err := os.ReadDir(s.usageOutbox)
		if err != nil {
			return err
		}
		s.usagePending = 0
		for _, file := range files {
			if strings.HasSuffix(file.Name(), ".json") {
				s.usagePending++
			}
		}
		s.usagePendingInitialized = true
	}
	if s.usagePending >= usageOutboxLimit {
		return fmt.Errorf("usage outbox reached %d pending events", usageOutboxLimit)
	}
	data, err := json.Marshal(event)
	if err != nil {
		return err
	}
	f, err := os.CreateTemp(s.usageOutbox, ".usage-")
	if err != nil {
		return err
	}
	tmp := f.Name()
	defer os.Remove(tmp)
	if err = f.Chmod(0600); err == nil {
		_, err = f.Write(data)
	}
	if err == nil {
		err = f.Sync()
	}
	closeErr := f.Close()
	if err != nil {
		return err
	}
	if closeErr != nil {
		return closeErr
	}
	if err = os.Rename(tmp, path+".json"); err != nil {
		return err
	}
	s.usagePending++
	return syncUsageDirectory(s.usageOutbox)
}

func syncUsageDirectory(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return f.Sync()
}

func (s *TelemetryService) usageWorker() {
	defer s.wg.Done()
	tick := time.NewTicker(10 * time.Second)
	defer tick.Stop()
	cleanup := time.NewTicker(10 * time.Minute)
	defer cleanup.Stop()
	s.cleanupUsageAcknowledgements()
	s.flushUsage()
	for {
		select {
		case <-s.ctx.Done():
			return
		case <-cleanup.C:
			s.cleanupUsageAcknowledgements()
		case <-tick.C:
			s.flushUsage()
		case <-s.usageWake:
			s.flushUsage()
		}
	}
}

func (s *TelemetryService) flushUsage() {
	files, err := os.ReadDir(s.usageOutbox)
	if err != nil {
		return
	}
	for _, file := range files {
		if s.ctx.Err() != nil {
			return
		}
		name := file.Name()
		if !strings.HasSuffix(name, ".json") {
			continue
		}
		path := filepath.Join(s.usageOutbox, name)
		info, err := file.Info()
		if err != nil {
			continue
		}
		if time.Since(info.ModTime()) > usageRetention {
			if strings.HasSuffix(name, ".json") {
				logger.Logger.Warn().Msg("anonymous usage outbox expired an undelivered event after 30 days")
			}
			s.usageMu.Lock()
			if os.Remove(path) == nil && s.usagePendingInitialized {
				s.usagePending--
			}
			s.usageMu.Unlock()
			continue
		}
		if !strings.HasSuffix(name, ".json") {
			continue
		}
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		var event TelemetryEvent
		if json.Unmarshal(data, &event) != nil {
			continue
		}
		ctx, cancel := context.WithTimeout(s.ctx, s.timeout)
		err = s.sender(ctx, s.cfg.Endpoint, s.timeout, event)
		cancel()
		if err != nil {
			return
		} // retry later, without bypassing an unavailable relay
		s.usageMu.Lock()
		ackDir := filepath.Join(s.usageOutbox, "ack")
		err = os.MkdirAll(ackDir, 0700)
		if err == nil {
			err = os.Rename(path, filepath.Join(ackDir, strings.TrimSuffix(name, ".json")+".sent"))
		}
		if err == nil {
			if s.usagePendingInitialized {
				s.usagePending--
			}
			_ = syncUsageDirectory(ackDir)
			_ = syncUsageDirectory(s.usageOutbox)
		}
		s.usageMu.Unlock()
	}
}

// Acknowledgements are separate from the hot pending directory. Cleanup scans
// history only at startup and every ten minutes, never for each new receipt.
// Keep at most 10,000 recent acknowledgements after cleanup; relay EventID
// deduplication remains the guard for retries outside this local window.
func (s *TelemetryService) cleanupUsageAcknowledgements() {
	s.usageMu.Lock()
	defer s.usageMu.Unlock()
	dir := filepath.Join(s.usageOutbox, "ack")
	files, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	type ack struct {
		name     string
		modified time.Time
	}
	recent := make([]ack, 0, len(files))
	for _, file := range files {
		if !strings.HasSuffix(file.Name(), ".sent") {
			continue
		}
		info, err := file.Info()
		if err != nil {
			continue
		}
		if time.Since(info.ModTime()) > usageRetention {
			_ = os.Remove(filepath.Join(dir, file.Name()))
			continue
		}
		recent = append(recent, ack{file.Name(), info.ModTime()})
	}
	sort.Slice(recent, func(i, j int) bool { return recent[i].modified.After(recent[j].modified) })
	if len(recent) > usageOutboxLimit {
		for _, file := range recent[usageOutboxLimit:] {
			_ = os.Remove(filepath.Join(dir, file.name))
		}
	}
}
