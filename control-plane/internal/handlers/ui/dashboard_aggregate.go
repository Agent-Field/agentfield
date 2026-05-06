package ui

import (
	"context"
	"sort"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

// buildEnhancedOverviewForRange builds overview metrics for a specific time range
func (h *DashboardHandler) buildEnhancedOverviewForRange(agents []*types.AgentNode, executions []*types.Execution, startTime, endTime time.Time) EnhancedOverview {
	overview := EnhancedOverview{TotalAgents: len(agents)}

	for _, agent := range agents {
		overview.TotalReasoners += len(agent.Reasoners)
		overview.TotalSkills += len(agent.Skills)

		isDegraded := agent.LifecycleStatus == types.AgentStatusDegraded || agent.HealthStatus == types.HealthStatusInactive
		if isDegraded {
			overview.DegradedAgents++
			continue
		}

		status, err := h.agentService.GetAgentStatus(agent.ID)
		if err != nil {
			overview.OfflineAgents++
			continue
		}

		if status != nil && status.IsRunning {
			overview.ActiveAgents++
		} else {
			overview.OfflineAgents++
		}
	}

	if overview.OfflineAgents < 0 {
		overview.OfflineAgents = 0
	}

	var durationSamples []int64
	var durationSum float64
	var durationCount float64
	var successCount, failedCount int

	for _, exec := range executions {
		overview.ExecutionsLast24h++ // Repurposed as "executions in range"

		normalized := types.NormalizeExecutionStatus(exec.Status)
		switch normalized {
		case string(types.ExecutionStatusSucceeded):
			successCount++
		case string(types.ExecutionStatusFailed), string(types.ExecutionStatusCancelled), string(types.ExecutionStatusTimeout):
			failedCount++
		}

		if exec.DurationMS != nil {
			d := *exec.DurationMS
			durationSamples = append(durationSamples, d)
			durationSum += float64(d)
			durationCount++
		}
	}

	overview.ExecutionsLast7d = len(executions)
	if len(executions) > 0 {
		overview.SuccessRate24h = (float64(successCount) / float64(len(executions))) * 100
	}
	if durationCount > 0 {
		overview.AverageDurationMs24h = durationSum / durationCount
	}
	overview.MedianDurationMs24h = computeMedian(durationSamples)

	return overview
}

// buildExecutionTrendsForRange builds trends for the specified time range
func buildExecutionTrendsForRange(executions []*types.Execution, startTime, endTime time.Time, preset TimeRangePreset) ExecutionTrends {
	trend := ExecutionTrends{}
	duration := endTime.Sub(startTime)

	// Determine bucket size based on preset
	var bucketDuration time.Duration
	var numBuckets int
	switch preset {
	case TimeRangePreset1h:
		bucketDuration = 5 * time.Minute
		numBuckets = 12
	case TimeRangePreset24h:
		bucketDuration = time.Hour
		numBuckets = 24
	case TimeRangePreset7d:
		bucketDuration = 24 * time.Hour
		numBuckets = 7
	case TimeRangePreset30d:
		bucketDuration = 24 * time.Hour
		numBuckets = 30
	default:
		// For custom, use day buckets capped at 30
		bucketDuration = 24 * time.Hour
		numBuckets = int(duration.Hours() / 24)
		if numBuckets > 30 {
			numBuckets = 30
		}
		if numBuckets < 1 {
			numBuckets = 1
		}
	}

	// Create buckets
	dayBuckets := make(map[string]*ExecutionTrendPoint)
	orderedKeys := make([]string, 0, numBuckets)

	for i := numBuckets - 1; i >= 0; i-- {
		bucketTime := endTime.Add(-time.Duration(i) * bucketDuration)
		var key string
		if bucketDuration >= 24*time.Hour {
			key = bucketTime.Format("2006-01-02")
		} else {
			key = bucketTime.Format("2006-01-02T15:04")
		}
		orderedKeys = append(orderedKeys, key)
		dayBuckets[key] = &ExecutionTrendPoint{Date: key}
	}

	var totalInRange, successInRange, failedInRange int
	var durationSum float64
	var durationCount float64

	for _, exec := range executions {
		var bucketKey string
		if bucketDuration >= 24*time.Hour {
			bucketKey = exec.StartedAt.Format("2006-01-02")
		} else {
			// Round to bucket
			bucketKey = exec.StartedAt.Truncate(bucketDuration).Format("2006-01-02T15:04")
		}

		if point, ok := dayBuckets[bucketKey]; ok {
			point.Total++
			normalized := types.NormalizeExecutionStatus(exec.Status)
			switch normalized {
			case string(types.ExecutionStatusSucceeded):
				point.Succeeded++
			case string(types.ExecutionStatusFailed), string(types.ExecutionStatusCancelled), string(types.ExecutionStatusTimeout):
				point.Failed++
			}
		}

		totalInRange++
		normalized := types.NormalizeExecutionStatus(exec.Status)
		switch normalized {
		case string(types.ExecutionStatusSucceeded):
			successInRange++
		case string(types.ExecutionStatusFailed), string(types.ExecutionStatusCancelled), string(types.ExecutionStatusTimeout):
			failedInRange++
		}

		if exec.DurationMS != nil {
			durationSum += float64(*exec.DurationMS)
			durationCount++
		}
	}

	trend.Last7Days = make([]ExecutionTrendPoint, 0, len(orderedKeys))
	for _, key := range orderedKeys {
		trend.Last7Days = append(trend.Last7Days, *dayBuckets[key])
	}

	trend.Last24h.Total = totalInRange
	trend.Last24h.Succeeded = successInRange
	trend.Last24h.Failed = failedInRange
	if totalInRange > 0 {
		trend.Last24h.SuccessRate = (float64(successInRange) / float64(totalInRange)) * 100
		hours := duration.Hours()
		if hours > 0 {
			trend.Last24h.ThroughputPerHour = float64(totalInRange) / hours
		}
	}
	if durationCount > 0 {
		trend.Last24h.AverageDurationMs = durationSum / durationCount
	}

	return trend
}

// buildComparisonData creates comparison metrics between current and previous periods
func buildComparisonData(current, previous EnhancedOverview, prevStart, prevEnd time.Time) *ComparisonData {
	delta := EnhancedOverviewDelta{}

	// Executions delta
	delta.ExecutionsDelta = current.ExecutionsLast24h - previous.ExecutionsLast24h
	if previous.ExecutionsLast24h > 0 {
		delta.ExecutionsDeltaPct = (float64(delta.ExecutionsDelta) / float64(previous.ExecutionsLast24h)) * 100
	}

	// Success rate delta
	delta.SuccessRateDelta = current.SuccessRate24h - previous.SuccessRate24h

	// Duration delta
	delta.AvgDurationDeltaMs = current.AverageDurationMs24h - previous.AverageDurationMs24h
	if previous.AverageDurationMs24h > 0 {
		delta.AvgDurationDeltaPct = (delta.AvgDurationDeltaMs / previous.AverageDurationMs24h) * 100
	}

	return &ComparisonData{
		PreviousPeriod: TimeRangeInfo{
			StartTime: prevStart,
			EndTime:   prevEnd,
		},
		OverviewDelta: delta,
	}
}

// buildHotspotSummary aggregates failures by reasoner
func buildHotspotSummary(executions []*types.Execution) HotspotSummary {
	type reasonerStats struct {
		total     int
		failed    int
		errorMsgs map[string]int
	}

	statsMap := make(map[string]*reasonerStats)
	totalFailures := 0

	for _, exec := range executions {
		if exec.ReasonerID == "" {
			continue
		}

		stats, ok := statsMap[exec.ReasonerID]
		if !ok {
			stats = &reasonerStats{errorMsgs: make(map[string]int)}
			statsMap[exec.ReasonerID] = stats
		}

		stats.total++

		normalized := types.NormalizeExecutionStatus(exec.Status)
		if normalized == string(types.ExecutionStatusFailed) ||
			normalized == string(types.ExecutionStatusCancelled) ||
			normalized == string(types.ExecutionStatusTimeout) {
			stats.failed++
			totalFailures++

			if exec.ErrorMessage != nil && *exec.ErrorMessage != "" {
				// Truncate long error messages
				errMsg := *exec.ErrorMessage
				if len(errMsg) > 100 {
					errMsg = errMsg[:100] + "..."
				}
				stats.errorMsgs[errMsg]++
			}
		}
	}

	// Convert to slice and sort by failure count
	items := make([]HotspotItem, 0, len(statsMap))
	for reasonerID, stats := range statsMap {
		if stats.failed == 0 {
			continue
		}

		item := HotspotItem{
			ReasonerID:       reasonerID,
			TotalExecutions:  stats.total,
			FailedExecutions: stats.failed,
		}

		if stats.total > 0 {
			item.ErrorRate = (float64(stats.failed) / float64(stats.total)) * 100
		}
		if totalFailures > 0 {
			item.ContributionPct = (float64(stats.failed) / float64(totalFailures)) * 100
		}

		// Get top errors (up to 3)
		topErrors := make([]ErrorCount, 0, 3)
		for msg, count := range stats.errorMsgs {
			topErrors = append(topErrors, ErrorCount{Message: msg, Count: count})
		}
		sort.Slice(topErrors, func(i, j int) bool {
			return topErrors[i].Count > topErrors[j].Count
		})
		if len(topErrors) > 3 {
			topErrors = topErrors[:3]
		}
		item.TopErrors = topErrors

		items = append(items, item)
	}

	// Sort by failure count descending
	sort.Slice(items, func(i, j int) bool {
		return items[i].FailedExecutions > items[j].FailedExecutions
	})

	// Limit to top 10
	if len(items) > 10 {
		items = items[:10]
	}

	return HotspotSummary{TopFailingReasoners: items}
}

// buildActivityPatterns creates a 7x24 heatmap of execution activity
func buildActivityPatterns(executions []*types.Execution) ActivityPatterns {
	// Initialize 7x24 grid (Sunday=0 through Saturday=6)
	heatmap := make([][]HeatmapCell, 7)
	for day := 0; day < 7; day++ {
		heatmap[day] = make([]HeatmapCell, 24)
	}

	for _, exec := range executions {
		dayOfWeek := int(exec.StartedAt.Weekday())
		hourOfDay := exec.StartedAt.Hour()

		heatmap[dayOfWeek][hourOfDay].Total++

		normalized := types.NormalizeExecutionStatus(exec.Status)
		if normalized == string(types.ExecutionStatusFailed) ||
			normalized == string(types.ExecutionStatusCancelled) ||
			normalized == string(types.ExecutionStatusTimeout) {
			heatmap[dayOfWeek][hourOfDay].Failed++
		}
	}

	// Calculate error rates
	for day := 0; day < 7; day++ {
		for hour := 0; hour < 24; hour++ {
			if heatmap[day][hour].Total > 0 {
				heatmap[day][hour].ErrorRate = (float64(heatmap[day][hour].Failed) / float64(heatmap[day][hour].Total)) * 100
			}
		}
	}

	return ActivityPatterns{HourlyHeatmap: heatmap}
}

func (h *DashboardHandler) buildEnhancedOverview(now time.Time, agents []*types.AgentNode, executions []*types.Execution) EnhancedOverview {
	overview := EnhancedOverview{TotalAgents: len(agents)}

	for _, agent := range agents {
		// Count reasoners and skills
		overview.TotalReasoners += len(agent.Reasoners)
		overview.TotalSkills += len(agent.Skills)

		isDegraded := agent.LifecycleStatus == types.AgentStatusDegraded || agent.HealthStatus == types.HealthStatusInactive
		if isDegraded {
			overview.DegradedAgents++
			continue
		}

		status, err := h.agentService.GetAgentStatus(agent.ID)
		if err != nil {
			overview.OfflineAgents++
			continue
		}

		if status != nil && status.IsRunning {
			overview.ActiveAgents++
		} else {
			overview.OfflineAgents++
		}
	}

	// Ensure offline count is consistent
	if overview.OfflineAgents < 0 {
		overview.OfflineAgents = 0
	}

	last24h := now.Add(-24 * time.Hour)
	var durationSamples []int64
	var durationSum float64
	var durationCount float64
	var success24h, failed24h int

	for _, exec := range executions {
		if exec.StartedAt.After(last24h) || exec.StartedAt.Equal(last24h) {
			overview.ExecutionsLast24h++

			normalized := types.NormalizeExecutionStatus(exec.Status)
			switch normalized {
			case string(types.ExecutionStatusSucceeded):
				success24h++
			case string(types.ExecutionStatusFailed), string(types.ExecutionStatusCancelled), string(types.ExecutionStatusTimeout):
				failed24h++
			}

			if exec.DurationMS != nil {
				d := *exec.DurationMS
				durationSamples = append(durationSamples, d)
				durationSum += float64(d)
				durationCount++
			}
		}
	}

	overview.ExecutionsLast7d = len(executions)
	if overview.ExecutionsLast24h > 0 {
		overview.SuccessRate24h = (float64(success24h) / float64(overview.ExecutionsLast24h)) * 100
	}
	if durationCount > 0 {
		overview.AverageDurationMs24h = durationSum / durationCount
	}
	overview.MedianDurationMs24h = computeMedian(durationSamples)

	return overview
}

func buildExecutionTrends(now time.Time, executions []*types.Execution) ExecutionTrends {
	trend := ExecutionTrends{}
	last24h := now.Add(-24 * time.Hour)
	var total24h, success24h, failed24h int
	var durationSum float64
	var durationCount float64

	// Prepare day buckets for the last 7 days (including today)
	dayBuckets := make(map[string]*ExecutionTrendPoint)
	orderedDays := make([]string, 0, 7)
	for i := 6; i >= 0; i-- {
		day := now.AddDate(0, 0, -i)
		key := day.Format("2006-01-02")
		orderedDays = append(orderedDays, key)
		dayBuckets[key] = &ExecutionTrendPoint{Date: key}
	}

	for _, exec := range executions {
		dayKey := exec.StartedAt.Format("2006-01-02")
		point, ok := dayBuckets[dayKey]
		if ok {
			point.Total++
			normalized := types.NormalizeExecutionStatus(exec.Status)
			switch normalized {
			case string(types.ExecutionStatusSucceeded):
				point.Succeeded++
			case string(types.ExecutionStatusFailed), string(types.ExecutionStatusCancelled), string(types.ExecutionStatusTimeout):
				point.Failed++
			}
		}

		if exec.StartedAt.After(last24h) || exec.StartedAt.Equal(last24h) {
			total24h++
			normalized := types.NormalizeExecutionStatus(exec.Status)
			switch normalized {
			case string(types.ExecutionStatusSucceeded):
				success24h++
			case string(types.ExecutionStatusFailed), string(types.ExecutionStatusCancelled), string(types.ExecutionStatusTimeout):
				failed24h++
			}

			if exec.DurationMS != nil {
				durationSum += float64(*exec.DurationMS)
				durationCount++
			}
		}
	}

	trend.Last7Days = make([]ExecutionTrendPoint, 0, len(orderedDays))
	for _, key := range orderedDays {
		trend.Last7Days = append(trend.Last7Days, *dayBuckets[key])
	}

	trend.Last24h.Total = total24h
	trend.Last24h.Succeeded = success24h
	trend.Last24h.Failed = failed24h
	if total24h > 0 {
		trend.Last24h.SuccessRate = (float64(success24h) / float64(total24h)) * 100
		trend.Last24h.ThroughputPerHour = float64(total24h) / 24.0
	}
	if durationCount > 0 {
		trend.Last24h.AverageDurationMs = durationSum / durationCount
	}

	return trend
}

func (h *DashboardHandler) buildAgentHealthSummary(ctx context.Context, agents []*types.AgentNode) AgentHealthSummary {
	summary := AgentHealthSummary{Total: len(agents)}
	items := make([]AgentHealthItem, 0, len(agents))

	for _, agent := range agents {
		item := AgentHealthItem{
			ID:            agent.ID,
			TeamID:        agent.TeamID,
			Version:       agent.Version,
			Health:        string(agent.HealthStatus),
			Lifecycle:     string(agent.LifecycleStatus),
			LastHeartbeat: agent.LastHeartbeat,
			Reasoners:     len(agent.Reasoners),
			Skills:        len(agent.Skills),
		}

		isDegraded := agent.LifecycleStatus == types.AgentStatusDegraded || agent.HealthStatus == types.HealthStatusInactive
		if isDegraded {
			summary.Degraded++
			item.Status = "degraded"
			items = append(items, item)
			continue
		}

		status, err := h.agentService.GetAgentStatus(agent.ID)
		if err != nil {
			summary.Offline++
			item.Status = "offline"
			items = append(items, item)
			continue
		}

		if status != nil {
			item.Uptime = status.Uptime
			if status.IsRunning {
				summary.Active++
				item.Status = "running"
			} else {
				summary.Offline++
				item.Status = "offline"
			}
		} else {
			summary.Offline++
			item.Status = "offline"
		}

		items = append(items, item)
	}

	// Derive offline count if we encountered transient errors
	if summary.Offline < 0 {
		summary.Offline = 0
	}

	priority := map[string]int{
		"degraded": 0,
		"offline":  1,
		"running":  2,
		"unknown":  3,
	}

	sort.Slice(items, func(i, j int) bool {
		pi := priority[items[i].Status]
		pj := priority[items[j].Status]
		if pi == pj {
			return items[i].LastHeartbeat.After(items[j].LastHeartbeat)
		}
		return pi < pj
	})

	if len(items) > 12 {
		items = items[:12]
	}

	summary.Agents = items
	return summary
}

func buildWorkflowInsights(executions []*types.Execution, running []*types.Execution) WorkflowInsights {
	insights := WorkflowInsights{}
	workflowAggregates := make(map[string]*WorkflowStat)

	for _, exec := range executions {
		id := exec.RunID
		aggregate, ok := workflowAggregates[id]
		if !ok {
			aggregate = &WorkflowStat{
				WorkflowID: id,
				Name:       exec.ReasonerID,
			}
			workflowAggregates[id] = aggregate
		}

		aggregate.TotalExecutions++
		aggregate.LastActivity = maxTime(aggregate.LastActivity, exec.StartedAt)
		if exec.DurationMS != nil {
			aggregate.AverageDuration += float64(*exec.DurationMS)
		}

		normalized := types.NormalizeExecutionStatus(exec.Status)
		switch normalized {
		case string(types.ExecutionStatusSucceeded):
			aggregate.SuccessRate++
		case string(types.ExecutionStatusFailed), string(types.ExecutionStatusCancelled), string(types.ExecutionStatusTimeout):
			aggregate.FailedExecutions++
		}
	}

	topWorkflows := make([]WorkflowStat, 0, len(workflowAggregates))
	for _, aggregate := range workflowAggregates {
		if aggregate.TotalExecutions > 0 {
			aggregate.AverageDuration = aggregate.AverageDuration / float64(aggregate.TotalExecutions)
			aggregate.SuccessRate = (aggregate.SuccessRate / float64(aggregate.TotalExecutions)) * 100
		}
		topWorkflows = append(topWorkflows, *aggregate)
	}

	sort.Slice(topWorkflows, func(i, j int) bool {
		if topWorkflows[i].TotalExecutions == topWorkflows[j].TotalExecutions {
			return topWorkflows[i].LastActivity.After(topWorkflows[j].LastActivity)
		}
		return topWorkflows[i].TotalExecutions > topWorkflows[j].TotalExecutions
	})

	if len(topWorkflows) > 5 {
		topWorkflows = topWorkflows[:5]
	}

	insights.TopWorkflows = topWorkflows

	activeRuns := make([]ActiveWorkflowRun, 0, len(running))
	for _, exec := range running {
		elapsed := time.Since(exec.StartedAt).Milliseconds()
		activeRuns = append(activeRuns, ActiveWorkflowRun{
			ExecutionID: exec.ExecutionID,
			WorkflowID:  exec.RunID,
			Name:        exec.ReasonerID,
			StartedAt:   exec.StartedAt,
			ElapsedMs:   elapsed,
			AgentNodeID: exec.AgentNodeID,
			ReasonerID:  exec.ReasonerID,
			Status:      exec.Status,
		})
	}

	sort.Slice(activeRuns, func(i, j int) bool {
		return activeRuns[i].ElapsedMs > activeRuns[j].ElapsedMs
	})
	if len(activeRuns) > 6 {
		activeRuns = activeRuns[:6]
	}
	insights.ActiveRuns = activeRuns

	completed := make([]CompletedExecutionStat, 0, len(executions))
	for _, exec := range executions {
		if exec.DurationMS == nil || exec.CompletedAt == nil {
			continue
		}
		completed = append(completed, CompletedExecutionStat{
			ExecutionID: exec.ExecutionID,
			WorkflowID:  exec.RunID,
			Name:        exec.ReasonerID,
			DurationMs:  *exec.DurationMS,
			CompletedAt: exec.CompletedAt,
			Status:      exec.Status,
		})
	}

	sort.Slice(completed, func(i, j int) bool {
		if completed[i].DurationMs == completed[j].DurationMs {
			return completed[i].CompletedAt.After(*completed[j].CompletedAt)
		}
		return completed[i].DurationMs > completed[j].DurationMs
	})
	if len(completed) > 5 {
		completed = completed[:5]
	}

	insights.LongestExecutions = completed
	return insights
}

func buildIncidentItems(executions []*types.Execution, limit int) []IncidentItem {
	incidents := make([]IncidentItem, 0, limit)

	for _, exec := range executions {
		normalized := types.NormalizeExecutionStatus(exec.Status)
		if normalized != string(types.ExecutionStatusFailed) &&
			normalized != string(types.ExecutionStatusTimeout) &&
			normalized != string(types.ExecutionStatusCancelled) {
			continue
		}

		errorMessage := ""
		if exec.ErrorMessage != nil {
			errorMessage = *exec.ErrorMessage
		}

		incidents = append(incidents, IncidentItem{
			ExecutionID: exec.ExecutionID,
			WorkflowID:  exec.RunID,
			Name:        exec.ReasonerID,
			Status:      exec.Status,
			StartedAt:   exec.StartedAt,
			CompletedAt: exec.CompletedAt,
			AgentNodeID: exec.AgentNodeID,
			ReasonerID:  exec.ReasonerID,
			Error:       errorMessage,
		})
	}

	sort.Slice(incidents, func(i, j int) bool {
		return incidents[i].StartedAt.After(incidents[j].StartedAt)
	})

	if len(incidents) > limit {
		incidents = incidents[:limit]
	}

	return incidents
}

func computeMedian(values []int64) float64 {
	if len(values) == 0 {
		return 0
	}

	sort.Slice(values, func(i, j int) bool { return values[i] < values[j] })
	mid := len(values) / 2
	if len(values)%2 == 1 {
		return float64(values[mid])
	}
	return float64(values[mid-1]+values[mid]) / 2.0
}

func maxTime(current time.Time, candidate time.Time) time.Time {
	if current.IsZero() {
		return candidate
	}
	if candidate.After(current) {
		return candidate
	}
	return current
}

// getAgentsSummary collects agent statistics
func (h *DashboardHandler) getAgentsSummary(ctx context.Context) (AgentsSummary, error) {
	// Get all registered agents
	agents, err := h.storage.ListAgents(ctx, types.AgentFilters{})
	if err != nil {
		return AgentsSummary{}, err
	}

	total := len(agents)
	running := 0

	// Count running agents using the agent service
	for _, agent := range agents {
		if status, err := h.agentService.GetAgentStatus(agent.ID); err == nil && status.IsRunning {
			running++
		}
	}

	return AgentsSummary{
		Running: running,
		Total:   total,
	}, nil
}

// getExecutionsSummaryAndSuccessRate collects execution statistics and calculates success rate
func (h *DashboardHandler) getExecutionsSummaryAndSuccessRate(ctx context.Context, now time.Time) (ExecutionsSummary, float64, error) {
	today := time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, now.Location())
	yesterday := today.AddDate(0, 0, -1)
	tomorrow := today.AddDate(0, 0, 1)

	// Get today's executions
	todayFilters := types.ExecutionFilter{
		StartTime:       &today,
		EndTime:         &tomorrow,
		Limit:           10000,
		SortBy:          "started_at",
		SortDescending:  false,
		ExcludePayloads: true,
	}
	todayExecutions, err := h.store.QueryExecutionRecords(ctx, todayFilters)
	if err != nil {
		return ExecutionsSummary{}, 0, err
	}

	// Get yesterday's executions
	yesterdayFilters := types.ExecutionFilter{
		StartTime:       &yesterday,
		EndTime:         &today,
		Limit:           10000,
		SortBy:          "started_at",
		SortDescending:  false,
		ExcludePayloads: true,
	}
	yesterdayExecutions, err := h.store.QueryExecutionRecords(ctx, yesterdayFilters)
	if err != nil {
		return ExecutionsSummary{}, 0, err
	}

	// Calculate success rate from today's executions
	successRate := h.calculateSuccessRate(todayExecutions)

	return ExecutionsSummary{
		Today:     len(todayExecutions),
		Yesterday: len(yesterdayExecutions),
	}, successRate, nil
}

// calculateSuccessRate calculates the success rate from executions
func (h *DashboardHandler) calculateSuccessRate(executions []*types.Execution) float64 {
	if len(executions) == 0 {
		return 0.0
	}

	successCount := 0
	for _, exec := range executions {
		if types.NormalizeExecutionStatus(exec.Status) == types.ExecutionStatusSucceeded {
			successCount++
		}
	}

	return float64(successCount) / float64(len(executions)) * 100.0
}

// getPackagesSummary collects package statistics
func (h *DashboardHandler) getPackagesSummary(ctx context.Context) (PackagesSummary, error) {
	// Get all agent packages
	packages, err := h.storage.QueryAgentPackages(ctx, types.PackageFilters{})
	if err != nil {
		return PackagesSummary{}, err
	}

	available := len(packages)
	installed := 0

	// Count installed packages (packages with configuration or no configuration required)
	for _, pkg := range packages {
		configRequired := len(pkg.ConfigurationSchema) > 0

		if !configRequired {
			// No configuration required means it's installed
			installed++
		} else {
			// Check if configuration exists and is active
			if config, err := h.storage.GetAgentConfiguration(ctx, pkg.ID, pkg.ID); err == nil {
				if config.Status == types.ConfigurationStatusActive || config.Status == types.ConfigurationStatusDraft {
					installed++
				}
			}
		}
	}

	return PackagesSummary{
		Available: available,
		Installed: installed,
	}, nil
}
