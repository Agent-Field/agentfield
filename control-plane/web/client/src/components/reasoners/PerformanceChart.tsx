import { TrendingUp, TrendingDown, Minus } from '@/components/ui/icon-bridge';
import type { PerformanceMetrics } from '../../types/execution';

interface PerformanceChartProps {
  metrics?: PerformanceMetrics | null;
}

export function PerformanceChart({ metrics }: PerformanceChartProps) {
  if (!metrics) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        <p>No performance data available</p>
      </div>
    );
  }

  // Extract and sanitize trend data from performance_trend
  const responseTimes = metrics.performance_trend?.map(t => {
    const time = t.avg_response_time;
    return typeof time === 'number' && !isNaN(time) ? time : 0;
  }) || [];

  const successRates = metrics.performance_trend?.map(t => {
    const rate = t.success_rate;
    return typeof rate === 'number' && !isNaN(rate) ? rate : 0;
  }) || [];

  const executionCounts = metrics.performance_trend?.map(t => {
    const count = t.execution_count;
    return typeof count === 'number' && !isNaN(count) ? count : 0;
  }) || [];

  // Calculate safe max values, ensuring no NaN results
  const maxResponseTime = responseTimes.length > 0 ? Math.max(...responseTimes, 100) : 100;
  const maxExecutions = executionCounts.length > 0 ? Math.max(...executionCounts, 1) : 1;

  // Final validation to ensure no NaN values
  const safeMaxResponseTime = isNaN(maxResponseTime) ? 100 : maxResponseTime;
  const safeMaxExecutions = isNaN(maxExecutions) ? 1 : maxExecutions;

  const getTrendIcon = (current: number, previous: number) => {
    if (current > previous) return <TrendingUp className="h-4 w-4 text-green-500" />;
    if (current < previous) return <TrendingDown className="h-4 w-4 text-red-500" />;
    return <Minus className="h-4 w-4 text-muted-foreground" />;
  };

  const formatPercentage = (value: number) => `${(value * 100).toFixed(1)}%`;

  return (
    <div className="space-y-6">
      {/* Key Metrics Summary */}
      <div className="grid grid-cols-2 gap-4">
        <div className="p-4 border rounded-lg">
          <div className="flex items-center justify-between">
            <h4 className="text-base font-semibold">Avg Response Time</h4>
            {responseTimes.length > 1 && (
              getTrendIcon(
                responseTimes[responseTimes.length - 1],
                responseTimes[responseTimes.length - 2]
              )
            )}
          </div>
          <p className="text-2xl font-semibold tracking-tight">{metrics.avg_response_time_ms}ms</p>
          <p className="text-sm text-muted-foreground">Last 24 hours</p>
        </div>

        <div className="p-4 border rounded-lg">
          <div className="flex items-center justify-between">
            <h4 className="text-base font-semibold">Success Rate</h4>
            {successRates.length > 1 && (
              getTrendIcon(
                successRates[successRates.length - 1],
                successRates[successRates.length - 2]
              )
            )}
          </div>
          <p className="text-2xl font-semibold tracking-tight">{formatPercentage(metrics.success_rate)}</p>
          <p className="text-sm text-muted-foreground">Last 24 hours</p>
        </div>
      </div>

      {/* Response Time Chart */}
      {responseTimes.length > 0 && (
        <div className="space-y-3">
          <h4 className="text-base font-semibold">Response Time Trend</h4>
          <div className="space-y-2">
            {responseTimes.map((time, index) => (
              <div key={index} className="flex items-center gap-3">
                <span className="text-sm text-muted-foreground w-8">
                  -{responseTimes.length - index}h
                </span>
                <div className="flex-1 bg-muted rounded-full h-2 relative">
                  <div
                    className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                    style={{
                      width: `${(time / safeMaxResponseTime) * 100}%`,
                    }}
                  />
                </div>
                <span className="text-sm text-muted-foreground font-mono w-12 text-right">{time}ms</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Execution Count Chart */}
      {executionCounts.length > 0 && (
        <div className="space-y-3">
          <h4 className="text-base font-semibold">Execution Volume</h4>
          <div className="space-y-2">
            {executionCounts.map((count, index) => (
              <div key={index} className="flex items-center gap-3">
                <span className="text-sm text-muted-foreground w-8">
                  -{executionCounts.length - index}h
                </span>
                <div className="flex-1 bg-muted rounded-full h-2 relative">
                  <div
                    className="bg-green-500 h-2 rounded-full transition-all duration-300"
                    style={{
                      width: `${(count / safeMaxExecutions) * 100}%`,
                    }}
                  />
                </div>
                <span className="text-xs font-mono w-8 text-right">{count}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Success Rate Chart */}
      {successRates.length > 0 && (
        <div className="space-y-3">
          <h4 className="font-medium">Success Rate Trend</h4>
          <div className="space-y-2">
            {successRates.map((rate, index) => (
              <div key={index} className="flex items-center gap-3">
                <span className="text-sm text-muted-foreground w-8">
                  -{successRates.length - index}h
                </span>
                <div className="flex-1 bg-muted rounded-full h-2 relative">
                  <div
                    className={`h-2 rounded-full transition-all duration-300 ${
                      rate >= 0.9 ? 'bg-green-500' : rate >= 0.7 ? 'bg-yellow-500' : 'bg-red-500'
                    }`}
                    style={{
                      width: `${rate * 100}%`,
                    }}
                  />
                </div>
                <span className="text-xs font-mono w-12 text-right">
                  {formatPercentage(rate)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Performance Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-4 border-t">
        <div className="text-center">
          <p className="text-sm text-muted-foreground">Total Executions</p>
          <p className="text-base font-semibold">{metrics.total_executions}</p>
        </div>
        <div className="text-center">
          <p className="text-sm text-muted-foreground">Last 24h</p>
          <p className="text-base font-semibold">{metrics.executions_last_24h}</p>
        </div>
        <div className="text-center">
          <p className="text-sm text-muted-foreground">Cost (24h)</p>
          <p className="text-base font-semibold">
            {metrics.cost_last_24h ? `$${metrics.cost_last_24h.toFixed(4)}` : 'N/A'}
          </p>
        </div>
      </div>

      {/* Performance Insights */}
      <div className="space-y-2 p-4 bg-muted/50 rounded-lg">
        <h4 className="font-medium">Performance Insights</h4>
        <div className="space-y-1 text-sm">
          {metrics.avg_response_time_ms < 1000 && (
            <p className="text-green-700">✓ Fast response times (under 1s)</p>
          )}
          {metrics.avg_response_time_ms >= 5000 && (
            <p className="text-yellow-700">⚠ Slow response times (over 5s)</p>
          )}
          {metrics.success_rate >= 0.95 && (
            <p className="text-green-700">✓ Excellent reliability (95%+ success)</p>
          )}
          {metrics.success_rate < 0.8 && (
            <p className="text-red-700">⚠ Low success rate (under 80%)</p>
          )}
          {metrics.executions_last_24h === 0 && (
            <p className="text-muted-foreground">• No recent activity</p>
          )}
          {metrics.executions_last_24h > 100 && (
            <p className="text-blue-700">• High usage volume (100+ executions/day)</p>
          )}
        </div>
      </div>
    </div>
  );
}
