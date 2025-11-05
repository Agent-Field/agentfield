# MCP UI System Documentation

## Overview

The MCP (Model Context Protocol) UI System is a comprehensive frontend implementation for managing and monitoring MCP servers, tools, and performance metrics. This system provides a complete interface for interacting with MCP infrastructure in both user and developer modes.

## Architecture

### Core Components

The MCP UI system is built with a modular architecture consisting of:

1. **Components** - Reusable UI components for MCP functionality
2. **Hooks** - Custom React hooks for state management and API integration
3. **Services** - API service layer for backend communication
4. **Utilities** - Helper functions for data processing and formatting
5. **Types** - TypeScript definitions for type safety

### Component Hierarchy

```
MCP UI System
├── Components (agentfield/web/client/src/components/mcp/)
│   ├── MCPHealthIndicator - Status indicators and health displays
│   ├── MCPServerCard - Individual server information cards
│   ├── MCPServerList - List of all MCP servers
│   ├── MCPToolExplorer - Tool discovery and exploration
│   ├── MCPToolTester - Interactive tool testing interface
│   └── MCPServerControls - Bulk server management controls
├── Hooks (agentfield/web/client/src/hooks/)
│   ├── useMCPHealth - Health monitoring and real-time updates
│   ├── useMCPServers - Server management operations
│   ├── useMCPTools - Tool discovery and execution
│   ├── useMCPMetrics - Performance metrics monitoring
│   └── useSSE - Server-Sent Events for real-time updates
├── Utilities (agentfield/web/client/src/utils/)
│   └── mcpUtils - Formatting, validation, and helper functions
└── Integration (agentfield/web/client/src/mcp/)
    └── index.ts - Centralized exports and integration patterns
```

## Features

### 1. Real-Time Health Monitoring

- **Live Status Updates**: Real-time server status via Server-Sent Events
- **Health Indicators**: Visual status indicators with color coding
- **Mode-Aware Display**: Different information levels for user vs developer modes
- **Automatic Refresh**: Configurable auto-refresh intervals

### 2. Server Management

- **Individual Operations**: Start, stop, restart individual servers
- **Bulk Operations**: Manage multiple servers simultaneously
- **Status Tracking**: Track operation progress and results
- **Error Handling**: Comprehensive error reporting and retry logic

### 3. Tool Discovery and Testing

- **Dynamic Tool Loading**: Discover available tools from running servers
- **Interactive Testing**: Test tools with custom parameters
- **Parameter Validation**: Schema-based parameter validation
- **Execution History**: Track tool execution results and performance

### 4. Performance Monitoring

- **Real-Time Metrics**: Live performance data collection
- **Historical Data**: Trend analysis and historical metrics
- **Performance Alerts**: Configurable performance thresholds
- **Resource Monitoring**: CPU, memory, and response time tracking

### 5. Accessibility and UX

- **Screen Reader Support**: Full ARIA compliance and screen reader support
- **Keyboard Navigation**: Complete keyboard accessibility
- **Loading States**: Skeleton screens for better perceived performance
- **Error Boundaries**: Graceful error handling and recovery

## Usage Examples

### Basic Health Monitoring

```typescript
import { useMCPHealth } from '@/mcp';

function MyComponent({ nodeId }: { nodeId: string }) {
  const { summary, servers, loading, error } = useMCPHealth(nodeId, {
    enableRealTime: true,
    refreshInterval: 30000
  });

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <div>
      <h2>Health: {summary?.overall_health}%</h2>
      <p>Running: {summary?.running_servers}/{summary?.total_servers}</p>
    </div>
  );
}
```

### Server Management

```typescript
import { useMCPServers } from '@/mcp';

function ServerManager({ nodeId }: { nodeId: string }) {
  const { startServer, stopServer, restartServer, loading } = useMCPServers(nodeId);

  const handleRestart = async (serverId: string) => {
    await restartServer(serverId, {
      onComplete: (result) => console.log('Restart completed:', result),
      onError: (error) => console.error('Restart failed:', error)
    });
  };

  return (
    <button 
      onClick={() => handleRestart('my-server')}
      disabled={loading}
    >
      Restart Server
    </button>
  );
}
```

### Tool Testing

```typescript
import { useMCPTools } from '@/mcp';

function ToolTester({ nodeId, serverId }: { nodeId: string; serverId: string }) {
  const { tools, executeTool, executions } = useMCPTools(nodeId, serverId);

  const handleExecute = async (toolName: string) => {
    const result = await executeTool(toolName, { param1: 'value1' }, {
      validateParams: true,
      timeoutMs: 30000,
      onComplete: (result) => console.log('Tool executed:', result)
    });
  };

  return (
    <div>
      {tools.map(tool => (
        <button key={tool.name} onClick={() => handleExecute(tool.name)}>
          Execute {tool.name}
        </button>
      ))}
    </div>
  );
}
```

### Complete Integration

```typescript
import { useMCPIntegration } from '@/mcp';

function MCPDashboard({ nodeId }: { nodeId: string }) {
  const mcp = useMCPIntegration(nodeId, {
    enableRealTime: true,
    enableMetrics: true,
    enableTools: true
  });

  return (
    <div>
      <h1>MCP Dashboard</h1>
      
      {/* Health Overview */}
      <div>
        Health: {mcp.health.isHealthy ? 'Good' : 'Issues Detected'}
        Servers: {mcp.health.runningCount}/{mcp.health.totalCount}
      </div>

      {/* Quick Actions */}
      <div>
        <button onClick={mcp.servers.startAll}>Start All</button>
        <button onClick={mcp.servers.stopAll}>Stop All</button>
        <button onClick={mcp.refreshAll}>Refresh</button>
      </div>

      {/* Metrics */}
      {mcp.metrics && (
        <div>
          Performance: {mcp.metrics.getPerformanceSummary()?.avgResponseTime}ms
        </div>
      )}
    </div>
  );
}
```

## Configuration

### Environment Variables

```bash
# API Base URL
VITE_API_BASE_URL=/api/ui/v1

# Development mode
NODE_ENV=development
```

### Hook Options

Most hooks accept configuration options:

```typescript
interface MCPHealthOptions {
  refreshInterval?: number;     // Auto-refresh interval (ms)
  enableRealTime?: boolean;     // Enable SSE updates
  cacheTtl?: number;           // Cache TTL (ms)
  sortByPriority?: boolean;    // Sort servers by priority
  onHealthChange?: (health: MCPSummaryForUI) => void;
  onServerStatusChange?: (server: MCPServerHealthForUI) => void;
}
```

## Error Handling

The system includes comprehensive error handling:

### Error Boundaries

```typescript
import { MCPErrorBoundary } from '@/components/ErrorBoundary';

<MCPErrorBoundary nodeId={nodeId} componentName="Server List">
  <MCPServerList servers={servers} nodeId={nodeId} />
</MCPErrorBoundary>
```

### API Error Handling

- **Automatic Retry**: Configurable retry logic for transient failures
- **Error Classification**: Distinguish between retryable and permanent errors
- **User-Friendly Messages**: Mode-aware error message formatting

### Accessibility Features

- **Screen Reader Support**: Full ARIA compliance
- **Keyboard Navigation**: Tab order and keyboard shortcuts
- **Live Regions**: Status announcements for dynamic content
- **Focus Management**: Proper focus handling for modals and interactions

## Performance Optimizations

### Memoization

Components use React.memo and useMemo for optimal re-rendering:

```typescript
const MemoizedServerCard = React.memo(MCPServerCard);
```

### Loading States

Skeleton screens provide better perceived performance:

```typescript
import { LoadingWrapper, MCPServerListSkeleton } from '@/components/LoadingSkeleton';

<LoadingWrapper 
  loading={loading} 
  skeleton={<MCPServerListSkeleton />}
>
  <MCPServerList servers={servers} />
</LoadingWrapper>
```

### Caching

- **Response Caching**: Configurable TTL for API responses
- **State Persistence**: Maintain state across component unmounts
- **Optimistic Updates**: Immediate UI updates with rollback on failure

## Testing

### Unit Tests

```typescript
import { renderHook } from '@testing-library/react-hooks';
import { useMCPHealth } from '@/mcp';

test('should fetch health data', async () => {
  const { result, waitForNextUpdate } = renderHook(() => 
    useMCPHealth('test-node')
  );
  
  await waitForNextUpdate();
  
  expect(result.current.summary).toBeDefined();
});
```

### Integration Tests

```typescript
import { render, screen } from '@testing-library/react';
import { MCPServerList } from '@/mcp';

test('should display server list', () => {
  render(<MCPServerList servers={mockServers} nodeId="test" />);
  
  expect(screen.getByText('Server 1')).toBeInTheDocument();
});
```

## Troubleshooting

### Common Issues

1. **SSE Connection Failures**
   - Check network connectivity
   - Verify API endpoint availability
   - Check browser SSE support

2. **Performance Issues**
   - Reduce refresh intervals
   - Disable real-time updates if not needed
   - Check for memory leaks in long-running components

3. **Type Errors**
   - Ensure all MCP types are properly imported
   - Check API response format matches type definitions

### Debug Mode

Enable debug logging in development:

```typescript
// In development, detailed logs are available
if (process.env.NODE_ENV === 'development') {
  console.log('MCP Debug Info:', debugData);
}
```

## Contributing

### Adding New Components

1. Create component in `agentfield/web/client/src/components/mcp/`
2. Add proper TypeScript types
3. Include accessibility features
4. Add error boundary support
5. Export from `agentfield/web/client/src/components/mcp/index.ts`

### Adding New Hooks

1. Create hook in `agentfield/web/client/src/hooks/`
2. Follow existing patterns for state management
3. Include proper cleanup and error handling
4. Add TypeScript documentation
5. Export from `agentfield/web/client/src/mcp/index.ts`

## API Reference

See the complete API reference in the individual component and hook files. Each export includes comprehensive JSDoc documentation with usage examples and parameter descriptions.