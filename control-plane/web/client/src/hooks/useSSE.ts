import { useEffect, useRef, useState, useCallback } from 'react';
import { getGlobalApiKey } from '../services/api';

/**
 * Configuration options for SSE connection
 */
interface SSEOptions {
  /** Whether to automatically reconnect on connection loss */
  autoReconnect?: boolean;
  /** Maximum number of reconnection attempts */
  maxReconnectAttempts?: number;
  /** Base delay between reconnection attempts (ms) */
  reconnectDelayMs?: number;
  /** Whether to use exponential backoff for reconnection delays */
  exponentialBackoff?: boolean;
  /** Custom event types to listen for */
  eventTypes?: string[];
  /** Callback for connection state changes */
  onConnectionChange?: (connected: boolean) => void;
  /** Callback for errors */
  onError?: (error: Event) => void;
}

/**
 * SSE connection state
 */
interface SSEState {
  /** Whether the connection is currently active */
  connected: boolean;
  /** Whether currently attempting to reconnect */
  reconnecting: boolean;
  /** Current reconnection attempt number */
  reconnectAttempt: number;
  /** Last error that occurred */
  lastError: Event | null;
}

/**
 * Event data structure for typed events
 */
interface SSEEvent<T = any> {
  type: string;
  data: T;
  timestamp: Date;
  id?: string;
}

/**
 * Custom hook for managing Server-Sent Events connections with automatic reconnection
 * and event filtering capabilities.
 *
 * @param url - The SSE endpoint URL
 * @param options - Configuration options for the SSE connection
 * @returns Object containing connection state, events, and control functions
 */
export function useSSE<T = any>(
  url: string | null,
  options: SSEOptions = {}
) {
  const {
    autoReconnect = true,
    maxReconnectAttempts = 5,
    reconnectDelayMs = 1000,
    exponentialBackoff = true,
    eventTypes = [],
    onConnectionChange,
    onError
  } = options;

  const [state, setState] = useState<SSEState>({
    connected: false,
    reconnecting: false,
    reconnectAttempt: 0,
    lastError: null
  });

  const [events, setEvents] = useState<SSEEvent<T>[]>([]);
  const [latestEvent, setLatestEvent] = useState<SSEEvent<T> | null>(null);

  const eventSourceRef = useRef<EventSource | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const mountedRef = useRef(true);

  /**
   * Clear any pending reconnection timeout
   */
  const clearReconnectTimeout = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
  }, []);

  /**
   * Close the current EventSource connection
   */
  const closeConnection = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    clearReconnectTimeout();
  }, [clearReconnectTimeout]);

  /**
   * Attempt to reconnect with exponential backoff
   */
  const attemptReconnect = useCallback(() => {
    if (!mountedRef.current || !autoReconnect || !url) return;

    setState(prev => {
      if (prev.reconnectAttempt >= maxReconnectAttempts) {
        return { ...prev, reconnecting: false };
      }

      const delay = exponentialBackoff
        ? reconnectDelayMs * Math.pow(2, prev.reconnectAttempt)
        : reconnectDelayMs;

      reconnectTimeoutRef.current = setTimeout(() => {
        if (mountedRef.current) {
          connect();
        }
      }, delay);

      return {
        ...prev,
        reconnecting: true,
        reconnectAttempt: prev.reconnectAttempt + 1
      };
    });
  }, [autoReconnect, url, maxReconnectAttempts, reconnectDelayMs, exponentialBackoff]);

  /**
   * Handle incoming SSE events
   */
  const handleEvent = useCallback((event: MessageEvent, eventType: string = 'message') => {
    if (!mountedRef.current) return;

    try {
      // Add defensive checks for event data
      if (!event.data) {
        console.warn('🚨 SSE: Received event with no data');
        return;
      }

      const data = JSON.parse(event.data);

      // Enhanced validation to prevent Object.entries() errors downstream
      if (!data || typeof data !== 'object' || data === null) {
        console.warn('🚨 SSE: Parsed data is not a valid object:', data);
        return;
      }

      // Extract the actual event type from the data if it's a NodeEvent structure
      // Backend sends NodeEvent with { type, node_id, status, timestamp, data }
      let actualEventType = eventType;
      if (data.type && typeof data.type === 'string') {
        actualEventType = data.type;
        console.log('🔍 SSE: Extracted event type from data:', actualEventType);
      }

      const sseEvent: SSEEvent<T> = {
        type: actualEventType,
        data,
        timestamp: new Date(),
        id: event.lastEventId || undefined
      };

      console.log('🔄 SSE: Processing event:', { type: actualEventType, hasData: !!data });
      setLatestEvent(sseEvent);
      setEvents(prev => [...prev.slice(-99), sseEvent]); // Keep last 100 events
    } catch (error) {
      console.warn('🚨 SSE: Failed to parse event data:', error, 'Raw data:', event.data);
    }
  }, []);

  /**
   * Establish SSE connection
   */
  const connect = useCallback(() => {
    if (!url || !mountedRef.current) return;

    console.log('🔄 SSE: connect() called for URL:', url);
    closeConnection();

    try {
      let finalUrl = url;
      const apiKey = getGlobalApiKey();
      if (apiKey) {
        const separator = url.includes('?') ? '&' : '?';
        finalUrl = `${url}${separator}api_key=${encodeURIComponent(apiKey)}`;
      }

      const eventSource = new EventSource(finalUrl);
      eventSourceRef.current = eventSource;
      console.log('🔌 SSE: EventSource created for URL:', finalUrl);

      eventSource.onopen = () => {
        if (!mountedRef.current) return;

        console.log('✅ SSE: Connection opened for URL:', url);
        setState(prev => ({
          ...prev,
          connected: true,
          reconnecting: false,
          reconnectAttempt: 0,
          lastError: null
        }));

        onConnectionChange?.(true);
      };

      eventSource.onerror = (error) => {
        if (!mountedRef.current) return;

        console.log('❌ SSE: Connection error for URL:', url, error);
        setState(prev => ({
          ...prev,
          connected: false,
          lastError: error
        }));

        onConnectionChange?.(false);
        onError?.(error);

        // Only attempt reconnect if the connection was previously established
        // or if this is not a permanent failure
        if (eventSource.readyState === EventSource.CLOSED) {
          console.log('🔄 SSE: Attempting reconnect for URL:', url);
          attemptReconnect();
        }
      };

      eventSource.onmessage = (event) => handleEvent(event, 'message');

      // Register custom event listeners
      eventTypes.forEach(eventType => {
        eventSource.addEventListener(eventType, (event) =>
          handleEvent(event as MessageEvent, eventType)
        );
      });

    } catch (error) {
      console.error('Failed to create EventSource:', error);
      setState(prev => ({
        ...prev,
        connected: false,
        lastError: error as Event
      }));
    }
  }, [url, eventTypes, handleEvent, closeConnection, attemptReconnect, onConnectionChange, onError]);


  /**
   * Manually reconnect the SSE connection
   */
  const reconnect = useCallback(() => {
    setState(prev => ({ ...prev, reconnectAttempt: 0 }));
    connect();
  }, [connect]);

  /**
   * Clear all stored events
   */
  const clearEvents = useCallback(() => {
    setEvents([]);
    setLatestEvent(null);
  }, []);

  /**
   * Filter events by type
   */
  const getEventsByType = useCallback((type: string): SSEEvent<T>[] => {
    return events.filter(event => event.type === type);
  }, [events]);

  // Initialize connection when URL changes
  useEffect(() => {
    console.log('🚀 SSE: Main useEffect triggered for URL:', url, 'Connected:', state.connected);
    if (url && !state.connected && !eventSourceRef.current) {
      console.log('🔗 SSE: Calling connect() for URL:', url);
      connect();
    } else if (!url) {
      console.log('🔌 SSE: No URL, closing connection');
      closeConnection();
      setState(prev => ({ ...prev, connected: false }));
    } else {
      console.log('🔄 SSE: Skipping connect - already connected or connecting');
    }

    return () => {
      console.log('🧹 SSE: Cleanup called for URL:', url);
      closeConnection();
    };
  }, [url]); // Only depend on URL changes, not the functions

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      mountedRef.current = false;
      closeConnection();
    };
  }, []); // No dependencies needed for cleanup

  return {
    // Connection state
    connected: state.connected,
    reconnecting: state.reconnecting,
    reconnectAttempt: state.reconnectAttempt,
    lastError: state.lastError,

    // Events
    events,
    latestEvent,

    // Control functions
    reconnect,
    disconnect: closeConnection,
    clearEvents,
    getEventsByType,

    // Utility
    hasEvents: events.length > 0,
    eventCount: events.length
  };
}

/**
 * Specialized hook for MCP health events
 */
export function useMCPHealthSSE(nodeId: string | null) {
  const url = nodeId ? `/api/ui/v1/nodes/${nodeId}/mcp/events` : null;

  return useSSE(url, {
    eventTypes: ['server_status_change', 'tool_execution', 'health_update', 'error'],
    autoReconnect: true,
    maxReconnectAttempts: 3,
    reconnectDelayMs: 2000,
    exponentialBackoff: true
  });
}

/**
 * Specialized hook for agent node events including status changes
 */
export function useNodeEventsSSE() {
  const url = '/api/ui/v1/nodes/events';

  return useSSE(url, {
    eventTypes: [
      'node_registered',
      'node_online',
      'node_offline',
      'node_status_updated',
      'node_health_changed',
      'node_removed',
      'mcp_health_changed',
      // New unified status events
      'node_unified_status_changed',
      'node_state_transition',
      'node_status_refreshed',
      'bulk_status_update'
    ],
    autoReconnect: true,
    maxReconnectAttempts: 5,
    reconnectDelayMs: 1000,
    exponentialBackoff: true
  });
}

/**
 * Specialized hook for unified status events
 */
export function useUnifiedStatusSSE() {
  const url = '/api/ui/v1/nodes/events';

  return useSSE(url, {
    eventTypes: [
      'node_unified_status_changed',
      'node_state_transition',
      'node_status_refreshed',
      'bulk_status_update'
    ],
    autoReconnect: true,
    maxReconnectAttempts: 5,
    reconnectDelayMs: 1000,
    exponentialBackoff: true
  });
}

/**
 * Specialized hook for unified status events for a specific node
 */
export function useNodeUnifiedStatusSSE(_nodeId: string | null) {
  // Note: Currently uses the same endpoint as useUnifiedStatusSSE since the backend
  // streams all node events and filtering happens on the client side
  // The nodeId parameter is reserved for future client-side filtering implementation
  const url = '/api/ui/v1/nodes/events';

  return useSSE(url, {
    eventTypes: [
      'node_unified_status_changed',
      'node_state_transition',
      'node_status_refreshed'
    ],
    autoReconnect: true,
    maxReconnectAttempts: 3,
    reconnectDelayMs: 2000,
    exponentialBackoff: true
  });
}
