// @ts-nocheck
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { ExecutionScatterPlot } from './ExecutionScatterPlot';
import type { WorkflowTimelineNode } from '../../types/workflows';

// Mock 'recharts' library
vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }) => <div data-testid="responsive-container">{children}</div>,
  ScatterChart: ({ children }) => <div data-testid="scatter-chart">{children}</div>,
  CartesianGrid: () => <div data-testid="cartesian-grid" />,
  XAxis: ({ domain }) => <div data-testid="x-axis" data-domain={JSON.stringify(domain)} />,
  YAxis: () => <div data-testid="y-axis" />,
  ZAxis: () => <div data-testid="z-axis" />,
  Tooltip: () => <div data-testid="tooltip" />,
  ReferenceLine: ({ y }) => <div data-testid="reference-line" data-y={y} />,
  Scatter: ({ data, onClick }) => (
    <div data-testid="scatter" data-points={data.length}>
      {data.map((p, i) => (
        <button key={i} aria-label={`point-${p.id}`} onClick={() => onClick({ payload: p })} />
      ))}
    </div>
  ),
  Cell: () => <div />,
}));

// Mock 'lucide-react' icons
vi.mock('lucide-react', () => ({
  ZoomIn: () => <div data-testid="zoom-in-icon" />,
  ZoomOut: () => <div data-testid="zoom-out-icon" />,
  RotateCcw: () => <div data-testid="reset-zoom-icon" />,
}));

const mockNodes: WorkflowTimelineNode[] = [
  {
    workflow_id: 'wf-1',
    execution_id: 'exec-1',
    agent_node_id: 'agent-1',
    reasoner_id: 'reasoner-1',
    status: 'succeeded',
    started_at: '2023-10-27T10:00:00Z',
    duration_ms: 500,
    agent_name: 'Agent A',
    workflow_depth: 0,
    children: [],
  },
  {
    workflow_id: 'wf-1',
    execution_id: 'exec-2',
    agent_node_id: 'agent-2',
    reasoner_id: 'reasoner-2',
    status: 'failed',
    started_at: '2023-10-27T10:01:00Z',
    duration_ms: 1200,
    agent_name: 'Agent B',
    workflow_depth: 0,
    children: [],
  },
];

describe('ExecutionScatterPlot', () => {
  it('renders empty state when no nodes are provided', () => {
    render(<ExecutionScatterPlot timedNodes={[]} />);
    expect(screen.getByText('No execution data to display')).toBeInTheDocument();
    expect(screen.queryByTestId('scatter-chart')).not.toBeInTheDocument();
  });

  it('renders the scatter plot with data', () => {
    render(<ExecutionScatterPlot timedNodes={mockNodes} />);
    expect(screen.getByTestId('scatter-chart')).toBeInTheDocument();
    expect(screen.getByTestId('scatter')).toHaveAttribute('data-points', '2');
    expect(screen.getByText(/Scatter plot shows distribution of 2 executions./)).toBeInTheDocument();
    // Check for legend
    expect(screen.getByText('Success')).toBeInTheDocument();
    expect(screen.getByText('Failed')).toBeInTheDocument();
  });

  it('calls onNodeClick when a point is clicked', () => {
    const onNodeClick = vi.fn();
    render(<ExecutionScatterPlot timedNodes={mockNodes} onNodeClick={onNodeClick} />);
    
    const pointButton = screen.getByLabelText('point-exec-2');
    fireEvent.click(pointButton);
    
    expect(onNodeClick).toHaveBeenCalledWith('exec-2');
  });

  it('handles zoom functionality', () => {
    render(<ExecutionScatterPlot timedNodes={mockNodes} />);

    const zoomInButton = screen.getByTestId('zoom-in-icon').closest('button');
    const zoomOutButton = screen.getByTestId('zoom-out-icon').closest('button');
    const resetButton = screen.getByTestId('reset-zoom-icon').closest('button');

    // Initially, zoom out and reset are disabled
    expect(zoomOutButton).toBeDisabled();
    expect(resetButton).toBeDisabled();
    
    const initialDomain = screen.getByTestId('x-axis').getAttribute('data-domain');

    // Zoom in
    fireEvent.click(zoomInButton);
    const zoomedInDomain = screen.getByTestId('x-axis').getAttribute('data-domain');
    expect(zoomedInDomain).not.toEqual(initialDomain);
    
    // Now zoom out and reset should be enabled
    expect(zoomOutButton).not.toBeDisabled();
    expect(resetButton).not.toBeDisabled();

    // Zoom out
    fireEvent.click(zoomOutButton);
    const zoomedOutDomain = screen.getByTestId('x-axis').getAttribute('data-domain');
    // This might go back to the initial domain or be slightly different due to float math
    expect(zoomedOutDomain).not.toEqual(zoomedInDomain);
    
    // Reset zoom
    fireEvent.click(resetButton);
    const resetDomain = screen.getByTestId('x-axis').getAttribute('data-domain');
    expect(resetDomain).toEqual(initialDomain);
  });
});
