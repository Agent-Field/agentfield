// @ts-nocheck
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { CompactWorkflowInputOutput } from './CompactWorkflowInputOutput';
import * as useMainNodeExecutionHook from '@/hooks/useMainNodeExecution';

const mockUseMainNodeExecution = vi.spyOn(useMainNodeExecutionHook, 'useMainNodeExecution');

// Mock child components
vi.mock('@/components/ui/card', () => ({
  Card: ({ children, className }) => <div className={`card ${className}`}>{children}</div>,
  CardHeader: ({ children }) => <div className="card-header">{children}</div>,
  CardTitle: ({ children }) => <h1 className="card-title">{children}</h1>,
  CardContent: ({ children }) => <div className="card-content">{children}</div>,
}));

vi.mock('@/components/ui/tabs', () => ({
  Tabs: ({ children, value, onValueChange }) => (
    <div data-testid="tabs" data-value={value}>
      {/* Attach a function to a child to simulate onValueChange */}
      {React.Children.map(children, child => {
        if (child.type.name === 'TabsList') {
          return React.cloneElement(child, { onValueChange, currentValue: value });
        }
        return child;
      })}
    </div>
  ),
  TabsList: ({ children, onValueChange, currentValue }) => (
    <div>
      {React.Children.map(children, child => {
        return React.cloneElement(child, { onValueChange, currentValue });
      })}
    </div>
  ),
  TabsTrigger: ({ children, value, disabled, onValueChange, currentValue }) => (
    <button
      data-testid={`tab-${value}`}
      disabled={disabled}
      onClick={() => onValueChange(value)}
      data-state={currentValue === value ? 'active' : 'inactive'}
    >
      {children}
    </button>
  ),
  TabsContent: ({ children, value }) => <div data-testid={`tab-content-${value}`}>{children}</div>,
}));

vi.mock('@/components/ui/UnifiedJsonViewer', () => ({
  UnifiedJsonViewer: ({ data }) => <pre data-testid="json-viewer">{JSON.stringify(data)}</pre>,
}));

vi.mock('@/components/execution/EnhancedModal', () => ({
  DataModal: ({ isOpen, title, data }) =>
    isOpen ? (
      <div data-testid="data-modal">
        <h2>{title}</h2>
        <pre>{JSON.stringify(data)}</pre>
      </div>
    ) : null,
}));

vi.mock('@/components/ui/button', () => ({
  Button: ({ children, onClick, 'aria-label': ariaLabel }) => (
    <button onClick={onClick} aria-label={ariaLabel}>{children}</button>
  ),
}));

vi.mock('@/components/ui/copy-button', () => ({
    CopyButton: ({ value }) => <button aria-label="Copy data" data-value={value} />,
}));

vi.mock('@/components/ui/segmented-control', () => ({
    SegmentedControl: ({ value }) => <div data-testid="segmented-control" data-value={value} />,
}));

vi.mock('@/components/ui/badge', () => ({
  Badge: ({ children }) => <span className="badge">{children}</span>,
}));

vi.mock('@/components/ui/icon-bridge', () => ({
  ArrowsOutSimple: () => <span data-testid="icon-arrows-out"></span>,
  DownloadSimple: () => <span data-testid="icon-download"></span>,
  UploadSimple: () => <span data-testid="icon-upload"></span>,
  Code: () => <span data-testid="icon-code"></span>,
  Eye: () => <span data-testid="icon-eye"></span>,
}));


const mockDagData = {
  root_workflow_id: 'wf-1',
  total_nodes: 1,
  max_depth: 1,
  dag: { execution_id: 'exec-1' },
};

const mockExecutionInput = {
  input_data: { message: 'hello' },
  input_size: 15,
};

const mockExecutionOutput = {
  output_data: { result: 'world' },
  output_size: 15,
};


describe('CompactWorkflowInputOutput', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  // ... (previous tests are fine)
  it('renders null when loading', () => {
    mockUseMainNodeExecution.mockReturnValue({ loading: true, error: null, execution: null, hasInputData: false, hasOutputData: false, isCompleted: false, isRunning: false });
    const { container } = render(<CompactWorkflowInputOutput dagData={mockDagData} />);
    expect(container.firstChild).toBeNull();
  });

  it('renders null when there is an error', () => {
    mockUseMainNodeExecution.mockReturnValue({ loading: false, error: 'Error', execution: null, hasInputData: false, hasOutputData: false, isCompleted: false, isRunning: false });
    const { container } = render(<CompactWorkflowInputOutput dagData={mockDagData} />);
    expect(container.firstChild).toBeNull();
  });

  it('renders with only input data', () => {
    mockUseMainNodeExecution.mockReturnValue({ loading: false, error: null, execution: mockExecutionInput, hasInputData: true, hasOutputData: false, isCompleted: false, isRunning: true });
    render(<CompactWorkflowInputOutput dagData={mockDagData} />);
    expect(screen.getByText('Workflow Input/Output')).toBeInTheDocument();
    expect(screen.getByTestId('tab-input')).not.toBeDisabled();
    expect(screen.getByTestId('tab-output')).toBeDisabled();
    expect(screen.getByTestId('tabs')).toHaveAttribute('data-value', 'input');
  });

  it('defaults to output tab when completed', () => {
    mockUseMainNodeExecution.mockReturnValue({ loading: false, error: null, execution: { ...mockExecutionInput, ...mockExecutionOutput }, hasInputData: true, hasOutputData: true, isCompleted: true, isRunning: false });
    render(<CompactWorkflowInputOutput dagData={mockDagData} />);
    expect(screen.getByTestId('tabs')).toHaveAttribute('data-value', 'output');
  });
  
  it('opens the data modal when view full button is clicked', () => {
    mockUseMainNodeExecution.mockReturnValue({
      loading: false,
      error: null,
      execution: mockExecutionInput,
      hasInputData: true,
      hasOutputData: false,
      isCompleted: false,
      isRunning: true,
    });


    
    render(<CompactWorkflowInputOutput dagData={mockDagData} />);

    expect(screen.queryByTestId('data-modal')).not.toBeInTheDocument();
    
    // The view full button is the one with the ArrowsOutSimple icon.
    // Let's find it by the icon within it.
    const viewFullButton = screen.getByTestId('icon-arrows-out').closest('button');
    expect(viewFullButton).toBeInTheDocument();

    fireEvent.click(viewFullButton);

    expect(screen.getByTestId('data-modal')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Input Data' })).toBeInTheDocument();
  });
});
