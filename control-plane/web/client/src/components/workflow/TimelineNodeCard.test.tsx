// @ts-nocheck
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { TimelineNodeCard, TimelineNodeCardSkeleton } from './TimelineNodeCard';
import type { ExecutionNote } from '../../types/notes';

// Mock child components to isolate the TimelineNodeCard
vi.mock('@/components/ui/button', () => ({
  Button: ({ children, onClick, ...props }) => (
    <button onClick={onClick} {...props}>
      {children}
    </button>
  ),
}));

vi.mock('@/components/ui/skeleton', () => ({
  Skeleton: ({ className }) => <div data-testid="skeleton" className={className} />,
}));

vi.mock('react-markdown', () => ({
  default: ({ children }) => <div data-testid="markdown-content">{children}</div>,
}));

const mockNode = {
  workflow_id: 'wf-1',
  execution_id: 'exec-1',
  agent_node_id: 'agent-node-1',
  reasoner_id: 'test-reasoner',
  status: 'succeeded',
  started_at: '2023-10-27T10:00:00Z',
  completed_at: '2023-10-27T10:05:00Z',
  duration_ms: 300000,
  workflow_depth: 0,
  children: [],
  agent_name: 'Test Agent',
  task_name: 'Test Task',
};

const mockNotes: ExecutionNote[] = [
  {
    message: 'This is the first note.',
    tags: ['info', 'debug'],
    timestamp: '2023-10-27T10:01:00Z',
  },
  {
    message: 'This is a second, much longer note designed to test the "Show more" and "Show less" functionality. It needs to be over two hundred characters to ensure that the truncation logic is properly triggered. We will add some more filler text here to make sure we cross that threshold with plenty of room to spare. This should be sufficient.',
    tags: ['warning'],
    timestamp: '2023-10-27T10:02:00Z',
  },
];

describe('TimelineNodeCard', () => {
  it('renders basic node information without notes', () => {
    render(<TimelineNodeCard node={mockNode} notes={[]} />);
    expect(screen.getByText('Test Task')).toBeInTheDocument();
    expect(screen.getByText('Test Agent')).toBeInTheDocument();
    expect(screen.getByText('10:00 AM')).toBeInTheDocument();
    // Should not be clickable or show expansion indicators
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
  });

  it('renders a node with a preview of notes when collapsed', () => {
    render(<TimelineNodeCard node={mockNode} notes={mockNotes} />);
    expect(screen.getByText('Test Task')).toBeInTheDocument();
    // Check for note preview
    expect(screen.getByText('This is the first note. (+1)')).toBeInTheDocument();
    // Check for compact tags
    expect(screen.getByText('#info')).toBeInTheDocument();
    expect(screen.getByText('#debug')).toBeInTheDocument();
  });

  it('expands to show full notes on card click', () => {
    render(<TimelineNodeCard node={mockNode} notes={mockNotes} />);
    // Initially, full notes are not visible
    expect(screen.queryByTestId('markdown-content')).not.toBeInTheDocument();
    
    // Click the card to expand
    fireEvent.click(screen.getByText('Test Task').closest('div.group'));

    // Now full notes should be visible
    expect(screen.getAllByTestId('markdown-content').length).toBe(2);
    expect(screen.getByText('This is the first note.')).toBeInTheDocument();
  });

  it('calls onTagClick when a tag is clicked', () => {
    const onTagClick = vi.fn();
    render(<TimelineNodeCard node={mockNode} notes={mockNotes} onTagClick={onTagClick} />);
    
    // Click a compact tag
    fireEvent.click(screen.getByText('#info'));
    expect(onTagClick).toHaveBeenCalledWith('info');
    
    // Expand the card
    fireEvent.click(screen.getByText('Test Task').closest('div.group'));
    
    // Click a tag in the full note view
    fireEvent.click(screen.getAllByText('#warning')[0]);
    expect(onTagClick).toHaveBeenCalledWith('warning');
  });

  it('calls onClick when the card is clicked', () => {
    const onClick = vi.fn();
    render(<TimelineNodeCard node={mockNode} notes={mockNotes} onClick={onClick} />);
    fireEvent.click(screen.getByText('Test Task').closest('div.group'));
    expect(onClick).toHaveBeenCalled();
  });

  it('renders different status dots based on node status', () => {
    const { rerender, container } = render(
      <TimelineNodeCard node={{ ...mockNode, status: 'succeeded' }} notes={[]} />
    );
    let statusDot = container.querySelector('.w-2.h-2.rounded-full');
    expect(statusDot).toHaveClass('bg-status-success');

    rerender(<TimelineNodeCard node={{ ...mockNode, status: 'running' }} notes={[]} />);
    statusDot = container.querySelector('.w-2.h-2.rounded-full');
    expect(statusDot).toHaveClass('bg-status-info', 'animate-pulse');

    rerender(<TimelineNodeCard node={{ ...mockNode, status: 'failed' }} notes={[]} />);
    statusDot = container.querySelector('.w-2.h-2.rounded-full');
    expect(statusDot).toHaveClass('bg-status-error');

    rerender(<TimelineNodeCard node={{ ...mockNode, status: 'cancelled' }} notes={[]} />);
    statusDot = container.querySelector('.w-2.h-2.rounded-full');
    expect(statusDot).toHaveClass('bg-text-quaternary');
  });

  it('toggles long notes with "Show more" and "Show less"', () => {
    render(<TimelineNodeCard node={mockNode} notes={mockNotes} forceExpanded />);
    
    // The long note should be truncated initially
    const longNoteContent = mockNotes[1].message;
    const truncatedContent = longNoteContent.substring(0, 200) + "...";
    expect(screen.getByText(truncatedContent)).toBeInTheDocument();

    // Click "Show more"
    const showMoreButton = screen.getByText('Show more');
    fireEvent.click(showMoreButton);

    // Full content should be visible, and "Show less" button should appear
    expect(screen.getByText(longNoteContent)).toBeInTheDocument();
    const showLessButton = screen.getByText('Show less');
    expect(showLessButton).toBeInTheDocument();

    // Click "Show less"
    fireEvent.click(showLessButton);
    
    // Content should be truncated again
    expect(screen.getByText(truncatedContent)).toBeInTheDocument();
    expect(screen.getByText('Show more')).toBeInTheDocument();
  });
});

describe('TimelineNodeCardSkeleton', () => {
  it('renders skeleton placeholders', () => {
    render(<TimelineNodeCardSkeleton />);
    const skeletons = screen.getAllByTestId('skeleton');
    expect(skeletons.length).toBe(5);
    expect(skeletons[0].className).toContain('h-2 w-2'); // status dot
    expect(skeletons[1].className).toContain('h-4 w-20'); // reasoner name
    expect(skeletons[2].className).toContain('h-1 w-1'); // separator
    expect(skeletons[3].className).toContain('h-3 w-16'); // agent name
    expect(skeletons[4].className).toContain('h-3 w-12'); // time
  });
});
