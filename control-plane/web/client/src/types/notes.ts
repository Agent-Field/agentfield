export interface ExecutionNote {
  message: string;
  tags: string[];
  timestamp: string;
}

export interface NotesResponse {
  execution_id: string;
  notes: ExecutionNote[];
  total: number;
}

export interface AddNoteRequest {
  message: string;
  tags?: string[];
}

export interface AddNoteResponse {
  success: boolean;
  note: ExecutionNote;
  message: string;
}

export interface NotesFilters {
  tags?: string[];
}

export interface TagColorMap {
  [tag: string]: string;
}

// Predefined tag colors using theme variables for dark mode compatibility
export const TAG_COLORS = [
  'bg-status-info/10 text-status-info border-status-info/30',
  'bg-status-success/10 text-status-success border-status-success/30',
  'bg-status-warning/10 text-status-warning border-status-warning/30',
  'bg-muted text-muted-foreground border-border',
  'bg-chart-1/10 text-chart-1 border-chart-1/20',
  'bg-chart-2/10 text-chart-2 border-chart-2/20',
  'bg-chart-3/10 text-chart-3 border-chart-3/20',
  'bg-chart-4/10 text-chart-4 border-chart-4/20',
  'bg-chart-5/10 text-chart-5 border-chart-5/20',
  'bg-chart-6/10 text-chart-6 border-chart-6/20',
] as const;

export type TagColor = typeof TAG_COLORS[number];

export interface NotesState {
  notes: ExecutionNote[];
  loading: boolean;
  error: string | null;
  filters: NotesFilters;
  sortOrder: 'asc' | 'desc';
  tagColorMap: TagColorMap;
}
