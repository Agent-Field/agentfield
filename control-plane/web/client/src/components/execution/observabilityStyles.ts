export const observabilityStyles = {
  card: "border-border/80 shadow-sm",
  cardContent: "space-y-3 px-4 py-4 sm:px-6 sm:py-5",
  toolbarGrid: "grid gap-3 xl:grid-cols-[minmax(0,1fr)_auto_auto_auto_auto] xl:items-end",
  compactToolbarGrid:
    "grid gap-3 xl:grid-cols-[max-content_max-content_minmax(18rem,1fr)] xl:items-end",
  toolbarActions: "flex flex-wrap items-center gap-2 xl:justify-end",
  filterGroup: "min-w-0 space-y-1.5",
  filterLabelRow: "flex items-center gap-1",
  filterLabel: "text-xs font-medium text-muted-foreground",
  surface: "rounded-xl border border-border/70 bg-muted/20 p-3",
  summaryBar:
    "mb-3 flex flex-col gap-2 border-b border-border/60 pb-3 text-xs text-muted-foreground sm:flex-row sm:items-center sm:justify-between",
  summaryFilters: "flex flex-wrap items-center gap-2",
  summaryStats: "flex flex-wrap items-center gap-2",
  helperText: "text-[11px] text-muted-foreground",
  scrollArea: "pr-3",
  structuredList: "overflow-hidden rounded-xl border border-border/70 bg-background/95",
  structuredRow: "border-b border-border/60 last:border-b-0",
  structuredRowInner: "px-3 py-2",
  structuredTimestamp:
    "min-w-[5.5rem] pt-0.5 font-mono text-[11px] text-muted-foreground",
  structuredMessageRow: "flex min-w-0 flex-wrap items-center gap-2",
  structuredMessage: "min-w-0 flex-1 truncate text-sm text-foreground",
  structuredMeta:
    "mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-muted-foreground",
  detailTrigger:
    "group inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[10px] text-muted-foreground transition-colors hover:bg-muted/50 hover:text-foreground",
  detailPanel: "mt-2 overflow-hidden rounded-lg border border-border/60 bg-muted/10",
  emptyState:
    "flex min-h-[16rem] items-center justify-center rounded-lg border border-dashed border-border/70 bg-background/80 px-6 text-center text-sm text-muted-foreground",
  loadingState:
    "flex h-full min-h-[16rem] items-center justify-center text-sm text-muted-foreground",
  processHeader: "space-y-4 p-4 pb-3 sm:p-5 sm:pb-3",
  processActions: "flex w-full min-w-0 flex-col gap-2 lg:w-auto lg:shrink-0 lg:max-w-full",
  processContent: "space-y-3 px-4 pb-4 pt-0 sm:px-6 sm:pb-5",
  processScroll: "h-[min(420px,50vh)] w-full rounded-md border border-border/80 bg-muted/20",
  processScrollInner: "p-1.5 text-[10px] leading-tight sm:text-[11px]",
  processRow:
    "grid grid-cols-1 items-start gap-x-2 gap-y-1 border-b border-border/30 py-1 last:border-b-0 sm:grid-cols-[9rem_min-content_minmax(0,1fr)] sm:gap-y-0 sm:py-0.5",
  processStructuredRow:
    "grid grid-cols-1 items-start gap-x-2 gap-y-1 px-2 py-1 sm:grid-cols-[8.5rem_min-content_minmax(0,1fr)_auto] sm:gap-y-0",
  processTimestamp:
    "flex min-w-0 max-w-full flex-nowrap items-baseline gap-x-1.5 truncate tabular-nums text-muted-foreground",
  processMeta: "mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[9px] text-muted-foreground",
} as const;
