import { forwardRef, type ElementType, type HTMLAttributes } from "react";
import { cn } from "@/lib/utils";

type Breakpoint = "base" | "sm" | "md" | "lg" | "xl" | "2xl";
type ColumnCount = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12;

type GridPreset =
  | "auto" // single column with responsive gap
  | "halves" // 1 → 2 columns
  | "thirds" // 1 → 2 → 3 columns
  | "quarters" // 1 → 2 → 3 → 4 columns
  | "dense" // 1 → 2 → 3 → 4 columns for compact dashboards
  | "stats" // 1 → 2 → 4 columns with tighter gap
  | "sidebar" // 1 → 2 column split favoring content
  | "detail"; // 1 → 2 columns for detail panels

type GapScale = "none" | "xs" | "sm" | "md" | "lg";

type ColumnsConfig = Partial<Record<Breakpoint, ColumnCount>>;

const GAP_CLASSES: Record<GapScale, string> = {
  none: "gap-0",
  xs: "gap-3",
  sm: "gap-4",
  md: "gap-6",
  lg: "gap-8",
};

const FLOW_CLASSES = {
  row: "",
  col: "grid-flow-col",
  dense: "grid-flow-dense",
} as const;

const ALIGN_CLASSES = {
  start: "items-start",
  center: "items-center",
  stretch: "",
} as const;

const COLUMN_CLASSES: Record<Breakpoint, Record<ColumnCount, string>> = {
  base: {
    1: "grid-cols-1",
    2: "grid-cols-2",
    3: "grid-cols-3",
    4: "grid-cols-4",
    5: "grid-cols-5",
    6: "grid-cols-6",
    7: "grid-cols-7",
    8: "grid-cols-8",
    9: "grid-cols-9",
    10: "grid-cols-10",
    11: "grid-cols-11",
    12: "grid-cols-12",
  },
  sm: {
    1: "sm:grid-cols-1",
    2: "sm:grid-cols-2",
    3: "sm:grid-cols-3",
    4: "sm:grid-cols-4",
    5: "sm:grid-cols-5",
    6: "sm:grid-cols-6",
    7: "sm:grid-cols-7",
    8: "sm:grid-cols-8",
    9: "sm:grid-cols-9",
    10: "sm:grid-cols-10",
    11: "sm:grid-cols-11",
    12: "sm:grid-cols-12",
  },
  md: {
    1: "md:grid-cols-1",
    2: "md:grid-cols-2",
    3: "md:grid-cols-3",
    4: "md:grid-cols-4",
    5: "md:grid-cols-5",
    6: "md:grid-cols-6",
    7: "md:grid-cols-7",
    8: "md:grid-cols-8",
    9: "md:grid-cols-9",
    10: "md:grid-cols-10",
    11: "md:grid-cols-11",
    12: "md:grid-cols-12",
  },
  lg: {
    1: "lg:grid-cols-1",
    2: "lg:grid-cols-2",
    3: "lg:grid-cols-3",
    4: "lg:grid-cols-4",
    5: "lg:grid-cols-5",
    6: "lg:grid-cols-6",
    7: "lg:grid-cols-7",
    8: "lg:grid-cols-8",
    9: "lg:grid-cols-9",
    10: "lg:grid-cols-10",
    11: "lg:grid-cols-11",
    12: "lg:grid-cols-12",
  },
  xl: {
    1: "xl:grid-cols-1",
    2: "xl:grid-cols-2",
    3: "xl:grid-cols-3",
    4: "xl:grid-cols-4",
    5: "xl:grid-cols-5",
    6: "xl:grid-cols-6",
    7: "xl:grid-cols-7",
    8: "xl:grid-cols-8",
    9: "xl:grid-cols-9",
    10: "xl:grid-cols-10",
    11: "xl:grid-cols-11",
    12: "xl:grid-cols-12",
  },
  "2xl": {
    1: "2xl:grid-cols-1",
    2: "2xl:grid-cols-2",
    3: "2xl:grid-cols-3",
    4: "2xl:grid-cols-4",
    5: "2xl:grid-cols-5",
    6: "2xl:grid-cols-6",
    7: "2xl:grid-cols-7",
    8: "2xl:grid-cols-8",
    9: "2xl:grid-cols-9",
    10: "2xl:grid-cols-10",
    11: "2xl:grid-cols-11",
    12: "2xl:grid-cols-12",
  },
};

const PRESET_COLUMNS: Record<GridPreset, ColumnsConfig> = {
  auto: { base: 1 },
  halves: { base: 1, md: 2 },
  thirds: { base: 1, md: 2, lg: 3 },
  quarters: { base: 1, sm: 2, lg: 3, xl: 4 },
  dense: { base: 1, sm: 2, md: 3, xl: 4 },
  stats: { base: 1, sm: 2, md: 3, lg: 4 },
  sidebar: { base: 1, md: 2, xl: 3 },
  detail: { base: 1, lg: 2, xl: 3 },
};

export interface ResponsiveGridProps extends HTMLAttributes<HTMLDivElement> {
  variant?: GridVariant;
  columns?: ColumnsConfig;
  preset?: GridPreset;
  gap?: GapScale;
  flow?: keyof typeof FLOW_CLASSES;
  align?: keyof typeof ALIGN_CLASSES;
}

type GridVariant =
  | "auto"
  | "stack"
  | "split"
  | "dashboard"
  | "metrics"
  | "detail";

const VARIANT_DEFAULTS: Record<
  GridVariant,
  {
    preset: GridPreset;
    gap: GapScale;
    align?: keyof typeof ALIGN_CLASSES;
  }
> = {
  auto: { preset: "auto", gap: "md", align: "stretch" },
  stack: { preset: "auto", gap: "lg", align: "stretch" },
  split: { preset: "sidebar", gap: "lg", align: "start" },
  dashboard: { preset: "quarters", gap: "md", align: "start" },
  metrics: { preset: "stats", gap: "sm", align: "center" },
  detail: { preset: "detail", gap: "md", align: "start" },
};

const SPAN_CLASSES: Record<Breakpoint, Record<ColumnCount, string>> = {
  base: {
    1: "col-span-1",
    2: "col-span-2",
    3: "col-span-3",
    4: "col-span-4",
    5: "col-span-5",
    6: "col-span-6",
    7: "col-span-7",
    8: "col-span-8",
    9: "col-span-9",
    10: "col-span-10",
    11: "col-span-11",
    12: "col-span-12",
  },
  sm: {
    1: "sm:col-span-1",
    2: "sm:col-span-2",
    3: "sm:col-span-3",
    4: "sm:col-span-4",
    5: "sm:col-span-5",
    6: "sm:col-span-6",
    7: "sm:col-span-7",
    8: "sm:col-span-8",
    9: "sm:col-span-9",
    10: "sm:col-span-10",
    11: "sm:col-span-11",
    12: "sm:col-span-12",
  },
  md: {
    1: "md:col-span-1",
    2: "md:col-span-2",
    3: "md:col-span-3",
    4: "md:col-span-4",
    5: "md:col-span-5",
    6: "md:col-span-6",
    7: "md:col-span-7",
    8: "md:col-span-8",
    9: "md:col-span-9",
    10: "md:col-span-10",
    11: "md:col-span-11",
    12: "md:col-span-12",
  },
  lg: {
    1: "lg:col-span-1",
    2: "lg:col-span-2",
    3: "lg:col-span-3",
    4: "lg:col-span-4",
    5: "lg:col-span-5",
    6: "lg:col-span-6",
    7: "lg:col-span-7",
    8: "lg:col-span-8",
    9: "lg:col-span-9",
    10: "lg:col-span-10",
    11: "lg:col-span-11",
    12: "lg:col-span-12",
  },
  xl: {
    1: "xl:col-span-1",
    2: "xl:col-span-2",
    3: "xl:col-span-3",
    4: "xl:col-span-4",
    5: "xl:col-span-5",
    6: "xl:col-span-6",
    7: "xl:col-span-7",
    8: "xl:col-span-8",
    9: "xl:col-span-9",
    10: "xl:col-span-10",
    11: "xl:col-span-11",
    12: "xl:col-span-12",
  },
  "2xl": {
    1: "2xl:col-span-1",
    2: "2xl:col-span-2",
    3: "2xl:col-span-3",
    4: "2xl:col-span-4",
    5: "2xl:col-span-5",
    6: "2xl:col-span-6",
    7: "2xl:col-span-7",
    8: "2xl:col-span-8",
    9: "2xl:col-span-9",
    10: "2xl:col-span-10",
    11: "2xl:col-span-11",
    12: "2xl:col-span-12",
  },
};

export interface ResponsiveGridItemProps
  extends HTMLAttributes<HTMLElement> {
  as?: ElementType;
  span?: Partial<Record<Breakpoint, ColumnCount>>;
}

const BaseResponsiveGrid = forwardRef<HTMLDivElement, ResponsiveGridProps>(
  function BaseResponsiveGrid(
    {
      variant,
      columns,
      preset,
      gap,
      flow = "row",
      align,
      className,
      children,
      ...props
    },
    ref
  ) {
    const fallback = variant
      ? VARIANT_DEFAULTS[variant]
      : VARIANT_DEFAULTS.auto;

    const resolvedPreset = preset ?? fallback.preset;
    const resolvedGap = gap ?? fallback.gap;
    const resolvedAlign = align ?? fallback.align ?? "stretch";

    const config =
      columns ?? PRESET_COLUMNS[resolvedPreset] ?? PRESET_COLUMNS.auto;

    const columnClasses = Object.entries(config).map(([breakpoint, count]) => {
      const bp = breakpoint as Breakpoint;
      const columnCount = count as ColumnCount | undefined;
      if (!columnCount) return null;
      return COLUMN_CLASSES[bp][columnCount];
    });

    return (
      <div
        ref={ref}
        className={cn(
          "grid",
          GAP_CLASSES[resolvedGap],
          FLOW_CLASSES[flow],
          ALIGN_CLASSES[resolvedAlign],
          columnClasses,
          className
        )}
        {...props}
      >
        {children}
      </div>
    );
  }
);

const ResponsiveGridItem = forwardRef<HTMLElement, ResponsiveGridItemProps>(
  function ResponsiveGridItem(
    { as: Component = "div", span, className, children, ...props },
    ref
  ) {
    const spanClasses = span
      ? Object.entries(span).map(([breakpoint, count]) => {
          const bp = breakpoint as Breakpoint;
          const columnCount = count as ColumnCount | undefined;
          if (!columnCount) return null;
          return SPAN_CLASSES[bp][columnCount];
        })
      : [];

    return (
      <Component
        ref={ref}
        className={cn(spanClasses, className)}
        {...props}
      >
        {children}
      </Component>
    );
  }
);

export const ResponsiveGrid = Object.assign(BaseResponsiveGrid, {
  Item: ResponsiveGridItem,
});

export type {
  Breakpoint,
  ColumnCount,
  GridVariant,
  GridPreset,
  GapScale,
  ColumnsConfig,
};
