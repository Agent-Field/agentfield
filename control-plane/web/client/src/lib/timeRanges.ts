const TIME_RANGE_SEQUENCE = ["1h", "24h", "7d", "30d", "all"] as const;

export type TimeRangeValue = (typeof TIME_RANGE_SEQUENCE)[number];

export function getNextTimeRange(current: string): TimeRangeValue | undefined {
  if (!current) {
    return TIME_RANGE_SEQUENCE[0];
  }

  const normalized = current as TimeRangeValue;
  const index = TIME_RANGE_SEQUENCE.indexOf(normalized);

  if (index === -1) {
    return "all";
  }

  if (index >= TIME_RANGE_SEQUENCE.length - 1) {
    return undefined;
  }

  return TIME_RANGE_SEQUENCE[index + 1];
}

export const TIME_RANGE_OPTIONS: TimeRangeValue[] = [...TIME_RANGE_SEQUENCE];
