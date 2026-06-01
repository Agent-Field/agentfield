/**
 * Formats a number to a human-readable string with abbreviations
 *
 * Examples:
 * - 999 -> "999"
 * - 1000 -> "1K"
 * - 1500 -> "1.5K"
 * - 1000000 -> "1M"
 * - 2500000 -> "2.5M"
 * - 1000000000 -> "1B"
 *
 * @param num - The number to format
 * @param decimals - Number of decimal places to show (default: 1)
 * @returns Formatted string with appropriate suffix
 */
export function formatNumber(num: number, decimals: number = 1): string {
  if (num < 1000) {
    return num.toString();
  }

  const units = [
    { value: 1e9, suffix: 'B' },
    { value: 1e6, suffix: 'M' },
    { value: 1e3, suffix: 'K' },
  ];

  for (const unit of units) {
    if (num >= unit.value) {
      const formatted = num / unit.value;
      // If the result is a whole number, don't show decimals
      if (formatted % 1 === 0) {
        return formatted.toFixed(0) + unit.suffix;
      }
      // Otherwise show the specified number of decimals
      return formatted.toFixed(decimals) + unit.suffix;
    }
  }

  return num.toString();
}

/**
 * Formats a number with proper separators for readability
 *
 * Examples:
 * - 1000 -> "1,000"
 * - 1000000 -> "1,000,000"
 *
 * @param num - The number to format
 * @returns Formatted string with thousands separators
 */
export function formatNumberWithCommas(num: number): string {
  return num.toLocaleString();
}
