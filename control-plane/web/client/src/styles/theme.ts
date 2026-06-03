export const typography = {
  // Headings
  'heading-xs': 'text-xs font-semibold tracking-wider uppercase',
  'heading-sm': 'text-sm font-semibold',
  'heading-base': 'text-base font-semibold',

  // Navigation
  'nav-item': 'text-sm font-medium',
  'nav-description': 'text-xs',

  // Utility Text
  'helper-text': 'text-xs text-muted-foreground'
} as const;

export const spacing = {
  // Navigation
  'nav-section': 'space-y-4',
  'nav-items': 'space-y-0.5',
  'section-padding': 'py-4',

  // Component Spacing
  'stack-sm': 'space-y-2',
  'stack-md': 'space-y-4',
  'stack-lg': 'space-y-6',

  // Padding
  'padding-sm': 'p-2',
  'padding-md': 'p-4',
  'padding-lg': 'p-6'
} as const;

export type TypographyToken = keyof typeof typography;
export type SpacingToken = keyof typeof spacing;
