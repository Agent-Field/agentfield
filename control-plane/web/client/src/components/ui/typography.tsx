import { cva } from "class-variance-authority";

export const typographyVariants = cva("", {
  variants: {
    variant: {
      display: "text-3xl font-bold tracking-tight",
      heading: "text-2xl font-semibold tracking-tight",
      subheading: "text-xl font-semibold",
      section: "text-base font-semibold",
      body: "text-sm",
      secondary: "text-sm text-muted-foreground",
      tertiary: "text-xs font-medium text-muted-foreground uppercase tracking-wider",
      disabled: "text-sm text-muted-foreground text-muted-foreground opacity-50",
    }
  },
  defaultVariants: {
    variant: "body",
  },
});
