import { Badge } from "./badge";

/**
 * A tag group rendered inside a tooltip.
 * - `label` (optional) renders an uppercase section header (e.g. "From", "Granted").
 * - `tags` is the list of tag strings to render as -style chips.
 *   An empty array renders a muted "any" placeholder.
 */
export interface TooltipTagGroup {
  label?: string;
  tags: string[];
}

export interface TooltipTagListProps {
  groups: TooltipTagGroup[];
}

/**
 * Renders one or more groups of tags as styled chips inside a tooltip.
 *
 * Uses the `tooltip` badge variant (semi-transparent  on dark bg).
 * Place inside `<TooltipContent>` for consistent styling across the app.
 */
export function TooltipTagList({ groups }: TooltipTagListProps) {
  return (
    <div className="space-y-1.5">
      {groups.map((group, i) => (
        <div key={i}>
          {group.label && (
            <div className="text-micro uppercase tracking-wider text-primary-foreground/60 mb-0.5">
              {group.label}
            </div>
          )}
          <div className="flex flex-wrap gap-1">
            {group.tags.length > 0 ? (
              group.tags.map((tag) => (
                <Badge key={tag} variant="tooltip" size="sm" showIcon={false}>
                  {tag}
                </Badge>
              ))
            ) : (
              <span className="text-primary-foreground/40 italic text-micro">
                any
              </span>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
