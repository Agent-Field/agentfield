import { Close } from "@/components/ui/icon-bridge";
import { TAG_COLORS, type TagColor } from "../../types/notes";
import { Button } from "../ui/button";

interface TagBadgeProps {
  tag: string;
  color?: TagColor;
  size?: 'sm' | 'md';
  removable?: boolean;
  onRemove?: (tag: string) => void;
  onClick?: (tag: string) => void;
  className?: string;
}

// Generate consistent color for a tag based on its hash
function getTagColor(tag: string): TagColor {
  let hash = 0;
  for (let i = 0; i < tag.length; i++) {
    const char = tag.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32-bit integer
  }
  const index = Math.abs(hash) % TAG_COLORS.length;
  return TAG_COLORS[index];
}

export function TagBadge({
  tag,
  color,
  size = 'sm',
  removable = false,
  onRemove,
  onClick,
  className = '',
}: TagBadgeProps) {
  const tagColor = color || getTagColor(tag);
  const sizeClasses = size === 'sm'
    ? 'px-2 py-1 text-xs'
    : 'px-3 py-1.5 text-sm';

  const baseClasses = `
    inline-flex items-center gap-1 rounded-md border font-medium
    transition-all duration-fast ease-smooth
    ${sizeClasses}
    ${tagColor}
    ${onClick ? 'cursor-pointer transition-colors cursor-pointer hover:bg-accent' : ''}
    ${className}
  `.trim();

  const handleClick = () => {
    if (onClick) {
      onClick(tag);
    }
  };

  const handleRemove = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (onRemove) {
      onRemove(tag);
    }
  };

  return (
    <span
      className={baseClasses}
      onClick={handleClick}
      role={onClick ? "button" : undefined}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={onClick ? (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          handleClick();
        }
      } : undefined}
    >
      <span className="truncate max-w-[120px]" title={tag}>
        {tag}
      </span>
      {removable && onRemove && (
        <Button
          variant="ghost"
          size="sm"
          className="h-4 w-4 p-0 hover:bg-accent rounded-full"
          onClick={handleRemove}
          aria-label={`Remove ${tag} tag`}
        >
          <Close size={12} />
        </Button>
      )}
    </span>
  );
}

// Export the color generation function for use in other components
export { getTagColor };
