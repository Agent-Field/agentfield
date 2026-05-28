import * as React from "react"
import { cn } from "../../lib/utils"

export interface AutoExpandingTextareaProps
  extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  maxHeight?: number;
}

const AutoExpandingTextarea = React.forwardRef<HTMLTextAreaElement, AutoExpandingTextareaProps>(
  ({ className, maxHeight = 120, ...props }, ref) => {
    const textareaRef = React.useRef<HTMLTextAreaElement>(null);
    const combinedRef = React.useCallback(
      (node: HTMLTextAreaElement) => {
        textareaRef.current = node;
        if (typeof ref === 'function') {
          ref(node);
        } else if (ref) {
          ref.current = node;
        }
      },
      [ref]
    );

    const adjustHeight = React.useCallback(() => {
      const textarea = textareaRef.current;
      if (!textarea) return;

      // Reset height to auto to get the correct scrollHeight
      textarea.style.height = 'auto';

      // Calculate the new height
      const scrollHeight = textarea.scrollHeight;
      const newHeight = Math.min(scrollHeight, maxHeight);

      // Set the new height
      textarea.style.height = `${newHeight}px`;

      // Enable/disable scrolling based on content
      textarea.style.overflowY = scrollHeight > maxHeight ? 'auto' : 'hidden';
    }, [maxHeight]);

    // Adjust height when value changes
    React.useEffect(() => {
      adjustHeight();
    }, [props.value, props.defaultValue, adjustHeight]);

    // Adjust height on input
    const handleInput = React.useCallback((e: React.FormEvent<HTMLTextAreaElement>) => {
      adjustHeight();
      if (props.onInput) {
        props.onInput(e);
      }
    }, [adjustHeight, props.onInput]);

    // Handle change event
    const handleChange = React.useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
      adjustHeight();
      if (props.onChange) {
        props.onChange(e);
      }
    }, [adjustHeight, props.onChange]);

    return (
      <textarea
        className={cn(
          // Base styles matching Input component
          "flex min-h-[2.5rem] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background",
          // Placeholder and focus styles
          "placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
          // Disabled styles
          "disabled:cursor-not-allowed disabled:opacity-50",
          // Textarea specific styles
          "resize-none transition-all duration-200 ease-out",
          // Custom scrollbar for better UX
          "scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border hover:scrollbar-thumb-muted-foreground",
          className
        )}
        ref={combinedRef}
        onInput={handleInput}
        onChange={handleChange}
        style={{
          minHeight: '2.5rem', // 40px - matches Input height
          maxHeight: `${maxHeight}px`,
          lineHeight: '1.5',
          overflowY: 'hidden', // Initially hidden, will be set by adjustHeight
        }}
        {...props}
      />
    )
  }
)

AutoExpandingTextarea.displayName = "AutoExpandingTextarea"

export { AutoExpandingTextarea }
