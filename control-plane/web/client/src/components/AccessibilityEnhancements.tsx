import React, { useEffect, useRef } from 'react';
import { cn } from '@/lib/utils';

/**
 * Screen reader only text component
 */
export function ScreenReaderOnly({
  children,
  className
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <span
      className={cn(
        'sr-only absolute left-[-10000px] top-auto w-[1px] h-[1px] overflow-hidden',
        className
      )}
    >
      {children}
    </span>
  );
}

/**
 * Skip link component for keyboard navigation
 */
export function SkipLink({
  href,
  children
}: {
  href: string;
  children: React.ReactNode;
}) {
  return (
    <a
      href={href}
      className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:px-4 focus:py-2 focus:bg-primary focus:text-primary-foreground focus:rounded-md focus:shadow-lg"
    >
      {children}
    </a>
  );
}

/**
 * Live region for announcing dynamic content changes
 */
export function LiveRegion({
  children,
  politeness = 'polite',
  atomic = false,
  className
}: {
  children: React.ReactNode;
  politeness?: 'off' | 'polite' | 'assertive';
  atomic?: boolean;
  className?: string;
}) {
  return (
    <div
      aria-live={politeness}
      aria-atomic={atomic}
      className={cn('sr-only', className)}
    >
      {children}
    </div>
  );
}

/**
 * Focus trap component for modal dialogs
 */
export function FocusTrap({
  children,
  active = true,
  restoreFocus = true
}: {
  children: React.ReactNode;
  active?: boolean;
  restoreFocus?: boolean;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const previousActiveElement = useRef<Element | null>(null);

  useEffect(() => {
    if (!active) return;

    // Store the previously focused element
    previousActiveElement.current = document.activeElement;

    const container = containerRef.current;
    if (!container) return;

    // Get all focusable elements
    const getFocusableElements = () => {
      return container.querySelectorAll(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      );
    };

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return;

      const focusableElements = getFocusableElements();
      const firstElement = focusableElements[0] as HTMLElement;
      const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement;

      if (e.shiftKey) {
        // Shift + Tab
        if (document.activeElement === firstElement) {
          e.preventDefault();
          lastElement?.focus();
        }
      } else {
        // Tab
        if (document.activeElement === lastElement) {
          e.preventDefault();
          firstElement?.focus();
        }
      }
    };

    // Focus the first focusable element
    const focusableElements = getFocusableElements();
    if (focusableElements.length > 0) {
      (focusableElements[0] as HTMLElement).focus();
    }

    document.addEventListener('keydown', handleKeyDown);

    return () => {
      document.removeEventListener('keydown', handleKeyDown);

      // Restore focus to the previously active element
      if (restoreFocus && previousActiveElement.current) {
        (previousActiveElement.current as HTMLElement).focus?.();
      }
    };
  }, [active, restoreFocus]);

  if (!active) {
    return <>{children}</>;
  }

  return (
    <div ref={containerRef} className="focus-trap">
      {children}
    </div>
  );
}

/**
 * Keyboard navigation helper component
 */
export function KeyboardNavigable({
  children,
  onEnter,
  onSpace,
  onEscape,
  onArrowUp,
  onArrowDown,
  onArrowLeft,
  onArrowRight,
  className
}: {
  children: React.ReactNode;
  onEnter?: () => void;
  onSpace?: () => void;
  onEscape?: () => void;
  onArrowUp?: () => void;
  onArrowDown?: () => void;
  onArrowLeft?: () => void;
  onArrowRight?: () => void;
  className?: string;
}) {
  const handleKeyDown = (e: React.KeyboardEvent) => {
    switch (e.key) {
      case 'Enter':
        e.preventDefault();
        onEnter?.();
        break;
      case ' ':
        e.preventDefault();
        onSpace?.();
        break;
      case 'Escape':
        e.preventDefault();
        onEscape?.();
        break;
      case 'ArrowUp':
        e.preventDefault();
        onArrowUp?.();
        break;
      case 'ArrowDown':
        e.preventDefault();
        onArrowDown?.();
        break;
      case 'ArrowLeft':
        e.preventDefault();
        onArrowLeft?.();
        break;
      case 'ArrowRight':
        e.preventDefault();
        onArrowRight?.();
        break;
    }
  };

  return (
    <div
      onKeyDown={handleKeyDown}
      className={className}
      tabIndex={0}
    >
      {children}
    </div>
  );
}

/**
 * Status announcer for dynamic status changes
 */
export function StatusAnnouncer({
  status,
  delay = 1000
}: {
  status: string;
  delay?: number;
}) {
  const [announcement, setAnnouncement] = React.useState('');

  useEffect(() => {
    const timer = setTimeout(() => {
      setAnnouncement(status);
    }, delay);

    return () => clearTimeout(timer);
  }, [status, delay]);

  return (
    <LiveRegion politeness="polite">
      {announcement}
    </LiveRegion>
  );
}

/**
 * Progress announcer for long-running operations
 */
export function ProgressAnnouncer({
  progress,
  total,
  label,
  announceInterval = 25
}: {
  progress: number;
  total: number;
  label?: string;
  announceInterval?: number;
}) {
  const [lastAnnounced, setLastAnnounced] = React.useState(0);
  const percentage = Math.round((progress / total) * 100);

  useEffect(() => {
    if (percentage - lastAnnounced >= announceInterval || percentage === 100) {
      setLastAnnounced(percentage);
    }
  }, [percentage, lastAnnounced, announceInterval]);

  const shouldAnnounce = percentage - lastAnnounced >= announceInterval || percentage === 100;

  return (
    <LiveRegion politeness="polite">
      {shouldAnnounce && (
        `${label ? `${label}: ` : ''}${percentage}% complete`
      )}
    </LiveRegion>
  );
}

/**
 * Error announcer for form validation and errors
 */
export function ErrorAnnouncer({
  error,
  clearAfter = 5000
}: {
  error: string | null;
  clearAfter?: number;
}) {
  const [currentError, setCurrentError] = React.useState<string | null>(null);

  useEffect(() => {
    if (error) {
      setCurrentError(error);

      if (clearAfter > 0) {
        const timer = setTimeout(() => {
          setCurrentError(null);
        }, clearAfter);

        return () => clearTimeout(timer);
      }
    }
  }, [error, clearAfter]);

  return (
    <LiveRegion politeness="assertive">
      {currentError}
    </LiveRegion>
  );
}


/**
 * Hook for managing focus and announcements
 */
export function useAccessibility() {
  const announceStatus = React.useCallback((message: string) => {
    // Create a temporary live region for announcements
    const liveRegion = document.createElement('div');
    liveRegion.setAttribute('aria-live', 'polite');
    liveRegion.setAttribute('aria-atomic', 'true');
    liveRegion.className = 'sr-only';
    liveRegion.textContent = message;

    document.body.appendChild(liveRegion);

    // Remove after announcement
    setTimeout(() => {
      document.body.removeChild(liveRegion);
    }, 1000);
  }, []);

  const announceError = React.useCallback((message: string) => {
    const liveRegion = document.createElement('div');
    liveRegion.setAttribute('aria-live', 'assertive');
    liveRegion.setAttribute('aria-atomic', 'true');
    liveRegion.className = 'sr-only';
    liveRegion.textContent = `Error: ${message}`;

    document.body.appendChild(liveRegion);

    setTimeout(() => {
      document.body.removeChild(liveRegion);
    }, 3000);
  }, []);

  const focusElement = React.useCallback((selector: string) => {
    const element = document.querySelector(selector) as HTMLElement;
    if (element) {
      element.focus();
      element.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, []);

  return {
    announceStatus,
    announceError,
    focusElement
  };
}
