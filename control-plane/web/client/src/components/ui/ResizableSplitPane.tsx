import React, { useState, useCallback, useRef, useEffect } from 'react';
import { cn } from '../../lib/utils';
import { ChevronLeft, ChevronRight } from '@/components/ui/icon-bridge';

interface ResizableSplitPaneProps {
  children: [React.ReactNode, React.ReactNode];
  defaultSizePercent?: number;
  minSizePercent?: number;
  maxSizePercent?: number;
  orientation?: 'horizontal' | 'vertical';
  className?: string;
  leftPanelClassName?: string;
  rightPanelClassName?: string;
  resizerClassName?: string;
  collapsible?: boolean;
  collapsedSize?: number;
  onSizeChange?: (sizePercent: number) => void;
}

export function ResizableSplitPane({
  children,
  defaultSizePercent = 30,
  minSizePercent = 15,
  maxSizePercent = 70,
  orientation = 'horizontal',
  className,
  leftPanelClassName,
  rightPanelClassName,
  resizerClassName,
  collapsible = false,
  collapsedSize = 48,
  onSizeChange,
}: ResizableSplitPaneProps) {
  const [leftSize, setLeftSize] = useState(defaultSizePercent);
  const [isResizing, setIsResizing] = useState(false);
  const [isCollapsed, setIsCollapsed] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const isHorizontal = orientation === 'horizontal';

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
  }, []);

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isResizing || !containerRef.current) return;

      const container = containerRef.current;
      const containerRect = container.getBoundingClientRect();

      let newSize: number;
      if (isHorizontal) {
        const containerWidth = containerRect.width;
        const mouseX = e.clientX - containerRect.left;
        newSize = (mouseX / containerWidth) * 100;
      } else {
        const containerHeight = containerRect.height;
        const mouseY = e.clientY - containerRect.top;
        newSize = (mouseY / containerHeight) * 100;
      }

      // Constrain size within bounds
      newSize = Math.max(minSizePercent, Math.min(maxSizePercent, newSize));

      setLeftSize(newSize);
      onSizeChange?.(newSize);
    },
    [isResizing, isHorizontal, minSizePercent, maxSizePercent, onSizeChange]
  );

  const handleMouseUp = useCallback(() => {
    setIsResizing(false);
  }, []);

  useEffect(() => {
    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = isHorizontal ? 'col-resize' : 'row-resize';
      document.body.style.userSelect = 'none';

      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
      };
    }
  }, [isResizing, handleMouseMove, handleMouseUp, isHorizontal]);

  const toggleCollapse = useCallback(() => {
    setIsCollapsed(prev => !prev);
  }, []);

  const actualLeftSize = isCollapsed ? collapsedSize : leftSize;

  return (
    <div
      ref={containerRef}
      className={cn(
        'flex h-full w-full min-h-0 min-w-0 overflow-hidden',
        isHorizontal ? 'flex-row' : 'flex-col',
        className
      )}
    >
      {/* Left/Top Panel */}
      <div
        className={cn(
          'relative transition-all duration-200 ease-out flex-shrink-0 overflow-hidden min-h-0 min-w-0 box-border',
          leftPanelClassName
        )}
        style={{
          [isHorizontal ? 'width' : 'height']: `${actualLeftSize}%`,
          [isHorizontal ? 'minWidth' : 'minHeight']: isCollapsed ? `${collapsedSize}px` : `${minSizePercent}%`,
        }}
      >
        <div className="h-full w-full min-h-0 overflow-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border">
          {children[0]}
        </div>
      </div>

      {/* Resizer */}
      <div
        className={cn(
          'group relative z-10 flex-shrink-0 select-none',
          isHorizontal ? 'w-3 cursor-col-resize' : 'h-3 cursor-row-resize',
          resizerClassName
        )}
        role="separator"
        aria-orientation={isHorizontal ? 'vertical' : 'horizontal'}
        aria-valuenow={Math.round(leftSize)}
        aria-valuemin={minSizePercent}
        aria-valuemax={maxSizePercent}
        onMouseDown={handleMouseDown}
      >
        <span
          className={cn(
            'pointer-events-none absolute inset-0 rounded-full transition-colors duration-150',
            'bg-transparent group-hover:bg-border-secondary/10',
            isResizing && 'bg-border-secondary/20'
          )}
        />
        <span
          className={cn(
            'pointer-events-none absolute rounded-full transition-colors duration-150',
            isHorizontal
              ? 'left-1/2 top-1/2 h-8 w-px -translate-x-1/2 -translate-y-1/2'
              : 'left-1/2 top-1/2 w-8 h-px -translate-x-1/2 -translate-y-1/2',
            'bg-border-tertiary group-hover:bg-border-secondary',
            isResizing && 'bg-border'
          )}
        />

        {/* Collapse Button */}
        {collapsible && (
          <button
            onClick={toggleCollapse}
            className={cn(
              'absolute z-20 flex h-5 w-5 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border border-border bg-background text-muted-foreground transition-all duration-150',
              'opacity-0 shadow-none hover:border-border/80 hover:text-foreground hover:shadow-sm group-hover:opacity-100',
              isHorizontal ? 'left-1/2 top-1/2' : 'left-1/2 top-1/2'
            )}
            title={isCollapsed ? 'Expand panel' : 'Collapse panel'}
          >
            {isHorizontal ? (
              isCollapsed ? <ChevronRight className="w-3 h-3" /> : <ChevronLeft className="w-3 h-3" />
            ) : (
              isCollapsed ? <ChevronRight className="w-3 h-3 rotate-90" /> : <ChevronLeft className="w-3 h-3 rotate-90" />
            )}
          </button>
        )}
      </div>

      {/* Right/Bottom Panel */}
      <div
        className={cn(
          'relative transition-all duration-200 ease-out flex-1 overflow-hidden min-h-0 min-w-0 box-border',
          rightPanelClassName
        )}
        style={{
          [isHorizontal ? 'minWidth' : 'minHeight']: `${100 - maxSizePercent}%`,
        }}
      >
        <div className="h-full w-full min-h-0 overflow-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border">
          {children[1]}
        </div>
      </div>
    </div>
  );
}

// Hook for responsive behavior
export function useResponsiveSplitPane(breakpoint: number = 768) {
  const [isSmallScreen, setIsSmallScreen] = useState(false);

  useEffect(() => {
    const checkScreenSize = () => {
      setIsSmallScreen(window.innerWidth < breakpoint);
    };

    checkScreenSize();
    window.addEventListener('resize', checkScreenSize);
    return () => window.removeEventListener('resize', checkScreenSize);
  }, [breakpoint]);

  return { isSmallScreen };
}
