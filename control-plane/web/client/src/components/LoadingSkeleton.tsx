import React from 'react';
import { Skeleton } from '@/components/ui/skeleton';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { cn } from '@/lib/utils';

interface LoadingSkeletonProps {
  className?: string;
  variant?: 'card' | 'list' | 'table' | 'metrics' | 'chart';
  count?: number;
  animated?: boolean;
}

/**
 * Reusable loading skeleton component for better perceived performance
 * Provides different variants for different UI patterns
 */
export function LoadingSkeleton({
  className,
  variant = 'card',
  count = 1,
  animated = true
}: LoadingSkeletonProps) {
  const skeletonClass = cn(
    animated && 'animate-pulse',
    className
  );

  const renderSkeleton = () => {
    switch (variant) {
      case 'card':
        return (
          <Card className={skeletonClass}>
            <CardHeader>
              <Skeleton className="h-6 w-3/4" />
              <Skeleton className="h-4 w-1/2" />
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-5/6" />
                <Skeleton className="h-4 w-4/6" />
              </div>
            </CardContent>
          </Card>
        );

      case 'list':
        return (
          <div className={cn('space-y-3', skeletonClass)}>
            {Array.from({ length: count }).map((_, i) => (
              <div key={i} className="flex items-center space-x-4 p-4 border rounded-lg">
                <Skeleton className="h-10 w-10 rounded-full" />
                <div className="space-y-2 flex-1">
                  <Skeleton className="h-4 w-1/4" />
                  <Skeleton className="h-3 w-1/2" />
                </div>
                <Skeleton className="h-8 w-20" />
              </div>
            ))}
          </div>
        );

      case 'table':
        return (
          <div className={cn('space-y-2', skeletonClass)}>
            {/* Header */}
            <div className="flex space-x-4 p-4 border-b">
              <Skeleton className="h-4 w-1/4" />
              <Skeleton className="h-4 w-1/4" />
              <Skeleton className="h-4 w-1/4" />
              <Skeleton className="h-4 w-1/4" />
            </div>
            {/* Rows */}
            {Array.from({ length: count }).map((_, i) => (
              <div key={i} className="flex space-x-4 p-4">
                <Skeleton className="h-4 w-1/4" />
                <Skeleton className="h-4 w-1/4" />
                <Skeleton className="h-4 w-1/4" />
                <Skeleton className="h-4 w-1/4" />
              </div>
            ))}
          </div>
        );

      case 'metrics':
        return (
          <div className={cn('grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4', skeletonClass)}>
            {Array.from({ length: 4 }).map((_, i) => (
              <Card key={i}>
                <CardContent className="p-6">
                  <div className="flex items-center justify-between">
                    <div className="space-y-2">
                      <Skeleton className="h-4 w-16" />
                      <Skeleton className="h-8 w-12" />
                    </div>
                    <Skeleton className="h-8 w-8 rounded" />
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        );

      case 'chart':
        return (
          <Card className={skeletonClass}>
            <CardHeader>
              <Skeleton className="h-6 w-1/3" />
              <Skeleton className="h-4 w-1/2" />
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex justify-between items-end h-32">
                  {Array.from({ length: 7 }).map((_, i) => (
                    <Skeleton
                      key={i}
                      className="w-8"
                      style={{ height: `${Math.random() * 80 + 20}%` }}
                    />
                  ))}
                </div>
                <div className="flex justify-between">
                  {Array.from({ length: 7 }).map((_, i) => (
                    <Skeleton key={i} className="h-3 w-8" />
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        );

      default:
        return <Skeleton className={cn('h-20 w-full', skeletonClass)} />;
    }
  };

  return count > 1 && variant !== 'list' ? (
    <div className="space-y-4">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i}>{renderSkeleton()}</div>
      ))}
    </div>
  ) : (
    renderSkeleton()
  );
}

/**
 * Conditional loading wrapper that shows skeleton while loading
 */
export function LoadingWrapper({
  loading,
  skeleton,
  children,
  fallback
}: {
  loading: boolean;
  skeleton?: React.ReactNode;
  children: React.ReactNode;
  fallback?: React.ReactNode;
}) {
  if (loading) {
    return skeleton || fallback || <LoadingSkeleton />;
  }

  return <>{children}</>;
}

/**
 * Hook for managing loading states with skeleton fallbacks
 */
export function useLoadingState(initialLoading = false) {
  const [loading, setLoading] = React.useState(initialLoading);

  const withLoading = React.useCallback(async <T,>(
    asyncFn: () => Promise<T>,
    showSkeleton = true
  ): Promise<T> => {
    if (showSkeleton) setLoading(true);
    try {
      const result = await asyncFn();
      return result;
    } finally {
      if (showSkeleton) setLoading(false);
    }
  }, []);

  return {
    loading,
    setLoading,
    withLoading
  };
}
