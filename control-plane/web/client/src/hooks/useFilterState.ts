import { useState, useEffect, useCallback, useMemo } from 'react';
import type { FilterTag } from '../types/filters';
import type { ExecutionFilters, ExecutionGrouping } from '../types/executions';
import {
  convertTagsToApiFormat,
  convertApiFormatToTags,
  serializeFiltersToUrl,
  deserializeFiltersFromUrl,
} from '../utils/filterUtils';

interface UseFilterStateOptions {
  initialFilters?: Partial<ExecutionFilters>;
  initialGrouping?: ExecutionGrouping;
  syncWithUrl?: boolean;
}

export interface UseFilterStateReturn {
  tags: FilterTag[];
  filters: Partial<ExecutionFilters>;
  grouping: ExecutionGrouping;
  hasFilters: boolean;
  updateTags: (tags: FilterTag[]) => void;
  addTag: (tag: FilterTag) => void;
  removeTag: (tagId: string) => void;
  clearTags: () => void;
}


export function useFilterState({
  initialFilters = {},
  initialGrouping = {
    group_by: 'none',
    sort_by: 'time',
    sort_order: 'desc',
  },
  syncWithUrl = true,
}: UseFilterStateOptions = {}):UseFilterStateReturn {
  // Initialize tags from URL or initial values
  const [tags, setTags] = useState<FilterTag[]>(() => {
    if (syncWithUrl && typeof window !== 'undefined') {
      const urlParams = new URLSearchParams(window.location.search);
      const urlTags = deserializeFiltersFromUrl(urlParams);
      if (urlTags.length > 0) {
        return urlTags;
      }
    }
    return convertApiFormatToTags(initialFilters, initialGrouping);
  });

  // Convert tags to API format - memoized to prevent infinite re-renders
  const { filters, grouping } = useMemo(() => {
    return convertTagsToApiFormat(tags);
  }, [tags]);

  // Update URL when tags change
  useEffect(() => {
    if (!syncWithUrl || typeof window === 'undefined') return;

    const urlString = serializeFiltersToUrl(tags);
    const newUrl = urlString
      ? `${window.location.pathname}?${urlString}`
      : window.location.pathname;

    // Only update if URL actually changed
    if (newUrl !== window.location.pathname + window.location.search) {
      window.history.replaceState({}, '', newUrl);
    }
  }, [tags, syncWithUrl]);

  // Handle browser back/forward
  useEffect(() => {
    if (!syncWithUrl || typeof window === 'undefined') return;

    const handlePopState = () => {
      const urlParams = new URLSearchParams(window.location.search);
      const urlTags = deserializeFiltersFromUrl(urlParams);
      setTags(urlTags);
    };

    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, [syncWithUrl]);

  const updateTags = useCallback((newTags: FilterTag[]) => {
    setTags(newTags);
  }, []);

  const addTag = useCallback((tag: FilterTag) => {
    setTags(prev => [...prev, tag]);
  }, []);

  const removeTag = useCallback((tagId: string) => {
    setTags(prev => prev.filter(tag => tag.id !== tagId));
  }, []);

  const clearTags = useCallback(() => {
    setTags([]);
  }, []);

  const hasFilters = tags.length > 0;

  return {
    tags,
    filters,
    grouping,
    hasFilters,
    updateTags,
    addTag,
    removeTag,
    clearTags,
  };
}
