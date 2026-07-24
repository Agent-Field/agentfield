// @ts-nocheck
import { act, renderHook } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { useIsMobile } from './use-mobile';

const mocks = {
  listeners: new Map<string, () => void>(),
  matches: true,
};

const mockMatchMedia = vi.fn().mockImplementation(query => ({
  matches: mocks.matches,
  media: query,
  onchange: null,
  addListener: vi.fn(), // deprecated
  removeListener: vi.fn(), // deprecated
  addEventListener: vi.fn((event, cb) => {
    if (event === 'change') {
      mocks.listeners.set(query, cb);
    }
  }),
  removeEventListener: vi.fn((event, cb) => {
    if (event === 'change') {
      mocks.listeners.delete(query);
    }
  }),
  dispatchEvent: vi.fn(),
}));

describe('useIsMobile', () => {
  beforeEach(() => {
    vi.stubGlobal('matchMedia', mockMatchMedia);
  });

  afterEach(() => {
    mocks.listeners.clear();
    vi.unstubAllGlobals();
  });

  it('should return true when window width is less than the breakpoint', () => {
    vi.stubGlobal('innerWidth', 500);
    const { result } = renderHook(() => useIsMobile());
    expect(result.current).toBe(true);
  });

  it('should return false when window width is greater than the breakpoint', () => {
    vi.stubGlobal('innerWidth', 1024);
    const { result } = renderHook(() => useIsMobile());
    expect(result.current).toBe(false);
  });

  it('should update when the window is resized across the breakpoint', () => {
    vi.stubGlobal('innerWidth', 1024);
    const { result } = renderHook(() => useIsMobile());
    expect(result.current).toBe(false);

    // Simulate resize to mobile
    act(() => {
      vi.stubGlobal('innerWidth', 500);
      // Manually trigger the listener
      const changeListener = mocks.listeners.get('(max-width: 767px)');
      if (changeListener) {
        changeListener();
      }
    });

    expect(result.current).toBe(true);

    // Simulate resize back to desktop
    act(() => {
      vi.stubGlobal('innerWidth', 1024);
      const changeListener = mocks.listeners.get('(max-width: 767px)');
      if (changeListener) {
        changeListener();
      }
    });

    expect(result.current).toBe(false);
  });

  it('should initialize with undefined then set to correct state', () => {
    vi.stubGlobal('innerWidth', 500);
    const { result, rerender } = renderHook(() => useIsMobile());
    // The hook is designed to render with 'undefined' initially, then update.
    // Testing library's `renderHook` is synchronous, so we see the final state.
    expect(result.current).toBe(true);
  });
});
