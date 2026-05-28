import { renderHook, act } from '@testing-library/react';
import { useDashboardTimeRange } from './useDashboardTimeRange';
import { beforeEach, describe, it, expect, vi } from 'vitest';
import { MemoryRouter, useSearchParams } from 'react-router-dom';
import React from 'react';

// Mock the useSearchParams hook
vi.mock('react-router-dom', async () => {
    const originalModule = await vi.importActual('react-router-dom');
    return {
        ...originalModule,
        useSearchParams: vi.fn(),
    };
});

const mockSetSearchParams = vi.fn();

type WrapperProps = React.PropsWithChildren<{ initialEntries?: string[] }>;

const wrapper = ({ children, initialEntries }: WrapperProps) => (
    <MemoryRouter initialEntries={initialEntries ?? ['/']}>{children}</MemoryRouter>
);

describe('useDashboardTimeRange', () => {
    beforeEach(() => {
        vi.mocked(useSearchParams).mockReturnValue([new URLSearchParams(), mockSetSearchParams]);
        mockSetSearchParams.mockClear();
    });

    it('initializes with default preset "24h"', () => {
        const { result } = renderHook(() => useDashboardTimeRange(), { wrapper });
        expect(result.current.timeRange.preset).toBe('24h');
        expect(result.current.label).toBe('Last 24 hours');
    });

    it('initializes with a custom default preset', () => {
        const { result } = renderHook(() => useDashboardTimeRange('7d'), { wrapper });
        expect(result.current.timeRange.preset).toBe('7d');
        expect(result.current.label).toBe('Last 7 days');
    });

    it('initializes from URL search params', () => {
        const searchParams = new URLSearchParams('range=7d&compare=true');
        vi.mocked(useSearchParams).mockReturnValue([searchParams, mockSetSearchParams]);
        const { result } = renderHook(() => useDashboardTimeRange(), { wrapper });

        expect(result.current.timeRange.preset).toBe('7d');
        expect(result.current.timeRange.compare).toBe(true);
    });

    it('initializes with custom date range from URL search params', () => {
        const startTime = '2023-01-01T00:00:00.000Z';
        const endTime = '2023-01-02T00:00:00.000Z';
        const searchParams = new URLSearchParams(`range=custom&start=${startTime}&end=${endTime}`);
        vi.mocked(useSearchParams).mockReturnValue([searchParams, mockSetSearchParams]);

        const { result } = renderHook(() => useDashboardTimeRange(), { wrapper });
        expect(result.current.timeRange.preset).toBe('custom');
        expect(result.current.timeRange.startTime).toEqual(new Date(startTime));
        expect(result.current.timeRange.endTime).toEqual(new Date(endTime));
    });

    it('handles invalid custom dates from URL by falling back', () => {
        const searchParams = new URLSearchParams('range=custom&start=invalid&end=invalid');
        vi.mocked(useSearchParams).mockReturnValue([searchParams, mockSetSearchParams]);
        
        const { result } = renderHook(() => useDashboardTimeRange('24h'), { wrapper });
        expect(result.current.timeRange.preset).toBe('custom');
        expect(result.current.timeRange.startTime).toBeNull();
        expect(result.current.timeRange.endTime).toBeNull();
    });


    it('setPreset updates the preset and syncs URL', () => {
        const { result } = renderHook(() => useDashboardTimeRange(), { wrapper });

        act(() => {
            result.current.setPreset('7d');
        });

        expect(result.current.timeRange.preset).toBe('7d');
        expect(result.current.label).toBe('Last 7 days');
        expect(mockSetSearchParams).toHaveBeenCalled();
        const lastCall = mockSetSearchParams.mock.calls[mockSetSearchParams.mock.calls.length - 1][0];
        expect(lastCall.get('range')).toBe('7d');
    });

    it('setCustomRange updates to a custom range and syncs URL', () => {
        const { result } = renderHook(() => useDashboardTimeRange(), { wrapper });
        const start = new Date('2023-03-15T10:00:00.000Z');
        const end = new Date('2023-03-16T10:00:00.000Z');

        act(() => {
            result.current.setCustomRange(start, end);
        });

        expect(result.current.timeRange.preset).toBe('custom');
        expect(result.current.timeRange.startTime).toEqual(start);
        expect(result.current.timeRange.endTime).toEqual(end);
        expect(mockSetSearchParams).toHaveBeenCalled();
        const lastCall = mockSetSearchParams.mock.calls[mockSetSearchParams.mock.calls.length - 1][0];
        expect(lastCall.get('range')).toBe('custom');
        expect(lastCall.get('start')).toBe(start.toISOString());
        expect(lastCall.get('end')).toBe(end.toISOString());
    });

    it('toggleCompare flips the compare flag and syncs URL', () => {
        const { result } = renderHook(() => useDashboardTimeRange(), { wrapper });
        expect(result.current.timeRange.compare).toBe(false);

        act(() => {
            result.current.toggleCompare();
        });

        expect(result.current.timeRange.compare).toBe(true);
        let lastCall = mockSetSearchParams.mock.calls[mockSetSearchParams.mock.calls.length - 1][0];
        expect(lastCall.get('compare')).toBe('true');

        act(() => {
            result.current.toggleCompare();
        });

        expect(result.current.timeRange.compare).toBe(false);
        lastCall = mockSetSearchParams.mock.calls[mockSetSearchParams.mock.calls.length - 1][0];
        expect(lastCall.has('compare')).toBe(false);
    });

    it('setCompare sets the compare flag and syncs URL', () => {
        const { result } = renderHook(() => useDashboardTimeRange(), { wrapper });
        
        act(() => {
            result.current.setCompare(true);
        });

        expect(result.current.timeRange.compare).toBe(true);
        const lastCall = mockSetSearchParams.mock.calls[mockSetSearchParams.mock.calls.length - 1][0];
        expect(lastCall.get('compare')).toBe('true');
    });

    it('getApiParams returns correct structure for presets', () => {
        const { result } = renderHook(() => useDashboardTimeRange('7d'), { wrapper });
        const params = result.current.getApiParams();
        expect(params).toEqual({
            preset: '7d',
            compare: false,
        });
    });

    it('getApiParams returns correct structure for custom ranges', () => {
        const { result } = renderHook(() => useDashboardTimeRange(), { wrapper });
        const start = new Date('2023-05-20T00:00:00.000Z');
        const end = new Date('2023-05-21T00:00:00.000Z');

        act(() => {
            result.current.setCustomRange(start, end);
        });

        const params = result.current.getApiParams();
        expect(params).toEqual({
            preset: 'custom',
            startTime: start.toISOString(),
            endTime: end.toISOString(),
            compare: false,
        });
    });

    it('does not set range param if preset is default', () => {
         const { result } = renderHook(() => useDashboardTimeRange('7d'), { wrapper });

         act(() => {
             result.current.setPreset('30d');
         });
         expect(result.current.timeRange.preset).toBe('30d');
         let lastCall = mockSetSearchParams.mock.calls[mockSetSearchParams.mock.calls.length - 1][0];
         expect(lastCall.get('range')).toBe('30d');

         act(() => {
             result.current.setPreset('7d');
         });

         expect(result.current.timeRange.preset).toBe('7d');
         lastCall = mockSetSearchParams.mock.calls[mockSetSearchParams.mock.calls.length - 1][0];
         expect(lastCall.has('range')).toBe(false);
    });
});
