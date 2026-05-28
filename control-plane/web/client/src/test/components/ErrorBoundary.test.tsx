import React from "react";
import { render, renderHook, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  ErrorBoundary,
  useErrorHandler,
  withErrorBoundary,
} from "@/components/ErrorBoundary";

vi.mock("@/components/ui/alert", () => ({
  Alert: ({
    children,
    ...props
  }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => (
    <div {...props}>{children}</div>
  ),
}));

vi.mock("@/components/ui/button", () => ({
  Button: ({
    children,
    ...props
  }: React.PropsWithChildren<React.ButtonHTMLAttributes<HTMLButtonElement>>) => (
    <button {...props}>{children}</button>
  ),
}));

vi.mock("@/components/ui/card", () => ({
  Card: ({
    children,
    ...props
  }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => (
    <div {...props}>{children}</div>
  ),
  CardContent: ({
    children,
    ...props
  }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => (
    <div {...props}>{children}</div>
  ),
  CardHeader: ({
    children,
    ...props
  }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => (
    <div {...props}>{children}</div>
  ),
  CardTitle: ({
    children,
    ...props
  }: React.PropsWithChildren<React.HTMLAttributes<HTMLHeadingElement>>) => (
    <h2 {...props}>{children}</h2>
  ),
}));

vi.mock("@/components/ui/icon-bridge", () => ({
  Restart: (props: React.HTMLAttributes<HTMLSpanElement>) => <span {...props}>restart</span>,
  WarningFilled: (props: React.HTMLAttributes<HTMLSpanElement>) => (
    <span {...props}>warning</span>
  ),
}));

describe("ErrorBoundary", () => {
  beforeEach(() => {
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders children when no error occurs", () => {
    render(
      <ErrorBoundary>
        <div>Healthy child</div>
      </ErrorBoundary>
    );

    expect(screen.getByText("Healthy child")).toBeInTheDocument();
  });

  it("renders the default fallback, reports errors, and resets on demand", async () => {
    const user = userEvent.setup();
    const onError = vi.fn();
    let shouldThrow = true;

    function FlakyChild({ crash }: { crash: boolean }) {
      if (crash) {
        throw new Error("Boom");
      }
      return <div>Recovered child</div>;
    }

    const { rerender } = render(
      <ErrorBoundary onError={onError}>
        <FlakyChild crash={shouldThrow} />
      </ErrorBoundary>
    );

    expect(screen.getByText("Something went wrong")).toBeInTheDocument();
    expect(screen.getByText(/Boom/)).toBeInTheDocument();
    expect(onError).toHaveBeenCalledTimes(1);

    shouldThrow = false;
    rerender(
      <ErrorBoundary onError={onError}>
        <FlakyChild crash={shouldThrow} />
      </ErrorBoundary>
    );

    await user.click(screen.getByRole("button", { name: /Try Again/i }));
    expect(screen.getByText("Recovered child")).toBeInTheDocument();
  });

  it("supports custom fallbacks, reset keys, HOCs, and hooks", () => {
    let crash = true;

    function Crashy({ active }: { active: boolean }) {
      if (active) {
        throw new Error("Crash");
      }
      return <div>Recovered after key change</div>;
    }

    const { rerender } = render(
      <ErrorBoundary
        fallback={<div>Custom fallback</div>}
        resetOnPropsChange
        resetKeys={["first"]}
      >
        <Crashy active={crash} />
      </ErrorBoundary>
    );

    expect(screen.getByText("Custom fallback")).toBeInTheDocument();

    crash = false;
    rerender(
      <ErrorBoundary resetOnPropsChange resetKeys={["second"]}>
        <Crashy active={crash} />
      </ErrorBoundary>
    );

    expect(screen.getByText("Recovered after key change")).toBeInTheDocument();

    const Wrapped = withErrorBoundary(({ label }: { label: string }) => <div>{label}</div>);
    render(<Wrapped label="Wrapped component" />);
    expect(screen.getByText("Wrapped component")).toBeInTheDocument();

    const { result } = renderHook(() => useErrorHandler());
    expect(() => result.current(new Error("Manual error"))).toThrow("Manual error");
  });
});
