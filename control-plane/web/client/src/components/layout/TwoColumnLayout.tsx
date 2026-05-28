import type { ReactNode } from "react";

interface TwoColumnLayoutProps {
  leftColumn: ReactNode;
  rightColumn: ReactNode;
  leftWidth?: "2/3" | "1/2" | "3/4";
  rightWidth?: "1/3" | "1/2" | "1/4";
  className?: string;
  gap?: "sm" | "md" | "lg";
}

export function TwoColumnLayout({
  leftColumn,
  rightColumn,
  leftWidth = "2/3",
  rightWidth = "1/3",
  className = "",
  gap = "md",
}: TwoColumnLayoutProps) {
  const gapClasses = {
    sm: "gap-4",
    md: "gap-6",
    lg: "gap-8",
  };

  const leftWidthClasses = {
    "2/3": "w-full lg:w-2/3",
    "1/2": "w-full lg:w-1/2",
    "3/4": "w-full lg:w-3/4",
  };

  const rightWidthClasses = {
    "1/3": "w-full lg:w-1/3",
    "1/2": "w-full lg:w-1/2",
    "1/4": "w-full lg:w-1/4",
  };

  return (
    <div className={`flex flex-col lg:flex-row ${gapClasses[gap]} ${className}`}>
      {/* Left Column - DAG */}
      <div className={`${leftWidthClasses[leftWidth]} min-w-0`}>
        {leftColumn}
      </div>

      {/* Right Column - Timeline/Notes */}
      <div className={`${rightWidthClasses[rightWidth]} min-w-0`}>
        {rightColumn}
      </div>
    </div>
  );
}
