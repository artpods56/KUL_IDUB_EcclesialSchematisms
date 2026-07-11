"use client";

import * as React from "react";
import * as stylex from "@stylexjs/stylex";
import { Tabs as TabsPrimitive } from "@base-ui/react/tabs";
import { tokens } from "@/lib/stylex/tokens.stylex";
import { cx } from "@/lib/utils";

const s = stylex.create({
  list: {
    display: "inline-flex",
    alignItems: "center",
    gap: tokens.space1,
    padding: "3px",
    borderRadius: tokens.radiusMd,
    backgroundColor: tokens.colorSurfaceRaised,
    border: `1px solid ${tokens.colorBorder}`,
  },
  trigger: {
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    gap: tokens.space1,
    height: "28px",
    paddingInline: tokens.space3,
    borderRadius: tokens.radiusSm,
    fontSize: tokens.fontSizeSm,
    fontWeight: 600,
    color: tokens.colorMuted,
    backgroundColor: "transparent",
    border: "none",
    cursor: "pointer",
    whiteSpace: "nowrap",
  },
  content: {
    paddingTop: tokens.space4,
    outline: "none",
  },
});

type TabsRootProps = Omit<
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.Root>,
  "onValueChange"
> & {
  onValueChange?: (value: string) => void;
};

export function Tabs({ onValueChange, ...props }: TabsRootProps) {
  return (
    <TabsPrimitive.Root
      {...props}
      onValueChange={
        onValueChange
          ? (value) => onValueChange(String(value))
          : undefined
      }
    />
  );
}

export const TabsList = React.forwardRef<
  HTMLDivElement,
  Omit<React.ComponentPropsWithoutRef<typeof TabsPrimitive.List>, "className"> & {
    className?: string;
  }
>(({ className, ...props }, ref) => (
  <TabsPrimitive.List
    ref={ref}
    className={cx(stylex.props(s.list).className, "ns-tabs-list", className)}
    {...props}
  />
));
TabsList.displayName = "TabsList";

export const TabsTrigger = React.forwardRef<
  HTMLButtonElement,
  Omit<React.ComponentPropsWithoutRef<typeof TabsPrimitive.Tab>, "className"> & {
    className?: string;
  }
>(({ className, ...props }, ref) => (
  <TabsPrimitive.Tab
    ref={ref}
    className={cx(stylex.props(s.trigger).className, "ns-tabs-trigger", className)}
    {...props}
  />
));
TabsTrigger.displayName = "TabsTrigger";

export const TabsContent = React.forwardRef<
  HTMLDivElement,
  Omit<React.ComponentPropsWithoutRef<typeof TabsPrimitive.Panel>, "className"> & {
    className?: string;
  }
>(({ className, ...props }, ref) => (
  <TabsPrimitive.Panel
    ref={ref}
    className={cx(stylex.props(s.content).className, className)}
    {...props}
  />
));
TabsContent.displayName = "TabsContent";
