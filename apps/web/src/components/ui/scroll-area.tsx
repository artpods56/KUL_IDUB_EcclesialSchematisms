"use client";

import * as React from "react";
import * as stylex from "@stylexjs/stylex";
import { ScrollArea as ScrollAreaPrimitive } from "@base-ui/react/scroll-area";
import { tokens } from "@/lib/stylex/tokens.stylex";
import { cx } from "@/lib/utils";

const s = stylex.create({
  root: { position: "relative", overflow: "hidden" },
  viewport: { width: "100%", height: "100%", borderRadius: "inherit" },
  scrollbar: {
    display: "flex",
    userSelect: "none",
    touchAction: "none",
    padding: "2px",
    transition: "background-color 120ms",
  },
  thumb: {
    backgroundColor: tokens.colorBorderStrong,
    borderRadius: "9999px",
    flexGrow: 1,
  },
  corner: { backgroundColor: "transparent" },
  vertical: { width: "10px", height: "100%" },
  horizontal: { height: "10px", width: "100%", flexDirection: "column" },
});

export const ScrollArea = React.forwardRef<
  HTMLDivElement,
  Omit<React.ComponentPropsWithoutRef<typeof ScrollAreaPrimitive.Root>, "className"> & {
    className?: string;
  }
>(({ className, children, ...props }, ref) => (
  <ScrollAreaPrimitive.Root
    ref={ref}
    className={cx(stylex.props(s.root).className, className)}
    {...props}
  >
    <ScrollAreaPrimitive.Viewport {...stylex.props(s.viewport)}>
      {children}
    </ScrollAreaPrimitive.Viewport>
    <ScrollAreaPrimitive.Scrollbar
      orientation="vertical"
      {...stylex.props(s.scrollbar, s.vertical)}
    >
      <ScrollAreaPrimitive.Thumb {...stylex.props(s.thumb)} />
    </ScrollAreaPrimitive.Scrollbar>
    <ScrollAreaPrimitive.Scrollbar
      orientation="horizontal"
      {...stylex.props(s.scrollbar, s.horizontal)}
    >
      <ScrollAreaPrimitive.Thumb {...stylex.props(s.thumb)} />
    </ScrollAreaPrimitive.Scrollbar>
    <ScrollAreaPrimitive.Corner {...stylex.props(s.corner)} />
  </ScrollAreaPrimitive.Root>
));
ScrollArea.displayName = "ScrollArea";
