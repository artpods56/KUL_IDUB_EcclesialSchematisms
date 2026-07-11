"use client";

import * as React from "react";
import * as stylex from "@stylexjs/stylex";
import { Tooltip as TooltipPrimitive } from "@base-ui/react/tooltip";
import { tokens } from "@/lib/stylex/tokens.stylex";
import { cx } from "@/lib/utils";

const s = stylex.create({
  content: {
    borderRadius: tokens.radiusSm,
    backgroundColor: "#05070f",
    border: `1px solid ${tokens.colorBorderStrong}`,
    color: tokens.colorText,
    fontSize: tokens.fontSizeXs,
    padding: `${tokens.space1} ${tokens.space2}`,
    boxShadow: "0 6px 20px rgba(0,0,0,0.5)",
    zIndex: 60,
    maxWidth: "260px",
  },
});

export interface TooltipProviderProps
  extends React.ComponentPropsWithoutRef<typeof TooltipPrimitive.Provider> {
  /** @deprecated Use `delay`. Kept for Radix API compatibility. */
  delayDuration?: number;
  /** @deprecated Use `timeout`. Kept for Radix API compatibility. */
  skipDelayDuration?: number;
}

export function TooltipProvider({
  delay,
  timeout,
  delayDuration,
  skipDelayDuration,
  ...props
}: TooltipProviderProps) {
  return (
    <TooltipPrimitive.Provider
      delay={delay ?? delayDuration}
      timeout={timeout ?? skipDelayDuration}
      {...props}
    />
  );
}

export const Tooltip = TooltipPrimitive.Root;
export const TooltipTrigger = TooltipPrimitive.Trigger;

export const TooltipContent = React.forwardRef<
  HTMLDivElement,
  Omit<React.ComponentPropsWithoutRef<typeof TooltipPrimitive.Popup>, "className"> & {
    sideOffset?: number;
    className?: string;
  }
>(({ className, sideOffset = 6, ...props }, ref) => (
  <TooltipPrimitive.Portal>
    <TooltipPrimitive.Positioner sideOffset={sideOffset}>
      <TooltipPrimitive.Popup
        ref={ref}
        className={cx(stylex.props(s.content).className, className)}
        {...props}
      />
    </TooltipPrimitive.Positioner>
  </TooltipPrimitive.Portal>
));
TooltipContent.displayName = "TooltipContent";
