"use client";

import * as React from "react";
import * as stylex from "@stylexjs/stylex";
import { Switch as SwitchPrimitive } from "@base-ui/react/switch";
import { tokens } from "@/lib/stylex/tokens.stylex";
import { cx } from "@/lib/utils";

const s = stylex.create({
  root: {
    position: "relative",
    display: "inline-flex",
    height: "22px",
    width: "40px",
    borderRadius: "9999px",
    borderWidth: 1,
    borderStyle: "solid",
    cursor: "pointer",
    flexShrink: 0,
    transitionProperty: "background-color, border-color",
    transitionDuration: "120ms",
  },
  rootOff: {
    backgroundColor: tokens.colorSurfaceRaised,
    borderColor: tokens.colorBorderStrong,
  },
  rootOn: {
    backgroundColor: tokens.colorAccent,
    borderColor: "transparent",
  },
  thumb: {
    position: "absolute",
    left: "2px",
    top: "50%",
    transform: "translateY(-50%)",
    height: "16px",
    width: "16px",
    borderRadius: "9999px",
    backgroundColor: "#ffffff",
    boxShadow: "0 1px 3px rgba(0,0,0,0.4)",
    transitionProperty: "transform",
    transitionDuration: "120ms",
  },
  thumbOn: { transform: "translateY(-50%) translateX(18px)" },
});

export interface SwitchProps
  extends Omit<
    React.ComponentPropsWithoutRef<typeof SwitchPrimitive.Root>,
    "onCheckedChange" | "className"
  > {
  checked?: boolean;
  onCheckedChange?: (checked: boolean) => void;
  className?: string;
}

export const Switch = React.forwardRef<HTMLElement, SwitchProps>(
  ({ className, checked, onCheckedChange, ...props }, ref) => (
    <SwitchPrimitive.Root
      ref={ref}
      checked={checked}
      onCheckedChange={
        onCheckedChange
          ? (nextChecked) => onCheckedChange(nextChecked)
          : undefined
      }
      className={cx(
        stylex.props(s.root, checked ? s.rootOn : s.rootOff).className,
        className,
      )}
      {...props}
    >
      <SwitchPrimitive.Thumb
        {...stylex.props(s.thumb, checked ? s.thumbOn : null)}
      />
    </SwitchPrimitive.Root>
  ),
);
Switch.displayName = "Switch";
