"use client";

import * as React from "react";
import * as stylex from "@stylexjs/stylex";
import { tokens } from "@/lib/stylex/tokens.stylex";
import { cx } from "@/lib/utils";

const s = stylex.create({
  base: {
    width: "100%",
    display: "block",
    height: "36px",
    borderRadius: tokens.radiusMd,
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: { default: tokens.colorBorder, ":focus": tokens.colorAccent },
    backgroundColor: tokens.colorSurfaceRaised,
    color: tokens.colorText,
    paddingInline: tokens.space3,
    fontSize: tokens.fontSizeSm,
    fontFamily: "inherit",
    outlineWidth: 0,
    boxShadow: { default: "none", ":focus": `0 0 0 3px ${tokens.colorAccentSoft}` },
    transitionProperty: "border-color, box-shadow",
    transitionDuration: "120ms",
  },
});

export type InputProps = React.InputHTMLAttributes<HTMLInputElement>;

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, ...props }, ref) => (
    <input
      ref={ref}
      className={cx(stylex.props(s.base).className, className)}
      {...props}
    />
  ),
);
Input.displayName = "Input";
