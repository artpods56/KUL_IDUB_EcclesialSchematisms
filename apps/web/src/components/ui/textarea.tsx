"use client";

import * as React from "react";
import * as stylex from "@stylexjs/stylex";
import { tokens } from "@/lib/stylex/tokens.stylex";
import { cx } from "@/lib/utils";

const s = stylex.create({
  base: {
    width: "100%",
    display: "block",
    minHeight: "80px",
    borderRadius: tokens.radiusMd,
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: { default: tokens.colorBorder, ":focus": tokens.colorAccent },
    backgroundColor: tokens.colorSurfaceRaised,
    color: tokens.colorText,
    padding: `${tokens.space3} ${tokens.space3}`,
    fontSize: tokens.fontSizeSm,
    fontFamily: "inherit",
    lineHeight: 1.5,
    resize: "vertical",
    outlineWidth: 0,
    boxShadow: { default: "none", ":focus": `0 0 0 3px ${tokens.colorAccentSoft}` },
    transitionProperty: "border-color, box-shadow",
    transitionDuration: "120ms",
  },
});

export type TextareaProps = React.TextareaHTMLAttributes<HTMLTextAreaElement>;

export const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, ...props }, ref) => (
    <textarea
      ref={ref}
      className={cx(stylex.props(s.base).className, className)}
      {...props}
    />
  ),
);
Textarea.displayName = "Textarea";
