"use client";

import * as React from "react";
import * as stylex from "@stylexjs/stylex";
import { tokens } from "@/lib/stylex/tokens.stylex";
import { cx } from "@/lib/utils";

const s = stylex.create({
  base: {
    display: "inline-block",
    fontSize: tokens.fontSizeXs,
    fontWeight: 600,
    color: tokens.colorMuted,
    marginBottom: tokens.space1,
    letterSpacing: "0.02em",
    textTransform: "uppercase",
  },
});

export const Label = React.forwardRef<
  HTMLLabelElement,
  React.LabelHTMLAttributes<HTMLLabelElement>
>(({ className, ...props }, ref) => (
  <label
    ref={ref}
    className={cx(stylex.props(s.base).className, className)}
    {...props}
  />
));
Label.displayName = "Label";
