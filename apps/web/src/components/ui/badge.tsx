import * as React from "react";
import * as stylex from "@stylexjs/stylex";
import { tokens } from "@/lib/stylex/tokens.stylex";
import { cx } from "@/lib/utils";

export type BadgeVariant = "default" | "accent" | "success" | "warning" | "danger" | "info";

const s = stylex.create({
  base: {
    display: "inline-flex",
    alignItems: "center",
    gap: tokens.space1,
    paddingInline: tokens.space2,
    paddingBlock: "2px",
    borderRadius: "9999px",
    fontSize: tokens.fontSizeXs,
    fontWeight: 600,
    lineHeight: 1.4,
    letterSpacing: "0.02em",
    whiteSpace: "nowrap",
    borderStyle: "solid",
    borderWidth: 1,
  },
  default: {
    backgroundColor: "#1a2238",
    color: tokens.colorMuted,
    borderColor: tokens.colorBorder,
  },
  accent: {
    backgroundColor: tokens.colorAccentSoft,
    color: tokens.colorAccent,
    borderColor: "transparent",
  },
  success: {
    backgroundColor: "rgba(52,211,153,0.16)",
    color: tokens.colorSuccess,
    borderColor: "transparent",
  },
  warning: {
    backgroundColor: "rgba(251,191,36,0.16)",
    color: tokens.colorWarning,
    borderColor: "transparent",
  },
  danger: {
    backgroundColor: "rgba(248,113,113,0.16)",
    color: tokens.colorDanger,
    borderColor: "transparent",
  },
  info: {
    backgroundColor: "rgba(96,165,250,0.16)",
    color: tokens.colorInfo,
    borderColor: "transparent",
  },
});

const variantStyles: Record<BadgeVariant, stylex.StyleXStyles> = {
  default: s.default,
  accent: s.accent,
  success: s.success,
  warning: s.warning,
  danger: s.danger,
  info: s.info,
};

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant;
}

export function Badge({ variant = "default", className, ...props }: BadgeProps) {
  return (
    <span
      className={cx(stylex.props(s.base, variantStyles[variant]).className, className)}
      {...props}
    />
  );
}
