"use client";

import * as React from "react";
import { Button as BaseButton } from "@base-ui/react/button";
import * as stylex from "@stylexjs/stylex";
import { tokens } from "@/lib/stylex/tokens.stylex";
import { cx } from "@/lib/utils";

export type ButtonVariant =
  | "primary"
  | "secondary"
  | "outline"
  | "ghost"
  | "danger";
export type ButtonSize = "sm" | "md" | "lg" | "icon";

const s = stylex.create({
  base: {
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    gap: tokens.space2,
    borderRadius: tokens.radiusMd,
    fontFamily: "inherit",
    fontWeight: 600,
    fontSize: tokens.fontSizeSm,
    lineHeight: 1,
    borderWidth: 0,
    borderStyle: "solid",
    borderColor: "transparent",
    cursor: { default: "pointer", ":disabled": "not-allowed" },
    color: { default: tokens.colorText, ":disabled": tokens.colorSubtle },
    opacity: { default: 1, ":disabled": 0.55 },
    backgroundColor: { default: "transparent", ":disabled": "transparent" },
    transitionProperty: "background-color, color, border-color, opacity",
    transitionDuration: "120ms",
    userSelect: "none",
    whiteSpace: "nowrap",
  },
  primary: {
    backgroundColor: { default: tokens.colorAccent, ":hover": "#a5acff" },
    color: "#0b1020",
  },
  secondary: {
    backgroundColor: { default: tokens.colorSurfaceRaised, ":hover": "#1d2942" },
    color: tokens.colorText,
    borderColor: tokens.colorBorder,
    borderWidth: 1,
  },
  outline: {
    backgroundColor: { default: "transparent", ":hover": tokens.colorAccentSoft },
    color: tokens.colorText,
    borderColor: tokens.colorBorderStrong,
    borderWidth: 1,
  },
  ghost: {
    backgroundColor: { default: "transparent", ":hover": "#1a2238" },
    color: tokens.colorMuted,
  },
  danger: {
    backgroundColor: { default: "rgba(248,113,113,0.16)", ":hover": "rgba(248,113,113,0.24)" },
    color: tokens.colorDanger,
    borderColor: "rgba(248,113,113,0.4)",
    borderWidth: 1,
  },
  sm: { height: "30px", paddingInline: tokens.space3, fontSize: tokens.fontSizeXs },
  md: { height: "36px", paddingInline: tokens.space4 },
  lg: { height: "42px", paddingInline: tokens.space5, fontSize: tokens.fontSizeMd },
  icon: { height: "34px", width: "34px", padding: 0 },
});

const variantStyles: Record<ButtonVariant, stylex.StyleXStyles> = {
  primary: s.primary,
  secondary: s.secondary,
  outline: s.outline,
  ghost: s.ghost,
  danger: s.danger,
};

const sizeStyles: Record<ButtonSize, stylex.StyleXStyles> = {
  sm: s.sm,
  md: s.md,
  lg: s.lg,
  icon: s.icon,
};

export interface ButtonProps
  extends Omit<
    React.ComponentPropsWithoutRef<typeof BaseButton>,
    "render" | "type" | "className"
  > {
  variant?: ButtonVariant;
  size?: ButtonSize;
  /** @deprecated Prefer `render`. Kept for Radix Slot compatibility. */
  asChild?: boolean;
  render?: React.ComponentPropsWithoutRef<typeof BaseButton>["render"];
  type?: React.ButtonHTMLAttributes<HTMLButtonElement>["type"];
  className?: string;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      variant = "secondary",
      size = "md",
      asChild,
      render,
      className,
      type,
      children,
      nativeButton,
      ...props
    },
    ref,
  ) => {
    const buttonClassName = cx(
      stylex.props(s.base, variantStyles[variant], sizeStyles[size]).className,
      className,
    );

    const childElement =
      asChild && React.isValidElement(children)
        ? (React.Children.only(children) as React.ReactElement)
        : undefined;

    const resolvedRender =
      render ??
      (childElement as React.ComponentPropsWithoutRef<typeof BaseButton>["render"]);
    const usesCustomElement = Boolean(resolvedRender);
    const resolvedNativeButton =
      nativeButton ?? (usesCustomElement ? childElement?.type === "button" : true);

    return (
      <BaseButton
        ref={ref}
        type={usesCustomElement ? undefined : (type ?? "button")}
        nativeButton={resolvedNativeButton}
        render={resolvedRender}
        className={buttonClassName}
        {...props}
      >
        {usesCustomElement ? null : children}
      </BaseButton>
    );
  },
);
Button.displayName = "Button";
