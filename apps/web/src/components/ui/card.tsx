import * as React from "react";
import * as stylex from "@stylexjs/stylex";
import { tokens } from "@/lib/stylex/tokens.stylex";
import { cx } from "@/lib/utils";

const s = stylex.create({
  card: {
    borderRadius: tokens.radiusLg,
    backgroundColor: tokens.colorSurface,
    border: `1px solid ${tokens.colorBorder}`,
    boxShadow: tokens.shadowCard,
  },
  header: {
    display: "flex",
    flexDirection: "column",
    gap: tokens.space1,
    padding: `${tokens.space4} ${tokens.space4} ${tokens.space2}`,
  },
  title: {
    fontSize: tokens.fontSizeMd,
    fontWeight: 700,
    color: tokens.colorText,
    lineHeight: 1.3,
  },
  description: {
    fontSize: tokens.fontSizeXs,
    color: tokens.colorMuted,
  },
  content: {
    padding: `${tokens.space2} ${tokens.space4} ${tokens.space4}`,
  },
});

export const Card = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div ref={ref} className={cx(stylex.props(s.card).className, className)} {...props} />
));
Card.displayName = "Card";

export const CardHeader = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cx(stylex.props(s.header).className, className)}
    {...props}
  />
));
CardHeader.displayName = "CardHeader";

export const CardTitle = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div ref={ref} className={cx(stylex.props(s.title).className, className)} {...props} />
));
CardTitle.displayName = "CardTitle";

export const CardDescription = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cx(stylex.props(s.description).className, className)}
    {...props}
  />
));
CardDescription.displayName = "CardDescription";

export const CardContent = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cx(stylex.props(s.content).className, className)}
    {...props}
  />
));
CardContent.displayName = "CardContent";
