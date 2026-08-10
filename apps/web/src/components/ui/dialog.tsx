"use client";

import * as React from "react";
import * as stylex from "@stylexjs/stylex";
import { Dialog as DialogPrimitive } from "@base-ui/react/dialog";
import { X } from "lucide-react";
import { tokens } from "@/lib/stylex/tokens.stylex";
import { cx } from "@/lib/utils";

const s = stylex.create({
  overlay: {
    position: "fixed",
    zIndex: 100,
    inset: 0,
    backgroundColor: "light-dark(rgba(15, 18, 25, 0.35), rgba(2, 6, 20, 0.62))",
  },
  content: {
    position: "fixed",
    zIndex: 100,
    left: "50%",
    top: "50%",
    transform: "translate(-50%, -50%)",
    width: "640px",
    maxWidth: "92vw",
    maxHeight: "85vh",
    borderRadius: tokens.radiusLg,
    backgroundColor: tokens.colorSurface,
    border: `1px solid ${tokens.colorBorder}`,
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
  },
  header: {
    padding: `${tokens.space4} ${tokens.space4} ${tokens.space2}`,
    paddingRight: tokens.space6,
  },
  title: {
    fontSize: tokens.fontSizeLg,
    fontWeight: 700,
    color: tokens.colorText,
    lineHeight: 1.3,
  },
  description: {
    fontSize: tokens.fontSizeSm,
    color: tokens.colorMuted,
    marginTop: tokens.space1,
  },
  body: {
    padding: `${tokens.space2} ${tokens.space4} ${tokens.space4}`,
    overflow: "auto",
    flex: 1,
  },
  close: {
    position: "absolute",
    top: tokens.space3,
    right: tokens.space3,
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    height: "28px",
    width: "28px",
    borderRadius: tokens.radiusSm,
    color: tokens.colorMuted,
    backgroundColor: "transparent",
    border: "none",
    cursor: "pointer",
  },
});

export const Dialog = DialogPrimitive.Root;

export const DialogContent = React.forwardRef<
  HTMLDivElement,
  Omit<React.ComponentPropsWithoutRef<typeof DialogPrimitive.Popup>, "className"> & {
    className?: string;
  }
>(({ className, children, ...props }, ref) => (
  <DialogPrimitive.Portal>
    <DialogPrimitive.Backdrop
      className={cx(stylex.props(s.overlay).className, "ns-dialog-backdrop")}
    />
    <DialogPrimitive.Popup
      ref={ref}
      className={cx(
        stylex.props(s.content).className,
        "ns-dialog-popup",
        className,
      )}
      {...props}
    >
      {children}
      <DialogPrimitive.Close
        {...stylex.props(s.close)}
        aria-label="Close"
      >
        <X size={16} />
      </DialogPrimitive.Close>
    </DialogPrimitive.Popup>
  </DialogPrimitive.Portal>
));
DialogContent.displayName = "DialogContent";

export function DialogHeader({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cx(stylex.props(s.header).className, className)} {...props} />
  );
}

export const DialogTitle = React.forwardRef<
  HTMLHeadingElement,
  Omit<React.ComponentPropsWithoutRef<typeof DialogPrimitive.Title>, "className"> & {
    className?: string;
  }
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Title
    ref={ref}
    className={cx(stylex.props(s.title).className, className)}
    {...props}
  />
));
DialogTitle.displayName = "DialogTitle";

export const DialogDescription = React.forwardRef<
  HTMLParagraphElement,
  Omit<React.ComponentPropsWithoutRef<typeof DialogPrimitive.Description>, "className"> & {
    className?: string;
  }
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Description
    ref={ref}
    className={cx(stylex.props(s.description).className, className)}
    {...props}
  />
));
DialogDescription.displayName = "DialogDescription";

export function DialogBody({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cx(stylex.props(s.body).className, className)} {...props} />
  );
}
