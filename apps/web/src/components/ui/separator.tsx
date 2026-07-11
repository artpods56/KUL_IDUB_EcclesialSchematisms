"use client";

import * as React from "react";
import * as stylex from "@stylexjs/stylex";
import { Separator as SeparatorPrimitive } from "@base-ui/react/separator";
import { tokens } from "@/lib/stylex/tokens.stylex";
import { cx } from "@/lib/utils";

const s = stylex.create({
  base: {
    backgroundColor: tokens.colorBorder,
    flexShrink: 0,
  },
  horizontal: { height: "1px", width: "100%" },
  vertical: { width: "1px", height: "100%" },
});

export const Separator = React.forwardRef<
  HTMLDivElement,
  Omit<React.ComponentPropsWithoutRef<typeof SeparatorPrimitive>, "className"> & {
    className?: string;
  }
>(({ className, orientation = "horizontal", ...props }, ref) => (
  <SeparatorPrimitive
    ref={ref}
    orientation={orientation}
    className={cx(
      stylex.props(s.base, orientation === "vertical" ? s.vertical : s.horizontal)
        .className,
      className,
    )}
    {...props}
  />
));
Separator.displayName = "Separator";
