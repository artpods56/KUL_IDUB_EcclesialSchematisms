"use client";

import * as React from "react";
import * as stylex from "@stylexjs/stylex";
import { Select as SelectPrimitive } from "@base-ui/react/select";
import { Check, ChevronDown } from "lucide-react";
import { tokens } from "@/lib/stylex/tokens.stylex";
import { cx } from "@/lib/utils";

const s = stylex.create({
  trigger: {
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: tokens.space2,
    height: "36px",
    width: "100%",
    paddingInline: tokens.space3,
    borderRadius: tokens.radiusMd,
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: { default: tokens.colorBorder, ":focus": tokens.colorAccent },
    backgroundColor: tokens.colorSurfaceRaised,
    color: tokens.colorText,
    fontSize: tokens.fontSizeSm,
    fontFamily: "inherit",
    cursor: "pointer",
    outline: "none",
  },
  value: {
    display: "flex",
    alignItems: "center",
    gap: tokens.space2,
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  placeholder: { color: tokens.colorSubtle },
  icon: { color: tokens.colorMuted, flexShrink: 0 },
  content: {
    overflow: "hidden",
    borderRadius: tokens.radiusMd,
    backgroundColor: tokens.colorSurfaceRaised,
    border: `1px solid ${tokens.colorBorderStrong}`,
    boxShadow: "0 16px 40px rgba(0,0,0,0.5)",
    width: "var(--anchor-width)",
    maxHeight: "var(--available-height)",
    zIndex: 50,
  },
  list: { padding: tokens.space1 },
  item: {
    position: "relative",
    display: "flex",
    alignItems: "center",
    height: "34px",
    borderRadius: tokens.radiusSm,
    paddingInline: tokens.space3,
    paddingInlineEnd: tokens.space6,
    fontSize: tokens.fontSizeSm,
    color: tokens.colorText,
    cursor: "pointer",
    outline: "none",
  },
  itemIndicator: {
    position: "absolute",
    right: tokens.space3,
    display: "inline-flex",
    alignItems: "center",
    color: tokens.colorAccent,
  },
  scrollButton: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    height: "28px",
    color: tokens.colorMuted,
    cursor: "default",
  },
});

type SelectRootProps = Omit<
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Root>,
  "onValueChange"
> & {
  onValueChange?: (value: string) => void;
};

export function Select({ onValueChange, children, ...props }: SelectRootProps) {
  return (
    <SelectPrimitive.Root
      {...props}
      onValueChange={
        onValueChange
          ? (value) => {
              if (value != null) onValueChange(String(value));
            }
          : undefined
      }
    >
      {children}
    </SelectPrimitive.Root>
  );
}

export const SelectGroup = SelectPrimitive.Group;

export interface SelectValueProps
  extends Omit<
    React.ComponentPropsWithoutRef<typeof SelectPrimitive.Value>,
    "children" | "className"
  > {
  placeholder?: string;
  className?: string;
  children?: React.ReactNode | ((value: unknown) => React.ReactNode);
}

export const SelectValue = React.forwardRef<HTMLSpanElement, SelectValueProps>(
  ({ className, placeholder, children, ...props }, ref) => {
    return (
      <SelectPrimitive.Value
        ref={ref}
        placeholder={placeholder}
        className={(state) =>
          cx(
            stylex.props(s.value, state.placeholder ? s.placeholder : null)
              .className,
            className,
          )
        }
        {...props}
      >
        {children}
      </SelectPrimitive.Value>
    );
  },
);
SelectValue.displayName = "SelectValue";

export const SelectTrigger = React.forwardRef<
  HTMLButtonElement,
  Omit<React.ComponentPropsWithoutRef<typeof SelectPrimitive.Trigger>, "className"> & {
    className?: string;
  }
>(({ className, children, ...props }, ref) => (
  <SelectPrimitive.Trigger
    ref={ref}
    className={cx(stylex.props(s.trigger).className, className)}
    {...props}
  >
    <span {...stylex.props(s.value)}>{children}</span>
    <SelectPrimitive.Icon {...stylex.props(s.icon)}>
      <ChevronDown size={16} />
    </SelectPrimitive.Icon>
  </SelectPrimitive.Trigger>
));
SelectTrigger.displayName = "SelectTrigger";

export interface SelectContentProps
  extends Omit<React.ComponentPropsWithoutRef<typeof SelectPrimitive.Popup>, "className"> {
  /** @deprecated Base UI uses Positioner; `popper` maps to alignItemWithTrigger=false. */
  position?: "item-aligned" | "popper";
  className?: string;
}

export const SelectContent = React.forwardRef<HTMLDivElement, SelectContentProps>(
  ({ className, children, position = "popper", ...props }, ref) => (
    <SelectPrimitive.Portal>
      <SelectPrimitive.Positioner
        sideOffset={4}
        alignItemWithTrigger={position !== "popper"}
      >
        <SelectPrimitive.Popup
          ref={ref}
          className={cx(stylex.props(s.content).className, className)}
          {...props}
        >
          <SelectPrimitive.ScrollUpArrow {...stylex.props(s.scrollButton)}>
            <ChevronDown size={14} style={{ transform: "rotate(180deg)" }} />
          </SelectPrimitive.ScrollUpArrow>
          <SelectPrimitive.List {...stylex.props(s.list)}>
            {children}
          </SelectPrimitive.List>
          <SelectPrimitive.ScrollDownArrow {...stylex.props(s.scrollButton)}>
            <ChevronDown size={14} />
          </SelectPrimitive.ScrollDownArrow>
        </SelectPrimitive.Popup>
      </SelectPrimitive.Positioner>
    </SelectPrimitive.Portal>
  ),
);
SelectContent.displayName = "SelectContent";

export const SelectItem = React.forwardRef<
  HTMLDivElement,
  Omit<React.ComponentPropsWithoutRef<typeof SelectPrimitive.Item>, "className"> & {
    className?: string;
  }
>(({ className, children, value, ...props }, ref) => {
  const itemLabel =
    typeof children === "string" || typeof children === "number"
      ? String(children)
      : undefined;

  return (
    <SelectPrimitive.Item
      ref={ref}
      value={value}
      label={itemLabel}
      className={cx(stylex.props(s.item).className, "ns-select-item", className)}
      {...props}
    >
      <SelectPrimitive.ItemText>{children}</SelectPrimitive.ItemText>
      <span {...stylex.props(s.itemIndicator)}>
        <SelectPrimitive.ItemIndicator>
          <Check size={14} />
        </SelectPrimitive.ItemIndicator>
      </span>
    </SelectPrimitive.Item>
  );
});
SelectItem.displayName = "SelectItem";
