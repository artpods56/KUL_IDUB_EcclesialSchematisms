"use client";

import * as React from "react";
import * as stylex from "@stylexjs/stylex";
import Link from "next/link";

import { useTheme } from "@/components/theme";
import { tokens } from "@/lib/stylex/tokens.stylex";

const s = stylex.create({
  page: {
    display: "grid",
    gridTemplateRows: "auto minmax(0, 1fr)",
    height: "100svh",
    backgroundColor: tokens.colorBg,
    color: tokens.colorText,
  },
  header: {
    display: "flex",
    alignItems: "center",
    gap: "12px",
    minHeight: "48px",
    padding: "8px 16px",
    borderBottomWidth: 1,
    borderBottomStyle: "solid",
    borderBottomColor: tokens.colorBorder,
    backgroundColor: tokens.colorChrome,
  },
  crumb: {
    color: tokens.colorMuted,
    fontSize: tokens.fontSizeSm,
  },
  title: {
    minWidth: 0,
    flex: 1,
    overflow: "hidden",
    fontSize: tokens.fontSizeMd,
    fontWeight: 600,
    letterSpacing: "-0.01em",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  headerButton: {
    height: "28px",
    display: "inline-flex",
    alignItems: "center",
    paddingInline: "9px",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: tokens.colorBorder,
    borderRadius: tokens.radiusSm,
    backgroundColor: {
      default: "transparent",
      ":hover": tokens.colorHover,
    },
    color: tokens.colorText,
    cursor: "pointer",
    fontSize: tokens.fontSizeXs,
  },
  body: {
    display: "grid",
    gridTemplateRows: "auto auto minmax(0, 1fr)",
    minHeight: 0,
  },
  toolbar: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    flexWrap: "wrap",
    padding: "10px 16px 0",
  },
  variant: {
    height: "28px",
    paddingInline: "10px",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: tokens.colorBorder,
    borderRadius: tokens.radiusSm,
    backgroundColor: {
      default: "transparent",
      ":hover": tokens.colorHover,
    },
    color: tokens.colorText,
    cursor: "pointer",
    fontSize: tokens.fontSizeSm,
  },
  variantActive: {
    backgroundColor: tokens.colorAccentSoft,
    borderColor: tokens.colorBorder,
  },
  note: {
    padding: "8px 16px 12px",
    color: tokens.colorMuted,
    fontSize: tokens.fontSizeSm,
    lineHeight: 1.45,
  },
  stage: {
    minHeight: 0,
    overflow: "auto",
    padding: "28px 32px 48px",
    backgroundColor: tokens.colorBg,
    backgroundImage: `linear-gradient(to right, ${tokens.colorGrid} 1px, transparent 1px), linear-gradient(to bottom, ${tokens.colorGrid} 1px, transparent 1px)`,
    backgroundSize: "20px 20px",
  },
});

export function SandboxShell({
  title,
  note,
  variants,
  activeVariant,
  onVariant,
  children,
}: {
  title: string;
  note: string;
  variants: readonly { id: string; label: string }[];
  activeVariant: string;
  onVariant: (id: string) => void;
  children: React.ReactNode;
}) {
  const { cycleTheme, preference } = useTheme();

  return (
    <div {...stylex.props(s.page)}>
      <header {...stylex.props(s.header)}>
        <Link href="/sandbox" {...stylex.props(s.crumb)}>
          Sandbox
        </Link>
        <span {...stylex.props(s.crumb)}>/</span>
        <h1 {...stylex.props(s.title)}>{title}</h1>
        <button
          type="button"
          {...stylex.props(s.headerButton)}
          onClick={cycleTheme}
        >
          Theme · {preference}
        </button>
      </header>
      <div {...stylex.props(s.body)}>
        <div {...stylex.props(s.toolbar)}>
          {variants.map((variant) => (
            <button
              key={variant.id}
              type="button"
              {...stylex.props(
                s.variant,
                variant.id === activeVariant ? s.variantActive : null,
              )}
              onClick={() => onVariant(variant.id)}
            >
              {variant.label}
            </button>
          ))}
        </div>
        <p {...stylex.props(s.note)}>{note}</p>
        <div {...stylex.props(s.stage)}>{children}</div>
      </div>
    </div>
  );
}
