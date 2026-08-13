import * as stylex from "@stylexjs/stylex";

import { tokens } from "./tokens.stylex";

/**
 * Quiet-wash overlay chrome. Floating pickers and menus use a hairline
 * border and raised shadow — not the ink ring on `shadowNodeSelected`.
 * Interactive rows fill on hover/active; they do not grow a second box.
 */
export const overlay = stylex.create({
  popup: {
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: tokens.colorBorder,
    borderRadius: tokens.radiusLg,
    backgroundColor: tokens.colorSurface,
    boxShadow: tokens.shadowNodeRaised,
    color: tokens.colorText,
  },
  item: {
    borderWidth: 0,
    backgroundColor: {
      default: "transparent",
      ":hover": tokens.colorHover,
    },
  },
  itemActive: {
    backgroundColor: {
      default: tokens.colorAccentSoft,
      ":hover": tokens.colorAccentSoft,
    },
  },
});
