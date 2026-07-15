import * as stylex from "@stylexjs/stylex";

/**
 * Global design tokens for Notarius Studio.
 *
 * Color tokens use CSS `light-dark()` so the active palette follows
 * `document.documentElement.style.colorScheme` (set by ThemeProvider).
 */
export const tokens = stylex.defineVars({
  // surfaces
  colorBg: "light-dark(#f7f8fa, #111214)",
  colorSurface: "light-dark(#ffffff, #242629)",
  colorSurfaceRaised: "light-dark(#f0f1f3, #2c2f33)",
  colorSurfaceSunken: "light-dark(#e8eaed, #1e2023)",
  colorSurfaceMuted: "light-dark(#f3f4f6, #1b1d20)",
  colorChrome: "light-dark(#ffffff, #242629)",
  // borders
  colorBorder: "light-dark(#d4d7dc, #35383d)",
  colorBorderStrong: "light-dark(#b8bcc4, #44484f)",
  colorDivider: "light-dark(rgba(0, 0, 0, 0.1), rgba(255, 255, 255, 0.1))",
  // text
  colorText: "light-dark(#1a1c20, #eef0f2)",
  colorTextEmphasis: "light-dark(#2a2d32, #f0f1f3)",
  colorMuted: "light-dark(#5c6169, #a7abb2)",
  colorSubtle: "light-dark(#8b9099, #747982)",
  colorTextDisabled: "light-dark(#a0a4ab, #77727f)",
  colorOnAccent: "light-dark(#ffffff, #ffffff)",
  // interaction
  colorHover: "light-dark(rgba(0, 0, 0, 0.05), rgba(255, 255, 255, 0.07))",
  colorHoverStrong: "light-dark(rgba(0, 0, 0, 0.08), rgba(255, 255, 255, 0.1))",
  colorDangerHover: "light-dark(rgba(220, 92, 92, 0.1), rgba(232, 105, 105, 0.12))",
  // accents
  colorAccent: "light-dark(#6b52d4, #8067e8)",
  colorAccentHover: "light-dark(#7a63e0, #9077f0)",
  colorAccentDisabled: "light-dark(#c4b8e8, #3b3847)",
  colorAccentSoft: "light-dark(rgba(107, 82, 212, 0.12), rgba(128, 103, 232, 0.18))",
  colorAccentBorder: "light-dark(rgba(107, 82, 212, 0.45), rgba(190, 168, 255, 0.72))",
  colorProjectionPath: "light-dark(#6b52d4, #bdb4e7)",
  // status
  colorSuccess: "light-dark(#2a9d7c, #43c59e)",
  colorWarning: "light-dark(#c9920f, #fbbf24)",
  colorDanger: "light-dark(#dc5c5c, #f87171)",
  colorInfo: "light-dark(#4a8fd4, #60a5fa)",
  // canvas
  colorGrid: "light-dark(rgba(0, 0, 0, 0.06), rgba(255, 255, 255, 0.035))",
  colorFlowControls: "light-dark(#ffffff, rgba(29, 31, 35, 0.92))",
  // elevation
  shadowNode:
    "0 1px 2px light-dark(rgba(20, 24, 32, 0.1), rgba(0, 0, 0, 0.38)), 0 8px 22px light-dark(rgba(20, 24, 32, 0.1), rgba(0, 0, 0, 0.32))",
  shadowNodeSelected:
    "0 2px 5px light-dark(rgba(107, 82, 212, 0.16), rgba(128, 103, 232, 0.22)), 0 12px 30px light-dark(rgba(20, 24, 32, 0.14), rgba(0, 0, 0, 0.46))",
  // geometry
  radiusSm: "6px",
  radiusMd: "10px",
  radiusLg: "16px",
  space1: "4px",
  space2: "8px",
  space3: "12px",
  space4: "16px",
  space5: "20px",
  space6: "24px",
  fontSizeXs: "11px",
  fontSizeSm: "12px",
  fontSizeMd: "13px",
  fontSizeLg: "1.125rem",
  fontSizeXl: "1.5rem",
});
