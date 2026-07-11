import * as stylex from "@stylexjs/stylex";

/**
 * Global design tokens for Notarius Studio.
 *
 * Defined with `stylex.defineVars` so each token becomes a CSS custom
 * property (e.g. `--color-bg`) and can be referenced from any `stylex.create`
 * call across the app. This is the StyleX equivalent of the shadcn CSS-variable
 * theme. Dark-first for now; a light theme can be layered later with
 * `stylex.createTheme`.
 */
export const tokens = stylex.defineVars({
  // surfaces
  colorBg: "#111214",
  colorSurface: "#242629",
  colorSurfaceRaised: "#2c2f33",
  colorBorder: "#35383d",
  colorBorderStrong: "#44484f",
  // text
  colorText: "#eef0f2",
  colorMuted: "#a7abb2",
  colorSubtle: "#747982",
  // accents
  colorAccent: "#8067e8",
  colorAccentSoft: "rgba(128, 103, 232, 0.18)",
  // status
  colorSuccess: "#43c59e",
  colorWarning: "#fbbf24",
  colorDanger: "#f87171",
  colorInfo: "#60a5fa",
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
  fontSizeXs: "0.75rem",
  fontSizeSm: "0.875rem",
  fontSizeMd: "0.95rem",
  fontSizeLg: "1.125rem",
  fontSizeXl: "1.5rem",
  // misc
  shadowCard: "0 10px 30px rgba(0,0,0,0.35)",
});
