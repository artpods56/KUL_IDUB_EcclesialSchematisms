import { tokens } from "@/lib/stylex/tokens.stylex";

type CSSProperties = Record<string, string | number>;

export function handleStyle(
  top: number | string,
  color: string,
  variadic = false,
): CSSProperties {
  return {
    top: typeof top === "number" ? `${top}px` : top,
    width: "30px",
    height: "30px",
    borderRadius: "9999px",
    background: variadic
      ? `radial-gradient(circle, ${tokens.colorSurface} 0 3px, ${color} 4px 6px, ${tokens.colorSurface} 7px 8px, transparent 9px)`
      : `radial-gradient(circle, ${color} 0 5px, ${tokens.colorSurface} 6px 7px, transparent 8px)`,
    border: "none",
    boxShadow: "none",
    cursor: "crosshair",
    touchAction: "none",
  };
}
