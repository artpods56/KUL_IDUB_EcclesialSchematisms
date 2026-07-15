import { vi } from "vitest";

vi.mock("@/lib/stylex/tokens.stylex", () => ({
  tokens: {
    colorAccent: "#000000",
    colorSurface: "#ffffff",
  },
}));
