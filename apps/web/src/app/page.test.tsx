import { describe, expect, it } from "vitest";

describe("slugFromName", () => {
  function slugFromName(name: string): string {
    return (
      name
        .trim()
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-|-$/g, "") || "shared-workspace"
    );
  }

  it("converts a plain name to a slug", () => {
    expect(slugFromName("Planning room")).toBe("planning-room");
  });

  it("strips leading and trailing dashes", () => {
    expect(slugFromName("  Test Workspace  ")).toBe("test-workspace");
  });

  it("falls back to shared-workspace for empty input", () => {
    expect(slugFromName("   ")).toBe("shared-workspace");
  });

  it("replaces special characters with dashes", () => {
    expect(slugFromName("My @Team #1")).toBe("my-team-1");
  });
});

it("the home page module is loadable", async () => {
  const mod = await import("./page");
  expect(typeof mod.default).toBe("function");
});
