import { describe, expect, it } from "vitest";

import { shouldBlockAuthoringCommand } from "./authoring-command-guard";

describe("shouldBlockAuthoringCommand", () => {
  it("applies commands while local authoring is enabled", () => {
    expect(shouldBlockAuthoringCommand(true)).toBe(false);
    expect(
      shouldBlockAuthoringCommand(true, { isUploadCompletion: true }),
    ).toBe(false);
  });

  it("blocks unrelated commands while local authoring is paused", () => {
    expect(shouldBlockAuthoringCommand(false)).toBe(true);
    expect(shouldBlockAuthoringCommand(false, {})).toBe(true);
  });

  it("lets a file-upload completion through while its own upload pauses authoring", () => {
    expect(
      shouldBlockAuthoringCommand(false, { isUploadCompletion: true }),
    ).toBe(false);
  });

  it("never treats a room replay as new authoring", () => {
    expect(shouldBlockAuthoringCommand(false, { syncRoom: false })).toBe(false);
  });

  it("does not let the upload-completion exemption cover unrelated commands", () => {
    // A command that is not flagged as the upload's own completion stays
    // blocked even though an upload is in flight and pausing authoring.
    expect(shouldBlockAuthoringCommand(false)).toBe(true);
    expect(shouldBlockAuthoringCommand(false, { isUploadCompletion: false })).toBe(
      true,
    );
  });
});