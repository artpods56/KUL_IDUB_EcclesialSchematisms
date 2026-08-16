import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";

import { ThresholdStatus } from "./threshold-status";

describe("ThresholdStatus", () => {
  it("uses the animated brand mark while loading", () => {
    const markup = renderToStaticMarkup(
      <ThresholdStatus
        title="Loading graph location"
        detail="Checking your current access…"
        loading
      />,
    );
    expect(markup).toContain("grafy-brand-loader");
    expect(markup).toContain("Loading graph location");
    expect(markup).toContain("Checking your current access…");
    expect(markup).not.toContain("grafy-brand-wordmark");
  });

  it("uses the wordmark and action once the route has settled", () => {
    const markup = renderToStaticMarkup(
      <ThresholdStatus
        title="Graph location not found"
        detail="This graph location is not available to your account."
        action={<a href="/graphs">Return to graphs</a>}
      />,
    );
    expect(markup).toContain("grafy-brand-wordmark");
    expect(markup).toContain("Return to graphs");
    expect(markup).not.toContain("grafy-brand-loader");
  });
});
