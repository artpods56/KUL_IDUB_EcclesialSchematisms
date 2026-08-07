import { beforeEach, expect, it, vi } from "vitest";

const redirect = vi.hoisted(() => vi.fn());

vi.mock("next/navigation", () => ({ redirect }));

import Home from "./page";

beforeEach(() => {
  redirect.mockClear();
});

it("redirects the root route to the workspace directory", () => {
  Home();

  expect(redirect).toHaveBeenCalledOnce();
  expect(redirect).toHaveBeenCalledWith("/workspaces");
});
