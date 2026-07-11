import { request } from "./client";
import type { Project } from "./types";

export function listProjects() {
  return request<Project[]>("GET", "/v1/projects");
}

export function createProject(input: {
  name: string;
  description?: string | null;
}) {
  return request<Project>("POST", "/v1/projects", {
    body: { name: input.name, description: input.description ?? null },
  });
}
