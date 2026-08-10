import { request } from "./client";
import type {
  ImportModuleReleaseRequest,
  ImportModuleReleaseResponse,
  ModuleLibraryEntry,
  ModuleList,
  PublishModuleReleaseRequest,
} from "./contract";

export function listWorkspaceModules(workspaceId: string) {
  return request<ModuleList>(
    "GET",
    `/v1/workspaces/${encodeURIComponent(workspaceId)}/modules`,
  );
}

export function getWorkspaceModule(workspaceId: string, moduleId: string) {
  return request<ModuleLibraryEntry>(
    "GET",
    `/v1/workspaces/${encodeURIComponent(workspaceId)}/modules/${encodeURIComponent(moduleId)}`,
  );
}

export function publishModuleRelease(
  workspaceId: string,
  body: PublishModuleReleaseRequest,
) {
  return request<ModuleLibraryEntry>(
    "POST",
    `/v1/workspaces/${encodeURIComponent(workspaceId)}/modules/publish`,
    { body },
  );
}

export function deprecateModule(workspaceId: string, moduleId: string) {
  return request<ModuleLibraryEntry>(
    "POST",
    `/v1/workspaces/${encodeURIComponent(workspaceId)}/modules/${encodeURIComponent(moduleId)}/deprecate`,
  );
}

export function withdrawModule(workspaceId: string, moduleId: string) {
  return request<ModuleLibraryEntry>(
    "POST",
    `/v1/workspaces/${encodeURIComponent(workspaceId)}/modules/${encodeURIComponent(moduleId)}/withdraw`,
  );
}

export function importModuleRelease(
  workspaceId: string,
  body: ImportModuleReleaseRequest,
) {
  return request<ImportModuleReleaseResponse>(
    "POST",
    `/v1/workspaces/${encodeURIComponent(workspaceId)}/modules/import`,
    { body },
  );
}
