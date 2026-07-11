import { artifactPayloadUrl, request, uploadImageSource } from "./client";
import type {
  Artifact,
  ArtifactSequence,
  ImageSourceUploadResponse,
  Source,
  UUID,
} from "./types";

export { uploadImageSource, artifactPayloadUrl };

export function listProjectSources(projectId: UUID) {
  return request<Source[]>("GET", `/v1/projects/${projectId}/sources`);
}

export function getSource(sourceId: UUID) {
  return request<Source>("GET", `/v1/sources/${sourceId}`);
}

export function listSourceArtifacts(sourceId: UUID) {
  return request<Artifact[]>("GET", `/v1/sources/${sourceId}/artifacts`);
}

export function listSourceArtifactSequences(sourceId: UUID) {
  return request<ArtifactSequence[]>(
    "GET",
    `/v1/sources/${sourceId}/artifact-sequences`,
  );
}

export type { ImageSourceUploadResponse };
