import { artifactPayloadUrl, request } from "./client";
import type { Artifact, ArtifactInspection, UUID } from "./types";

export { artifactPayloadUrl };

export function getArtifact(artifactId: UUID) {
  return request<Artifact>("GET", `/v1/artifacts/${artifactId}`);
}

export function inspectArtifact(
  artifactId: UUID,
  options: { includePayload?: boolean; includeTextPayload?: boolean } = {},
) {
  return request<ArtifactInspection>(
    "GET",
    `/v1/artifacts/${artifactId}/inspect`,
    {
      query: {
        include_payload: options.includePayload ?? false,
        include_text_payload: options.includeTextPayload ?? false,
      },
    },
  );
}
