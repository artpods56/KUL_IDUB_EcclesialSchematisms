import type { components, paths } from "./generated/prototype";

type PrototypeSchemas = components["schemas"];

export type PrototypeArtifactTypeKey =
  PrototypeSchemas["PrototypeArtifactTypeKeyResponse"];
export type PrototypeArtifactTypeSpec =
  PrototypeSchemas["PrototypeArtifactTypeSpecResponse"];
export type PrototypeFieldProjection =
  PrototypeSchemas["PrototypeFieldProjectionResponse"];
export type PrototypePort = PrototypeSchemas["PrototypePortResponse"];
export type PrototypeNodeSpec = PrototypeSchemas["PrototypeNodeSpecResponse"];
export type PrototypeSelectionItem =
  PrototypeSchemas["PrototypeSelectionItemResponse"];
export type PrototypeRunNodeInput =
  PrototypeSchemas["PrototypeRunNodeRequest"];
export type PrototypeNodeConfigInput = NonNullable<
  PrototypeRunNodeInput["config"]
>;
export type PrototypeRunEdgeInput =
  PrototypeSchemas["PrototypeRunEdgeRequest"];
export type PrototypeRunEdgeProjectionInput = NonNullable<
  PrototypeRunEdgeInput["projection"]
>;
export type PrototypeArtifactSummary =
  PrototypeSchemas["PrototypeArtifactSummaryResponse"];
export type PrototypeRunPortOutput =
  PrototypeSchemas["PrototypeRunPortOutputResponse"];
export type PrototypeRunNodeResult =
  PrototypeSchemas["PrototypeRunNodeResponse"];

export type PrototypeNodeRegistry =
  paths["/v1/prototype/nodes"]["get"]["responses"][200]["content"]["application/json"];
export type PrototypeUploadRequest =
  paths["/v1/prototype/uploads"]["post"]["requestBody"]["content"]["application/json"];
export type PrototypeUploadResponse =
  paths["/v1/prototype/uploads"]["post"]["responses"][200]["content"]["application/json"];
export type PrototypeSampleRequest =
  paths["/v1/prototype/samples"]["post"]["requestBody"]["content"]["application/json"];
export type PrototypeSampleResponse =
  paths["/v1/prototype/samples"]["post"]["responses"][200]["content"]["application/json"];
export type PrototypeRunRequest =
  paths["/v1/prototype/run"]["post"]["requestBody"]["content"]["application/json"];
export type PrototypeRunResponse =
  paths["/v1/prototype/run"]["post"]["responses"][200]["content"]["application/json"];

export type PrototypeNodeGroup = PrototypeNodeSpec["group"];
export type PrototypePortDirection = PrototypePort["direction"];
export type PrototypePortShape = PrototypePort["shape"];
export type PrototypeRunStatus = PrototypeRunResponse["status"];
export type PrototypeNodeRunStatus = PrototypeRunNodeResult["status"];
export type PrototypeJsonSchema = PrototypeNodeSpec["config_schema"];
