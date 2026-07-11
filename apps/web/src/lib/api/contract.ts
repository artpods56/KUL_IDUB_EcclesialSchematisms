import type { components, paths } from "./generated/notarius";

type Schemas = components["schemas"];

export type ArtifactTypeKey =
  Schemas["ArtifactTypeKeyResponse"];
export type ArtifactTypeSpec =
  Schemas["ArtifactTypeSpecResponse"];
export type FieldProjection =
  Schemas["FieldProjectionResponse"];
export type Port = Schemas["PortResponse"];
export type NodeSpec = Schemas["NodeSpecResponse"];
export type SelectionItem =
  Schemas["SelectionItemResponse"];
export type RunNodeInput =
  Schemas["RunNodeRequest"];
export type NodeConfigInput = NonNullable<
  RunNodeInput["config"]
>;
export type RunEdgeInput =
  Schemas["RunEdgeRequest"];
export type RunEdgeProjectionInput = NonNullable<
  RunEdgeInput["projection"]
>;
export type ArtifactSummary =
  Schemas["ArtifactSummaryResponse"];
export type RunPortOutput =
  Schemas["RunPortOutputResponse"];
export type RunNodeResult =
  Schemas["RunNodeResponse"];

export type NodeRegistry =
  paths["/v1/nodes"]["get"]["responses"][200]["content"]["application/json"];
export type UploadRequest =
  paths["/v1/uploads"]["post"]["requestBody"]["content"]["application/json"];
export type UploadResponse =
  paths["/v1/uploads"]["post"]["responses"][200]["content"]["application/json"];
export type RunRequest =
  paths["/v1/runs"]["post"]["requestBody"]["content"]["application/json"];
export type RunResponse =
  paths["/v1/runs"]["post"]["responses"][200]["content"]["application/json"];

export type NodeGroup = NodeSpec["group"];
export type PortDirection = Port["direction"];
export type PortShape = Port["shape"];
export type RunStatus = RunResponse["status"];
export type NodeRunStatus = RunNodeResult["status"];
export type JsonSchema = NodeSpec["config_schema"];
