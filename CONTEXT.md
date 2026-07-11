# Notarius Workbench Context

## Product scope

The active product is the typed artifact-graph workbench established by the
prototype. This repository intentionally contains only that product slice;
legacy extraction pipelines, Dagster jobs, distributed workers, experiments,
and compatibility modules are outside its scope.

## Domain vocabulary

### Artifact

A typed, versioned value produced or consumed by a node. Its payload may be
inline or stored as content, but its artifact type and schema version are
always explicit.

### Artifact type

The contract for an artifact payload. It owns the stable type id, schema
version, payload schema, display title, and any declared field projections.

### Artifact reference

A lightweight reference to a persisted artifact. Graph execution moves
references between nodes and materializes Python values only at a node input.

### Field projection

A declared path from one compound artifact payload to a value that satisfies
another artifact type. A projection is selected on an edge; it is not a visible
adapter node and does not introduce operation-specific leaf artifact types.

### Node

A typed operation with a configuration model, input model, output model, and a
single execution method. Port contracts are derived from its model annotations.

### Port

A named node input or output that declares an artifact type and cardinality.
Ports may carry one artifact, an ordered artifact sequence, or variadic incoming
edges when explicitly declared.

### Workflow graph

A set of configured node instances and directed edges. The graph must be
validated for port existence, required inputs, cardinality, compatibility,
declared projections, and cycles before any node executes.

### Workbench

The user-facing graph editor and its execution interface. Node configuration is
rendered on the node from JSON Schema, and nested artifact fields are selected
when connecting compatible ports.
