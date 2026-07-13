# Notarius Workbench Context

## Product scope

The active product is the typed artifact-graph workbench. This repository
intentionally contains only that product slice; legacy extraction pipelines,
Dagster jobs, distributed workers, experiments, and compatibility modules are
outside its scope.

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

### Plugin

An installable declaration that groups nodes, artifact types, and the runtime
resolver/writer factories they require under one stable slug. Built-in plugins
are installed explicitly by the host; external plugins are discovered from the
`notarius.plugins` Python entry-point group. Plugins depend inward on core
contracts and ports, never on the API host or concrete storage adapters.

### Port

A named node input or output that declares an artifact type and cardinality.
Ports may carry one artifact, an ordered artifact sequence, or variadic incoming
edges when explicitly declared. Port cardinality describes the value seen by
one operator invocation; it does not decide how many times the operator runs.

### Edge collection mode

The transport policy stored on an edge. `direct` passes the produced value to
the target with its collection shape unchanged. `map` connects a produced
sequence to one required item input, calls the target operator once for each
item, broadcasts its other inputs, and aggregates required item outputs into
ordered sequences. The runtime derives its internal invocation policy from
incoming edges. A target has at most one map driver; zip, Cartesian, and
implicit flattening semantics are not part of the contract.

### Workflow graph

A set of configured node instances and directed edges. The graph must be
validated for operator identity and version, edge collection modes, port
existence, required inputs, effective cardinality, compatibility, declared
projections, and cycles before any node executes.

### Workbench

The user-facing graph editor and its execution interface. Node configuration is
rendered on the node from JSON Schema. Nested artifact fields and collection
mapping are selected on each edge. The complete graph or an exact,
self-contained node selection can be executed.
