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
Installed plugins may also declare versioned conversions between artifact
types.

### Artifact reference

A lightweight reference to a persisted artifact. Graph execution moves
references between nodes and materializes Python values only at a node input.

### Upstream output pin

An exact `ArtifactRef` or `ArtifactRefSequence` copied from a materialized output
binding when a selected run begins. The incoming crossing edge remains in the
selected-subgraph request and keeps owning its projection, conversion, and
`direct`/`map` collection mode; the pin supplies that edge's source value
without executing its source node. The server consumes the submitted reference
and never performs a fuzzy "latest artifact" lookup.

### Materialized output binding

The durable record of a successful output for one exact saved graph revision,
node, and output port. In this workbench, **latest** means the binding identified
by `(graph id, graph revision, node id, output port)`; it never means the newest
artifact with a matching type or producer. A binding is reusable only when all
of its artifact references are accessible through the active runtime.
Inaccessible references are not advertised as available outputs.

### Field projection

A declared path from one compound artifact payload to a value that satisfies
another artifact type. A projection is selected on an edge; it is not a visible
adapter node and does not introduce operation-specific leaf artifact types.

### Artifact conversion

A declared, versioned, shape-preserving conversion from one artifact type to
another, such as `scalar.integer@1` to `scalar.text@1`. A conversion changes the
artifact representation; unlike a field projection, it does not select a nested
value. The selected conversion key is stored on the edge, validated against the
installed registry, and materialized as a target-typed artifact before the
downstream node runs. A conversion may be selected automatically while drawing
an edge when it is the only compatible route, but it is never inferred again at
execution time. Configurable, lossy, or domain-significant transformations
remain visible nodes.

### Node

A typed operation with a configuration model, input model, output model, and a
single execution method. Port contracts are derived from its model annotations.

### Plugin

An installable declaration that groups nodes, artifact types, artifact
conversions, and the runtime resolver/writer factories they require under one
stable slug. Built-in plugins are installed explicitly by the host; external
plugins are discovered from the `notarius.plugins` Python entry-point group.
Plugins depend inward on core contracts and ports, never on the API host or
concrete storage adapters.

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
projections and conversions, and cycles before any node executes. Edge value
handling has one fixed order: optional field projection, optional artifact
conversion, then `direct` or `map` collection handling. Conversion chains are
not implicit.

### Saved graph

A durable workbench document containing a workflow graph plus user-authored
canvas layout. It stores configured node identities, positions, semantic edge
endpoints, projections, conversion keys, collection modes, and edge routing
offsets. Registry metadata, callbacks, selection, viewport state, and execution
results are derived or runtime state and are not part of the saved aggregate.
Materialized output bindings are durable runtime records keyed to a saved
revision, not fields inside the saved graph. Upstream output pins belong to an
individual run request and remain transient. Drafts may be saved before they
are executable.

Saved graphs use optimistic revisions. Replacing a graph requires the revision
last read by the caller so competing edits are reported instead of silently
overwriting one another.

### Workbench

The user-facing graph editor and its execution interface. Node configuration is
rendered on the node from JSON Schema. Nested artifact fields and collection
mapping are selected on each edge. Compatible declared conversions are also
stored and displayed on the edge; a unique conversion may be selected
automatically when the user connects otherwise-incompatible ports. The complete
graph or a selected subgraph can be executed. By default, selected execution
includes internal and incoming crossing edges and reuses the exact materialized
output binding for each unselected source port. If a required binding is
missing, the run is blocked and the workbench directs the user to run the
upstream node or run with dependencies. `Run with dependencies` is a separate
action that expands the selection to its full upstream closure and executes that
expanded graph. Pins and live running state remain transient; revision-scoped
materialized outputs are restored when a saved graph is reopened.
