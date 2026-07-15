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
types. Together those declarations form the artifact conversion graph.

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

A path from one compound artifact payload to a value that can be materialized as
another artifact type. The registry derives structural projections for nested
JSON Schema `string` and `integer` leaves when canonical scalar targets are
installed; plugins may explicitly override a path when they need a different
target or title. A projection is selected on an edge, persisted by exact path,
and validated again at runtime. It is not a visible adapter node and does not
introduce operation-specific leaf artifact types. Arrays and schema-less dynamic
properties are not inferred as scalar projections.

### Materialized scalar

An artifact type that declares it can materialize one JSON primitive runtime
value. `scalar.text@1` consumes and produces runtime `str`; its writer wraps that
primitive in the stable `{ "value": string }` payload and its resolver unwraps
the payload again. `scalar.integer@1` follows the same boundary for runtime
`int`. Storage envelopes must not leak into node input, output, projection, or
conversion callables.

### Artifact conversion

A declared, versioned, shape-preserving conversion from one artifact type to
another, such as `scalar.integer@1` to `scalar.text@1`. A conversion changes the
artifact representation; unlike a field projection, it does not select a nested
value. Configurable, lossy, or domain-significant transformations remain visible
nodes.

### Artifact conversion path

A bounded, ordered sequence of exact conversion keys stored on one edge. The
installed conversions form a directed graph, so declarations `X -> Y` and
`Y -> Z` make `X -> Z` authorable without an adapter node. Each selected path is
simple: it cannot revisit an artifact type, even though the global registry may
contain conversions in both directions. Registry construction rejects adjacent
declarations whose runtime target and source types cannot compose. Authoring may
automatically select one unambiguous shortest path, but execution never searches
the registry again. It validates and replays the stored keys in order, composes
their pure callables in memory, and materializes only the final target-typed
artifact.

### Node

A typed operation with a configuration model, input model, output model, and a
single execution method. Port contracts are derived from its model annotations.

### Input plug

A stable, ordered input position owned by one node instance for a port that
explicitly supports instance plugs. Each plug accepts exactly one incoming edge
and keeps its identity when reordered, so the saved plug order—not canvas
coordinates or edge creation order—defines execution order. The edge continues
to own its projection and artifact conversion path.

### Plugin

An installable declaration that groups nodes, artifact types, artifact
conversions, and the runtime resolver/writer factories they require under one
stable slug. Built-in plugins are installed explicitly by the host; external
plugins are discovered from the `notarius.plugins` Python entry-point group.
Plugins depend inward on core contracts and ports, never on the API host or
concrete storage adapters.

### Port

A named node input or output that declares either one concrete artifact type or
a named artifact-type variable, plus cardinality. Every use of the same variable
on one node shares one concrete binding owned by that node instance. Ports may
carry one artifact, an ordered artifact sequence, or variadic incoming edges when
explicitly declared. Port cardinality describes the value seen by one operator
invocation; it does not decide how many times the operator runs.

### Edge collection mode

The transport policy stored on an edge. `direct` passes the produced value to
the target with its collection shape unchanged. `map` connects a produced
sequence to one required item input, calls the target operator once for each
item, broadcasts its other inputs, and aggregates required item outputs into
ordered sequences. The runtime derives its internal invocation policy from
incoming edges. A target has at most one map driver; zip, Cartesian, and
implicit flattening semantics are not part of the contract.

### Collect node

The generic cardinality-changing operation `Collect<T>`. A Collect node instance
binds `T` to one concrete artifact type; every ordered input plug then accepts
either one `T` artifact or one `T` sequence, and its output is a sequence of `T`.
It appends scalar references and expands sequence references exactly one level in
plug order, producing a fresh `ArtifactRefSequence` without rewriting its
artifact items. Different shapes may be combined, but different artifact types
may not. If any source sequence is unordered, that unordered state propagates to
the result. Collection is node behavior; every incoming edge remains `direct`,
and `map` is not valid for a Collect input.

### Workflow graph

A set of configured node instances and directed edges. The graph must be
validated for operator identity and version, edge collection modes, port
existence, required inputs, effective cardinality, compatibility, declared
projections and conversions, and cycles before any node executes. Edge value
handling has one fixed order: optional field projection, zero or more stored
artifact conversions, then `direct` or `map` collection handling.

### Saved graph

A durable workbench document containing a workflow graph plus user-authored
canvas layout. It stores configured node identities, positions, semantic edge
endpoints, ordered instance input plugs, node artifact-type bindings, projections,
conversion paths, collection modes, and edge routing offsets. A generic binding
survives even when its incident edges are temporarily removed; users reset it
explicitly before binding the node to another artifact type. Registry metadata,
callbacks, selection, viewport state, and execution results are derived or
runtime state and are not part of the saved aggregate.
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
mapping are selected on each edge. Compatible declared conversion paths are also
stored and displayed on the edge; a unique route may be selected automatically
when the user connects otherwise-incompatible ports. The complete
graph or a selected subgraph can be executed. By default, selected execution
includes internal and incoming crossing edges and reuses the exact materialized
output binding for each unselected source port. If a required binding is
missing, the run is blocked and the workbench directs the user to run the
upstream node or run with dependencies. `Run with dependencies` is a separate
action that expands the selection to its full upstream closure and executes that
expanded graph. Pins and live running state remain transient; revision-scoped
materialized outputs are restored when a saved graph is reopened.
