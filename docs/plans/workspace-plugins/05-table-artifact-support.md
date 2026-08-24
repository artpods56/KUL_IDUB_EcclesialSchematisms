# Slice 5: Table artifact support and runnable releases

- **Status:** Complete
- **Updated:** 2026-08-24
- **Depends on:** [Slice 3](03-artifact-invocation-protocol.md) and
  [Slice 4](04-oci-and-docker-runtime.md)
- **Outcome:** Workspace Plugin nodes can consume and produce `table.data@1`
  through the sandbox artifact boundary, and the catalog enables only releases
  whose complete contracts are executable

## Why this slice exists

Inline JSON proves the transport but does not exercise the primary research
workflow. `examples/plugin-notes` consumes `table.data@1`, and an artifact ref
alone cannot reconstruct that table inside a networkless Plugin runtime.

The existing Table artifact already has a persisted manifest and chunk layout.
This slice adapts that representation into a portable, validated bundle rather
than converting complete tables to JSON or importing Table Plugin Python into
FastAPI.

## Scope

- Host export and import for `table.data@1` bundles.
- Guest resolver/writer support for the existing Table wire representation.
- Efficient local staging and safe non-local copying.
- End-to-end execution of `examples/plugin-notes`.
- Limited Plugin-owned artifact support where the wire format is portable.
- Derived catalog `runnable` status and reason.
- Mixed host/Workspace Plugin graph integration tests.

## Non-goals

- Arbitrary host artifact types.
- Cross-Plugin Python resolver imports.
- General custom codec marketplace or dependency solver.
- Plugin-owned conversions between unrelated type families.
- Secrets, egress, external databases, or provider APIs.
- Enabling cache policies other than `NEVER`.

## Fixed decisions

`table.data@1` is a wire contract, not a dependency on the in-process Table
Plugin package. The host exports authorized stored content into a bundle; the
runtime profile or Plugin SDK provides the portable Table resolver/writer used
inside the image.

All storage adapters use bounded, digest-verified copies into
invocation-owned scratch. The current `FileStoragePort` deliberately exposes
streams rather than canonical filesystem paths, and Docker consumes one
portable archive, so a local-only hardlink/reflink branch would pierce both
boundaries without removing the archive copy. A storage-owned immutable
snapshot/link operation can be added later if profiling justifies it. Scratch
cleanup never addresses canonical objects; output cleanup records only objects
atomically created by the invocation.

A catalog node is runnable only when all of these are true:

- its release has a compatible runtime artifact and protocol version;
- every input and output type has a supported bundle implementation;
- its approved capabilities are supported by the deployment;
- the compiler can resolve an exact release pin;
- the invocation adapter is available.

`runnable` is derived readiness, not a manual promise.

## Expected ownership

- Table bundle export/import near existing Table artifact persistence
- Portable guest Table codec in the runtime/SDK boundary
- Explicit readiness whitelist containing only proven supported core formats
- Catalog readiness calculation in the Workspace release overlay
- `examples/plugin-notes` as the executable fixture

## Implementation checklist

### Table input export

- [x] Specify and version the portable `table.data@1` bundle manifest.
- [x] Map the existing stored Table manifest and chunks into that bundle
      without materializing the whole table in API memory.
- [x] Authorize the Table artifact row and every referenced stored object
      before staging.
- [x] Validate stored byte counts and hashes while exporting.
- [x] Use bounded, verified copies for every storage adapter; keep local
      hardlinks/reflinks deferred behind a future storage-owned snapshot API.
- [x] Preserve column schema, chunk order, row counts, and content provenance.

### Guest Table resolver and writer

- [x] Resolve a staged Table bundle into the Python value expected by the
      Plugin node without host database access.
- [x] Reject incompatible schema, missing chunks, reordered chunks, and hash
      mismatches before invoking the node.
- [x] Serialize returned Table values into a complete output bundle.
- [x] Produce deterministic manifest metadata and chunk ordering.
- [x] Enforce row, column, chunk, and total-byte limits from the invocation
      policy.
- [x] Keep Table codec code in a platform/runtime package rather than importing
      another Plugin working copy.

### Host Table import

- [x] Validate the complete output manifest and every chunk before persistence.
- [x] Mint canonical storage keys and artifact IDs on the host.
- [x] Import chunks without loading the complete result in memory.
- [x] Commit every artifact row for the invocation in one unit of work before
      returning any graph-level output binding.
- [x] Remove newly created partial objects after a failed import without
      deleting pre-existing content-addressed objects.
- [x] Preserve the existing artifact content digest and provenance semantics.

### Example Plugin and owned types

- [x] Make `examples/plugin-notes` depend on a freezeable SDK/runtime source.
- [x] Publish an image-backed `notes` release through the production path.
- [x] Execute `notes.table.summarize@1` with a real Table artifact input.
- [x] Execute its downstream rendering node from the produced summary artifact.
- [x] Keep the Plugin-owned summary type inline or define its explicit portable
      bundle format.
- [x] Document that another Plugin cannot consume that custom type until it
      independently supports the same stable wire contract.

### Catalog readiness

- [x] Calculate readiness from release, protocol, type-bundle, capability, and
      invoker support.
- [x] Return a stable non-runnable reason for unsupported types, missing image,
      incompatible protocol, or unsupported capabilities.
- [x] Enable Add Node only for a release whose complete contract is runnable.
- [x] Preserve old source-only and partially supported releases for pinned
      history while keeping them disabled for new authoring.
- [x] Keep Workspace Plugin cache policy `NEVER`.

## Verification checklist

- [x] A multi-chunk Table input is staged without serializing its complete
      content as JSON or loading it entirely into API memory.
- [x] Scratch and failed-import cleanup cannot remove pre-existing canonical
      objects.
- [x] A Table from another Workspace cannot be staged.
- [x] Missing, reordered, duplicated, tampered, or oversized chunks are
      rejected.
- [x] Table schema and content hashes survive a sandbox round trip.
- [x] Output import failure leaves no partial Table artifact or other visible
      output binding.
- [x] `examples/plugin-notes` publishes and executes with `table.data@1`.
- [x] A mixed graph executes host nodes and the pinned `notes` release.
- [x] Publishing `notes` release N+1 does not retarget or change an N-pinned
      graph.
- [x] Catalog readiness accurately explains every disabled release fixture.
- [x] Compatible Workspace Plugin nodes appear in Add Node and remain pinned
      when inserted.

## Exit criteria

- [x] `table.data@1` crosses the host/sandbox boundary with bounded memory and
      verified content.
- [x] The example Plugin executes its complete supported flow offline.
- [x] Catalog `runnable` is derived and fail-closed.
- [x] Host and Workspace Plugin nodes coexist in one graph.
- [x] Unsupported Plugin-owned and cross-Plugin artifact types remain disabled
      with a clear reason.

## Agent handoff

- **Owner:** Codex
- **Branch or PR:** —
- **Implementation evidence:** `runtime/table_bundle.py` defines the
  deterministic `grafy.plugin.table-bundle.v1` manifest/archive contract;
  `plugin_artifacts.py` authorizes and streams stored Table chunks across the
  boundary, imports verified output chunks, and compensates failed object
  writes; `plugin_guest.py` resolves and emits Table bundles under invocation
  limits. Catalog models derive readiness from the immutable runtime artifact,
  `grafy-plugin-invocation@2` digest, runtime profile, capabilities, and every
  declared artifact type. The workbench selector renders stable disabled
  reasons and inserts the exact `plugin_release` pin.
- **Verification evidence:** 15 deterministic/adversarial Table bundle tests;
  19 host artifact-boundary tests including foreign-Workspace rejection and
  failed-import cleanup; four local subprocess protocol integrations; exact-pin
  compiler/runtime coverage including host → pinned Plugin execution and N/N+1
  stability; eight readiness fixtures; generated OpenAPI checks; and 46 focused
  workbench tests plus TypeScript typecheck and ESLint. The real Docker
  integration builds the frozen example through the production publisher/OCI
  components and executes a stored 205-row, three-chunk Table through
  `notes.table.summarize@1` and `notes.summary.render@1` offline.
- **Performance evidence:** Local Docker Desktop (Apple Silicon), 2026-08-24,
  refreshed v2 SDK wheel: cold cache-restore/container/child path `3.278s`;
  warm child path `1.548s`.
- **Open decisions or blockers:** None for Slice 5. Local hardlink/reflink
  staging remains a profiling-led optimization that first needs a safe
  storage-owned immutable snapshot API. Plugin-owned inline JSON is portable
  only within releases that independently declare and support its stable
  contract; arbitrary custom and cross-Plugin codecs remain deferred.
