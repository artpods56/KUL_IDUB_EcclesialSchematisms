# Modules conceptual model

- **Status:** Draft hypothesis; belief-based (domain map without user research)
- **Date:** 2026-08-08
- **Audience:** Designers and engineers changing module publish, library scope,
  catalog browse, or nested module execution
- **Document type:** Explanation — Layers of Product Design conceptual model
- **Related:** [modules domain map](modules-domain.md),
  [modules interaction structure](modules-interaction-flow.md),
  [modules surface](modules-surface.md),
  [authentication and workspace tenancy](authentication-and-workspace-tenancy.md),
  [product vocabulary](../../CONTEXT.md)

## Summary

This note decides how Notarius models **reusable subgraphs and workspace
libraries**. It resolves the domain harvest into objects, relationships,
states, and ubiquitous language. It is not a database schema, wireframe, or
browse-UI spec.

**Hypothesis flag:** Built from the belief-based [domain map](modules-domain.md).
Revisit once observed behaviour or research challenges these decisions.

### Core job

Publish a validated workflow building block into a workspace library, call a
pinned release from other graphs, and import a release into another workspace
without live cross-tenant references.

### Material

- [Modules domain map](modules-domain.md)
- Existing product: a module is any in-workspace saved-graph revision that
  validates with Module Input/Output boundaries; tip revisions are
  catalog-visible automatically; discovery is workspace-scoped
- Tenancy: Workspace is the only share boundary; graph copy clears
  cross-workspace module references

### Implicit model being replaced

Today, “having valid boundaries” silently means “appears in the node catalog.”
This model replaces that with an explicit **Module** that enters a workspace
**library** only when someone **publishes a release**.

## Object definitions

### Workspace

**What it is:** The sole collaboration and tenancy boundary; it also hosts that
workspace’s module library.

**Attributes users care about:** name/slug, kind (`personal` or `shared`).

**Relationships:**

- hosts zero or many Modules (as home library)
- owns zero or many Saved graphs

**Actions:** (existing membership and graph actions; no new library aggregate)

**Decision:** “Personal toolkit” and “team library” are roles of a Workspace,
not separate objects. A personal workspace library is private to its owner; a
shared workspace library is visible to members according to existing
capabilities. There is no instance-public or org-wide library in this model.

---

### Saved graph

**What it is:** The durable authoring document (workflow graph + canvas layout).

**Attributes users care about:** name, revision (tip), whether it declares
module boundaries.

**Relationships:**

- belongs to one Workspace
- may be the source graph of zero or one Module
- contains zero or many Module calls

**Actions:** Edit, Checkpoint, Declare module boundaries (add Module Input /
Module Output), Open, Copy into workspace (existing share-by-copy).

**Decision:** A Saved graph with boundaries is only a *candidate* until a Module
is published from it. Boundaries alone do not list it in the library.

---

### Module

**What it is:** A reusable workflow building block belonging to one workspace
library, authored from one source graph.

**Attributes users care about:** name, description, publication state
(`published`, `deprecated`, `withdrawn`).

**Relationships:**

- belongs to one Workspace (as home library)
- authored from one Saved graph (as source graph)
- has one or many Module releases
- offers zero or one current library release (the release shown for Insert)

**Actions:**

- **Publish release** — create a Module release from a chosen source-graph
  revision (usually the tip) after the contract validates; first publish
  creates the Module
- **Deprecate** — keep callable pinned releases; discourage new inserts
- **Withdraw** — remove from the library browse/insert surface; existing
  Module calls keep resolving their pinned releases
- **Open source graph** — open the authoring Saved graph
- **Import into workspace** — copy a chosen Module release into another
  workspace (see action vocabulary)

**Decision:** Module is a first-class object, not “a graph that happens to
validate.” Name and description are module-facing; they may start from the
source graph name but are not required to track tip renames automatically
(implementation may; UX must not surprise callers mid-compose).

---

### Module release

**What it is:** An immutable, callable version of a Module with a fixed
contract, pinned to one source-graph revision.

**Attributes users care about:** release number (the pinned graph revision),
when it was published (if shown), whether it is the current library release.

**Relationships:**

- belongs to one Module
- pins exactly one Saved graph revision of the source graph
- exposes exactly one Module contract

**Actions:** Inspect contract; choose as target when **Upgrading** a Module
call. Users do not edit a release in place — they publish a new release.

**Decision:** Call sites always pin a Module release. “Track latest” is not a
stored mode in this model; staying current means **Upgrade module call**.

---

### Module contract

**What it is:** The callable surface of one Module release: named input and
output module ports with artifact types and requiredness.

**Attributes users care about:** port name, direction (input/output), artifact
type, requiredness, optional description.

**Relationships:**

- belongs to exactly one Module release
- composed of one or many module ports (at least one output)

**Actions:** Inspect (no independent edit — change the source graph and
publish a new release).

**Decision:** Contract is user-meaningful (composers depend on it) so it is
named as an object, even if storage derives it from boundary nodes at publish
time. Port **name** replaces product language “public name” to avoid
visibility polysemy. Boundary operators remain the authoring mechanism.

---

### Module call

**What it is:** A node in a Saved graph that invokes one Module release.

**Attributes users care about:** which Module, which pinned release, wiring to
module ports.

**Relationships:**

- belongs to one Saved graph (as containing graph)
- pins exactly one Module release
- must not target the Module whose source graph is the containing graph
  (no self-call while editing that module’s source)

**Actions:** Insert module call, Upgrade module call, Open source graph (when
the Module’s home workspace is the active workspace), Remove.

---

### Set aside (not objects in this model)

| Noun from harvest | Treatment |
| --- | --- |
| project | Rejected as product term → Workspace + Saved graph |
| library / personal toolkit / shared library | Role of Workspace, not a separate aggregate |
| catalog | Insert/browse *surface* (interaction layer), not an object |
| subgraph | Authoring structure inside a Saved graph, not a durable object |
| building block | Synonym → Module |
| draft | State of source work / unpublished candidate, not an object |
| fork | Outcome of Import, not a persistent type |
| dependency | Relationship via Module call → Module release |
| practitioner | User (existing) |

## Object map

```mermaid
erDiagram
  Workspace ||--o{ SavedGraph : owns
  Workspace ||--o{ Module : "hosts as home library"
  SavedGraph ||--o| Module : "authors as source graph"
  Module ||--|{ ModuleRelease : has
  Module ||--o| ModuleRelease : "offers as current library release"
  ModuleRelease ||--|| ModuleContract : exposes
  ModuleContract ||--|{ ModulePort : includes
  SavedGraph ||--o{ ModuleCall : contains
  ModuleCall }o--|| ModuleRelease : pins
```

Cardinality: `||` exactly one, `o|` zero or one, `|{` one or many, `o{` zero
or many. Labels read left entity → right entity as declared.

## State transitions

### Module publication state

```mermaid
stateDiagram-v2
  [*] --> Published: Publish release
  Published --> Published: Publish release
  Published --> Deprecated: Deprecate
  Deprecated --> Published: Publish release
  Deprecated --> Withdrawn: Withdraw
  Published --> Withdrawn: Withdraw
  Withdrawn --> Published: Publish release
```

| State | Library browse / Insert | Existing Module calls |
| --- | --- | --- |
| Published | Listed; current library release offered | Resolve pinned releases |
| Deprecated | Listed as deprecated; new inserts discouraged | Resolve pinned releases |
| Withdrawn | Hidden from library | Resolve pinned releases |

A Module does not exist until the first successful **Publish release**. A Saved
graph may declare boundaries and still have no Module.

### Source graph readiness (authoring, not Module state)

```mermaid
stateDiagram-v2
  [*] --> NoBoundaries
  NoBoundaries --> BoundariesPresent: Declare module boundaries
  BoundariesPresent --> ContractValid: Contract validates
  BoundariesPresent --> ContractBroken: Contract fails
  ContractValid --> ContractBroken: Edit breaks contract
  ContractBroken --> ContractValid: Edit restores contract
  ContractValid --> ContractValid: Publish release
```

**Publish release** is enabled only from `ContractValid` on the chosen
revision. Broken tips never become library releases; prior releases stay
callable.

### Temporal decisions

| Topic | Decision |
| --- | --- |
| History | Module releases are immutable; callers keep pins |
| Relationship temporality | “Current library release” means the release offered for new inserts *now*; existing calls do not move |
| Deletion | No hard delete in v1: Withdraw (and Deprecate) only. Existing Module calls keep resolving pinned releases; do not destroy releases out from under pins |
| Cross-workspace | **Import** copies a release into the destination workspace as a new source graph + Module; no live cross-workspace Module reference |
| Read model lag | UX requirement: after Publish/Withdraw, the active workspace library reflects the change before the user inserts again — how fresh is an open engineering question |

## Ubiquitous language

### Nouns

| Term | Rejected alternatives | Decision |
| --- | --- | --- |
| Module | building block, component, package, template | Product name for the reusable callable unit |
| Module release | tip-as-catalog-entry, version (alone), checkpoint | Immutable callable pin; number aligns with source graph revision |
| Module contract | API, signature, interface | Callable ports of one release |
| Module port / port name | public name, public port | Avoid “public” = visibility |
| Module call | nest, embed, reference (alone), invoke (UI) | Node that pins a Module release |
| Workspace library | catalog, registry, toolkit (as object) | Published modules hosted by a Workspace |
| Source graph | project, upstream module graph | The Saved graph a Module is authored from |
| Saved graph | project, pipeline, notebook | Keep existing CONTEXT.md term |
| Workspace | project (as tenant) | Keep existing tenancy term |

**Visibility language:** Prefer **published / deprecated / withdrawn** and
**personal vs shared workspace**. Do not use **public/private module** for
library scope — “private” already collides with personal workspace and
“public” collides with port names and internet sharing.

### Verbs

| Verb | Applies to | Rejected alternatives | Decision |
| --- | --- | --- | --- |
| Publish release | Module (creates Module on first success) | Share, expose, make public, sync tip | Deliberate offer of a validated revision to the workspace library |
| Deprecate | Module | Soft-delete, hide | Still visible as legacy; discourage new inserts |
| Withdraw | Module | Unpublish, remove, delete | Hide from library; keep pins resolving |
| Import into workspace | Module release → destination Workspace | Install, link, share across, reference | Copy-by-value into a new source graph + Module; no live link |
| Insert module call | Saved graph + Module release | Add node (when meaning module), nest, embed | Places a Module call pinned to a release |
| Upgrade module call | Module call | Update, sync, track latest | Repin to a newer Module release |
| Declare module boundaries | Saved graph | Make module (premature), expose ports | Authoring step; does not publish |
| Open source graph | Module or Module call | Edit module, drill in | Navigate to authoring document |

**Share** remains the existing workspace/graph meaning (membership or
copy-graph-into-workspace). It must not mean “publish module.”

**Edit** on a Module is rejected as a CTA — users **Open source graph**, edit
the Saved graph, then **Publish release**.

## Resolved decisions (binding)

1. **No hard delete (v1):** Retirement is **Deprecate** and **Withdraw from
   library**. Owners do not destroy Modules or releases out from under pins.
   Deprecated/withdrawn Modules’ existing Module calls remain executable as long
   as the underlying node/operator code can execute the pinned release.
2. **Who may Publish / Deprecate / Withdraw:**
   - **Publish release** — Editor + Owner (`publish_module`)
   - **Deprecate** and **Withdraw** — Owner only (`manage_module_library`);
     soft stewardship matches Owner-only withdraw, not Editor
3. **Insert release choice:** Composers may pick an older Module release at
   insert time (not only the current library release). Upgrade still repins to a
   chosen release (typically a newer one).
4. **Library entry points:** Workbench **and** workspace graphs overview.
5. **v1 scope:** Full breadboard (C) — Publish + Add node library listing +
   Workspace library manage (deprecate/withdraw) + Import into workspace. No
   instance-public modules.

## Remaining open questions

1. **Rename propagation:** When the source graph is renamed, does the Module
   name follow, stay independent, or prompt on next publish?
2. **Imported Module lineage:** Does the destination Module record that it was
   imported from another workspace’s release, or is lineage discarded like
   graph copy today? (v1 UI does not promise lineage.)
3. **Nested optional-input rules** and execution semantics stay as today’s
   runtime contracts; this model does not reopen them.

## Next step

Interaction breadboards are in [modules-interaction-flow.md](modules-interaction-flow.md).
Next: `/layers-surface` or implementation against a stable model.

This model was built without domain research — it is a hypothesis. Plan to
revisit it once you have evidence.
