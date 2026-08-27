# ADR 0005: Separate Plugin releases from installations

- **Status:** Accepted

A Plugin release is one immutable, scope-neutral package and OCI artifact. An
append-only Plugin installation assigns that release to System scope or one
Workspace and owns the execution policy, System distribution metadata, and
installing actor. A mutable Plugin selection points to the current installation
for one scoped family.

This split lets one verified release run globally or in specific Workspaces
without rebuilding it or rewriting immutable release data. Installation history
remains append-only so older graph pins still resolve after a selection moves.
Changing visibility creates an installation in another scope. It never changes
the release row.

The previous model put scope and policy on each release. That model made the OCI
identity, object path, publisher workflow, and runtime lookup depend on where the
package was first published. A single mutable installation row was also rejected
because it would lose proof that an older pinned release had been installed in
that scope.
