# Grafy SQL Plugin

Self-contained System publication input for the existing `sql` entry point.
The project vendors the exact Grafy SDK wheel referenced by `uv.lock`; it does
not resolve dependencies through the monorepo Workspace.

The published Plugin identity remains `external.sql`. SQL remains isolated-only
because query validation is not a process sandbox.
