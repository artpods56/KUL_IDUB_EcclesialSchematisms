# ADR 0007: Builtin families are application code, not Plugin releases

- **Status:** Accepted
- **Amends:** ADR 0005
- **Supersedes:** ADR 0004's bundled / optional / published System distribution
  and host-eligible Plugin execution model

A Plugin is always an independently published, installed package. Builtin node
families ship with the workbench application, execute in-process, and identify
as `kind=builtin` plus the deployment build digest. They are not Plugin
releases and do not carry a release pin.

PluginDistribution (`bundled`, `optional`, `published`) is removed.
System versus Workspace remains installation scope only. Published Plugins run
in isolated workers; host-eligible in-process Plugin execution is no longer a
product path.
