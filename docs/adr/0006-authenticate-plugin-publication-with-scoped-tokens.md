# ADR 0006: Authenticate Plugin publication with scoped tokens

- **Status:** Accepted

## Context

Plugin publication previously accepted Workspace ids and actor references as
CLI arguments. Those values described who the caller claimed to be; they did
not authenticate the caller. Global operations also reused an untyped platform
actor string, while Workspace operations depended on a User UUID supplied next
to it.

Publication needs two authorities. A Workspace User may publish only within the
Workspace bound to a personal access token. A deployment operator or CI job may
publish, promote, or revoke global Plugins without being a Workspace User.

## Decision

The CLI authenticates Workspace publication with a `PersonalAccessToken` whose
effective capabilities are the intersection of its scopes and the User's
current membership. The token must include `publish_plugin`. The authenticated
principal supplies both `workspace_id` and `user_id`.

Global operations authenticate with a separate `PlatformAccessToken`. Its
principal reference is the operation actor, and its scopes are limited to
`plugin.publish_global`, `plugin.promote_global`, and
`plugin.revoke_global`.

Interactive credentials are stored in the operating-system keychain after
`grafy auth login`. Non-interactive callers provide a protected file path in
`GRAFY_TOKEN_FILE`. The CLI does not accept a raw token argument or token-value
environment variable. Raw bearer values are parsed and hashed in the CLI
credential adapter; domain and persistence objects contain only digests and
safe references.

All operator commands use the `grafy admin` command group. There is no separate
`grafy-admin` executable.

## Consequences

- Changing or revoking Workspace membership immediately narrows or removes PAT
  publication authority.
- A global automation token cannot act as a Workspace User, and a PAT cannot
  perform global operations.
- Platform operators can rotate and independently scope publishing, promotion,
  and revocation credentials.
- The CLI no longer accepts `--actor` or `--workspace`; callers must
  authenticate with the credential that owns those values.
- Keychain access is deployment-specific, based on a sanitized database
  identity. CI files remain the caller's responsibility to protect.
