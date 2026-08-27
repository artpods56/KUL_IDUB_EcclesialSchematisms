# Publish Plugins from the CLI

Use a Workspace PAT for Workspace publication and a platform access token for
global publication. Run database migrations before issuing platform tokens.

## Configure an interactive credential

Create a Workspace PAT in Grafy with the `publish_plugin` scope, or ask a
deployment operator for a platform token. Then store and verify it:

```console
grafy auth login
grafy auth status
```

Grafy stores the token in the operating-system keychain. For CI, write the
token to a protected file and point Grafy to the file:

```console
export GRAFY_TOKEN_FILE=/run/secrets/grafy-token
```

The file must contain one token and may end with one newline. Do not put the
token itself in an environment variable or command argument.

## Publish to the token's Workspace

Check the directory before publication:

```console
grafy plugin check plugins/llm
grafy plugin publish plugins/llm --slug llm
```

Grafy derives the Workspace and User from the PAT. There is no Workspace or
actor argument.

## Issue a platform token

Run this on the deployment host:

```console
grafy admin platform-token create \
  --principal release-bot \
  --label "Plugin release automation" \
  --scope plugin.publish_global \
  --scope plugin.promote_global \
  --expires-at 2026-12-31T23:59:59Z
```

The command prints the token once. Store it with `grafy auth login` or in the
CI credential file. List and revoke tokens with:

```console
grafy admin platform-token list
grafy admin platform-token revoke TOKEN_ID
```

## Publish and promote globally

Global publication verifies the candidate in the configured publisher image
and appends an inactive release:

```console
grafy plugin publish plugins/llm \
  --slug llm \
  --global \
  --sandbox-image grafy-plugin-publisher:current
```

Activate the returned revision explicitly:

```console
grafy plugin promote llm@3
```

Use `--if-generation N` only when an automation job must reject a concurrent
selection change. Ordinary interactive promotion reads and advances the current
generation automatically.

Remove the interactive credential when it is no longer needed:

```console
grafy auth logout
```
