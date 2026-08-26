# Local Keycloak for Grafy

Dev-only OpenID Connect provider, following the same
`keycloak.localhost:8081` pattern as Stacking.

## Start

```bash
just keycloak-up
```

Admin console: [http://keycloak.localhost:8081](http://keycloak.localhost:8081)
(`admin` / `admin`).

Imported realm `grafy` includes:

| Item | Value |
| --- | --- |
| Issuer | `http://keycloak.localhost:8081/realms/grafy` |
| Client | `grafy-web` (public, Authorization Code + PKCE S256) |
| Callback | `http://localhost:3000/api/v1/auth/oidc/callback` |
| Test user | `owner` / `owner` (`owner@grafy.local`) |
| Subject | `11111111-1111-4111-8111-111111111111` |

Modern browsers resolve `*.localhost` to `127.0.0.1`; no `/etc/hosts` edit is
required.

## Start Grafy

With API env configured (see root `.env.example`):

```bash
just db-upgrade
```

Then `just api` and `just web`, open `http://localhost:3000`, and sign in as
`owner` / `owner`. Grafy creates the user's personal workspace during the first
successful OIDC callback.

## Reset realm import

Realm import only runs on an empty Keycloak data volume:

```bash
just keycloak-down
docker volume rm grafy-keycloak_keycloak-data
just keycloak-up
```
