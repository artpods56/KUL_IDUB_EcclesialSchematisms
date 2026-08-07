# Local Keycloak for Notarius

Dev-only OpenID Connect provider, following the same
`keycloak.localhost:8081` pattern as Stacking.

## Start

```bash
make keycloak-up
```

Admin console: [http://keycloak.localhost:8081](http://keycloak.localhost:8081)
(`admin` / `admin`).

Imported realm `notarius` includes:

| Item | Value |
| --- | --- |
| Issuer | `http://keycloak.localhost:8081/realms/notarius` |
| Client | `notarius-web` (public, Authorization Code + PKCE S256) |
| Callback | `http://localhost:3000/api/v1/auth/oidc/callback` |
| Test user | `owner` / `owner` (`owner@notarius.local`) |
| Subject | `11111111-1111-4111-8111-111111111111` |

Modern browsers resolve `*.localhost` to `127.0.0.1`; no `/etc/hosts` edit is
required.

## Bootstrap Notarius owner

With API env configured (see root `.env.example`):

```bash
make db-upgrade
uv run notarius-admin bootstrap-oidc-owner \
  --issuer http://keycloak.localhost:8081/realms/notarius \
  --subject 11111111-1111-4111-8111-111111111111
```

Then `make api` and `make web`, open `http://localhost:3000`, and sign in as
`owner` / `owner`.

## Reset realm import

Realm import only runs on an empty Keycloak data volume:

```bash
make keycloak-down
docker volume rm notarius-keycloak_keycloak-data
make keycloak-up
```
