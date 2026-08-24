# Run Grafy with the Compose same-origin gateway

This Compose project runs four processes:

1. `migrate` applies Alembic migrations and exits.
2. `api` runs one FastAPI process with the GIS, LLM, and OCR plugins installed.
   The SQL plugin is deliberately excluded: user-authored SQL must move to a
   separate networkless, least-privileged worker before production use.
3. `web` runs the minimal Next.js standalone server.
4. `gateway` is the only Grafy application listener (`127.0.0.1:8080` by
   default). It serves plain HTTP and does not terminate TLS.

Web and API stay on the Compose network. Do not scale the `api` service and
do not set `WEB_CONCURRENCY` or Uvicorn `--workers` above 1.

Production OIDC still requires an operator-supplied TLS endpoint for the exact
registered public origin. This repository does not provide that TLS or the
complete forwarded-header design between an additional edge proxy and the
Compose gateway.

## Prepare the VPS

Install Docker Engine with the Compose plugin, clone the repository, and create
the production environment file:

```bash
cp infra/docker/.env.production.example .env.production
openssl rand -base64 32
openssl rand -base64 32
openssl rand -base64 32
```

Put the three independently generated values in
`GRAFY_CREDENTIAL_ENCRYPTION_KEY`, `GRAFY_COMMAND_HMAC_KEY`, and
`GRAFY_OIDC_AUTH_WRAPPING_KEY`. Set `GRAFY_OIDC_ISSUER`,
`GRAFY_OIDC_CLIENT_ID`, and the client secret only when the provider uses a
confidential client. Then set the exact public HTTPS origin in
`GRAFY_PUBLIC_ORIGIN` and `GRAFY_CORS_ORIGINS`. Compose refuses to render
the deployment while a required value is unset or empty. Register the callback
`${GRAFY_PUBLIC_ORIGIN}/api/v1/auth/oidc/callback`
with the identity provider before opening login.

`GRAFY_DOCKER_DATABASE_URL` is deliberately separate from the local
`GRAFY_DATABASE_URL`. This prevents a development `.env` file with a relative
SQLite path from sending migrations to a container-local, non-persistent file.

Keep `.env.production` outside version control and restrict it to the deployment
user:

```bash
chmod 600 .env.production
```

## Start the application

```bash
docker compose \
  --env-file .env.production \
  -f infra/docker/compose.yaml \
  up --build --detach
```

Workspace Plugin execution is deliberately absent from the base Compose file.
To enable the local Docker sandbox adapter, include the opt-in override:

```bash
docker compose \
  --env-file .env.production \
  -f infra/docker/compose.yaml \
  -f infra/docker/compose.plugin-runtime.yaml \
  up --build --detach
```

The override mounts `/var/run/docker.sock` into the API container. Possession
of that socket is effectively root-equivalent authority over the Docker host:
an API-process compromise can create privileged containers or mount host
paths. The socket is never mounted into a Plugin container. Plugin containers
run with no network, a read-only root filesystem, UID/GID 65532, no Linux
capabilities, `no-new-privileges`, seccomp, and bounded tmpfs/resource limits.
This protects against mistakes and ordinary unsafe Plugin code; it is not a
hostile-public-tenant kernel boundary. Keep the API private, use one API owner,
and omit the override when Workspace Plugin execution is not required. A
narrow same-VPS runner can replace this socket-bearing adapter later without
changing the provider-neutral invocation contract.

Tune `GRAFY_MAX_ACTIVE_EXECUTIONS`, `GRAFY_MAX_PENDING_GRAPHS`,
`GRAFY_MAX_ACTIVE_PLUGIN_INVOCATIONS`, `GRAFY_MAX_LIVE_PLUGIN_SANDBOXES`, and
`GRAFY_MAX_DISTINCT_PLUGIN_RELEASES_PER_GRAPH` together using representative
CPU, memory, disk, and artifact workloads. The distinct-release limit must not
exceed the live-sandbox limit.

`GRAFY_PLUGIN_ROOTS` and `GRAFY_PLUGIN_AUTHORING_ROOT` default to
`/data/plugins` in the production example so coding-agent working copies live
on the durable `grafy-data` volume. Run `grafy plugin scaffold`, `review`, and
`publish-reviewed` inside the API container with an authorized Workspace owner
UUID. The scaffold builds the SDK wheel from the deployment-owned
`GRAFY_PLUGIN_SDK_PROJECT`; it never writes build output into that SDK source.

Check readiness and logs:

```bash
docker compose \
  --env-file .env.production \
  -f infra/docker/compose.yaml \
  ps

docker compose \
  --env-file .env.production \
  -f infra/docker/compose.yaml \
  logs --follow gateway api web
```

Compose waits for migration and Prefect health before starting the API, waits
for the API before starting the web server, and waits for both before starting
the gateway.

## SSH tunnel for plain-HTTP diagnostics

On a host such as `ai-test-ihpan`, forward the loopback gateway locally for
unauthenticated HTTP diagnostics:

```bash
ssh -N -L 8080:127.0.0.1:8080 ai-test-ihpan
```

The local end of this tunnel is `http://127.0.0.1:8080`. SSH protects the
transport, but it does not turn the gateway into an HTTPS server. Do not point
an HTTPS client at this port. OIDC login, Secure cookies, and an authenticated
production smoke must go through a separately configured TLS endpoint for the
registered public origin.

## Prefect

The Prefect image is pinned to the same `3.6.21` version used by the Grafy
API. Grafy submits local flows to `http://prefect:4200/api`; execution still
happens inside the API container, so this topology does not require a Prefect
worker.

Prefect stores its SQLite database in the separate `prefect-data` volume.
Its dashboard port is bound to `127.0.0.1:4200`. Access it without adding a
public gateway route:

```bash
ssh -L 4200:127.0.0.1:4200 deploy@your-vps
```

Then open `http://127.0.0.1:4200`.

## Same-origin routing contract

Checked configuration lives at `infra/docker/gateway/nginx.conf`:

The checked gateway listens on plain HTTP port 8080. Its forwarding-header
rules describe the direct gateway-to-API hop only; they overwrite incoming
forwarded headers with the gateway's direct peer and scheme. Do not treat this
file as TLS termination or as a complete two-proxy configuration.

- `/` proxies to the Next.js web process.
- `/api/` proxies to the API process with a trailing-slash `proxy_pass` that
  strips only `/api`, so public `/api/v1/...` becomes FastAPI `/v1/...`.
- Preserve WebSocket upgrade on graph-room paths and disable proxy buffering
  for SSE and long-lived responses.
- For this direct plain-HTTP hop, overwrite rather than append forwarding
  headers:

  ```nginx
  proxy_set_header Host $host;
  proxy_set_header X-Forwarded-For $remote_addr;
  proxy_set_header X-Forwarded-Proto $scheme;
  ```

  The Compose network uses subnet `172.30.0.0/24` with bridge gateway
  `172.30.0.1`. Because the gateway container receives a dynamic address in
  that subnet, Uvicorn trusts the configured subnet CIDR rather than the bridge
  gateway address. If the subnet overlaps another Docker network, set
  `GRAFY_DOCKER_SUBNET` and `GRAFY_DOCKER_GATEWAY` together; never use
  `*` as the trusted range.

Browser traffic uses the opaque Grafy session cookie after OIDC login. The
operator-supplied public endpoint must terminate TLS for that origin before
production login.

## One API owner

Collaboration rooms, journals, presence, and shared execution assume exactly one
FastAPI owner process. Compose sets `GRAFY_REQUIRE_SINGLE_API_OWNER=true` and
`WEB_CONCURRENCY=1`. On startup the API acquires an exclusive lock file under
the data workspace; a second owner fails closed. Do not add replicas, Uvicorn
`--workers`, or shared room pubsub across processes until a separate design is
accepted.

## First-owner bootstrap

After migrate and before ordinary login:

```bash
docker compose \
  --env-file .env.production \
  -f infra/docker/compose.yaml \
  exec api \
  .venv/bin/grafy-admin bootstrap-oidc-owner \
  --subject "<first-owner-subject>"
```

The command reads `GRAFY_OIDC_ISSUER` from the API container's configured
environment. Its optional `--issuer` argument is only an equality assertion;
it is not needed for this Compose invocation. In particular, `--env-file` does
not export variables into the host shell.

The matching identity's first valid OIDC callback consumes the mapping and owns
the migrated `local` workspace. Do not invent local passwords or anonymous
owners.

## Persistence, backup, and rollback

The `grafy-data` volume contains the Grafy SQLite database, staged
uploads, and local artifact objects. The `prefect-data` volume contains Prefect
orchestration state. Treat migrations as a maintenance window for SQLite:

Each staged file is limited to 64 MiB. The gateway allows 65 MiB request bodies
to leave room for multipart framing, while the API enforces the exact file-byte
limit. Staged files are durable graph inputs: saved graphs retain their upload
keys and reread the files on later runs. There is therefore no age-based cleanup
or workspace quota yet; monitor the data volume until upload promotion/reference
tracking can make deletion and transactional quota reservation safe.

1. stop gateway and API traffic (and Prefect if it might touch shared state);
2. confirm one API owner and no active execution;
3. checkpoint/truncate the SQLite WAL and create a consistent backup with the
   SQLite backup API or a stopped-volume snapshot;
4. back up the complete Grafy data volume and separately protect encryption,
   auth-wrapping, command-HMAC, and OIDC client secrets;
5. run `PRAGMA integrity_check` and restore the backup into a scratch location
   before upgrading.

Backup while the stack is stopped:

```bash
docker run --rm \
  --volume grafy_grafy-data:/source:ro \
  --volume "$PWD/backups:/backup" \
  alpine \
  tar -czf /backup/grafy-data.tgz -C /source .

docker run --rm \
  --volume grafy_prefect-data:/source:ro \
  --volume "$PWD/backups:/backup" \
  alpine \
  tar -czf /backup/prefect-data.tgz -C /source .
```

Restore only while the Compose project is stopped. Roll back by restoring the
volume backup, checking out the previous known-good image/git revision, and
bringing the stack up without re-running forward migrations against restored
data unless that revision's Alembic head matches. Use external PostgreSQL and
S3-compatible storage before introducing multiple API replicas; that topology
is out of scope for the current release gate.

## Phase 7 release checklist (operator)

Automatable pieces already covered in CI/unit tests: one-owner fence, Compose
gateway/proxy contract, and two-session API/WS collaboration acceptance
(`tests/unit/api/test_collaboration_acceptance.py`). The items below still need
a human on a real host/IdP/data copy.

- [ ] Exact `GRAFY_PUBLIC_ORIGIN` and OIDC callback registered
- [ ] Secrets generated and backed up outside the data volume
- [ ] `bootstrap-oidc-owner` mapping written for the intended first subject
- [ ] Plain HTTP gateway serves `/`, `/api/v1` on loopback `:8080` only
- [ ] Separate TLS endpoint and both proxy hops validated for the public origin
- [ ] One API replica / one worker; second owner fails startup
- [ ] Backup + integrity check rehearsed on a copy of realistic data
- [ ] Authenticated browser smoke through the gateway
- [ ] Live two-browser collaboration smoke (converge, presence, shared run, revoke)
- [ ] Collaboration drain before upgrade; restore rehearsed

## Stop or upgrade

```bash
docker compose \
  --env-file .env.production \
  -f infra/docker/compose.yaml \
  down

git pull --ff-only

docker compose \
  --env-file .env.production \
  -f infra/docker/compose.yaml \
  up --build --detach
```

Do not add `--volumes` to `down` unless deleting all persisted Grafy data is
intentional.
