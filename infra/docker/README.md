# Deploy Notarius behind the same-origin gateway

This Compose project runs five processes:

1. `migrate` applies Alembic migrations and exits.
2. `prefect` runs the pinned self-hosted Prefect server and background services.
3. `api` runs one FastAPI process with every bundled plugin installed.
4. `web` runs the minimal Next.js standalone server.
5. `gateway` is the only Notarius public listener (`127.0.0.1:8080` by default).

Web and API stay on the Compose network. Prefect publishes a separate loopback
dashboard port for SSH-tunneled ops access. Do not scale the `api` service and
do not set `WEB_CONCURRENCY` or Uvicorn `--workers` above 1.

## Prepare the VPS

Install Docker Engine with the Compose plugin, clone the repository, and create
the production environment file:

```bash
cp infra/docker/.env.production.example .env.production
openssl rand -base64 32
```

Put generated secrets in `NOTARIUS_CREDENTIAL_ENCRYPTION_KEY` and
`NOTARIUS_COMMAND_HMAC_KEY`, then set the exact public HTTPS origin in
`NOTARIUS_PUBLIC_ORIGIN` and `NOTARIUS_CORS_ORIGINS`. Register the OIDC
callback
`${NOTARIUS_PUBLIC_ORIGIN}/api/v1/auth/oidc/callback`
with the identity provider before opening login.

`NOTARIUS_DOCKER_DATABASE_URL` is deliberately separate from the local
`NOTARIUS_DATABASE_URL`. This prevents a development `.env` file with a relative
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

## SSH tunnel to the test host

On a host such as `ai-test-ihpan`, bind gateway port 8080 only to remote
loopback and forward it locally:

```bash
ssh -N -L 8080:127.0.0.1:8080 ai-test-ihpan
```

Open the exact registered HTTPS hostname and port, never an HTTP or
`127.0.0.1` alias. The tunnel protects the network hop; TLS supplies the stable
OIDC origin and Secure-cookie contract.

## Prefect

The Prefect image is pinned to the same `3.6.21` version used by the Notarius
API. Notarius submits local flows to `http://prefect:4200/api`; execution still
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

- `/` proxies to the Next.js web process.
- `/api/` proxies to the API process with a trailing-slash `proxy_pass` that
  strips only `/api`, so public `/api/v1/...` becomes FastAPI `/v1/...`.
- `/mcp` proxies to the same API process without rewriting the path.
- Preserve WebSocket upgrade on graph-room paths and disable proxy buffering
  for SSE and MCP long-lived responses.
- Overwrite, do not append, forwarding headers:

  ```nginx
  proxy_set_header Host $host;
  proxy_set_header X-Forwarded-For $remote_addr;
  proxy_set_header X-Forwarded-Proto $scheme;
  ```

  The Compose network uses subnet `172.30.0.0/24` with gateway `172.30.0.1`,
  and the API derives its trusted forwarded-header peer from that same gateway.
  If the subnet overlaps another Docker network, set
  `NOTARIUS_DOCKER_SUBNET` and `NOTARIUS_DOCKER_GATEWAY` together; never use
  `*` as the trusted peer.

Browser traffic uses the opaque Notarius session cookie after OIDC login. MCP
clients use a workspace-bound PAT in `Authorization: Bearer` against
`https://<origin>/mcp`. Terminate TLS for the registered public origin before
production login.

## One API owner

Collaboration rooms, journals, presence, and shared execution assume exactly one
FastAPI owner process. Compose sets `NOTARIUS_REQUIRE_SINGLE_API_OWNER=true` and
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
  .venv/bin/notarius-admin bootstrap-oidc-owner \
  --issuer "$NOTARIUS_OIDC_ISSUER" \
  --subject "<first-owner-subject>"
```

The matching identity's first valid OIDC callback consumes the mapping and owns
the migrated `local` workspace. Do not invent local passwords or anonymous
owners.

## Persistence, backup, and rollback

The `notarius-data` volume contains the Notarius SQLite database, staged
uploads, and local artifact objects. The `prefect-data` volume contains Prefect
orchestration state. Treat migrations as a maintenance window for SQLite:

1. stop gateway and API traffic (and Prefect if it might touch shared state);
2. confirm one API owner and no active execution;
3. checkpoint/truncate the SQLite WAL and create a consistent backup with the
   SQLite backup API or a stopped-volume snapshot;
4. back up the complete Notarius data volume and separately protect encryption,
   auth-wrapping, command-HMAC, and OIDC client secrets;
5. run `PRAGMA integrity_check` and restore the backup into a scratch location
   before upgrading.

Backup while the stack is stopped:

```bash
docker run --rm \
  --volume notarius_notarius-data:/source:ro \
  --volume "$PWD/backups:/backup" \
  alpine \
  tar -czf /backup/notarius-data.tgz -C /source .

docker run --rm \
  --volume notarius_prefect-data:/source:ro \
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

## MCP on the VPS

Streamable HTTP MCP is mounted on the API process at `/mcp` under the same
HTTPS gateway as `/api/v1`. Create a workspace-bound personal access token in
the browser UI, then point an MCP client at:

`https://<notarius-test-host>:8080/mcp`

with `Authorization: Bearer <token>`. First delivery is stateless: every request
re-resolves the PAT. There is no separate stdio MCP process or ambient API
credential. `executions:run` MCP tools are deferred; shared runs stay on the
browser/REST path.

## Phase 7 release checklist (operator)

Automatable pieces already covered in CI/unit tests: one-owner fence, Compose
gateway/proxy contract, and two-session API/WS collaboration acceptance
(`tests/unit/api/test_collaboration_acceptance.py`). The items below still need
a human on a real host/IdP/data copy.

- [ ] Exact `NOTARIUS_PUBLIC_ORIGIN` and OIDC callback registered
- [ ] Secrets generated and backed up outside the data volume
- [ ] `bootstrap-oidc-owner` mapping written for the intended first subject
- [ ] Gateway serves `/`, `/api/v1`, `/mcp` on loopback `:8080` only
- [ ] One API replica / one worker; second owner fails startup
- [ ] Backup + integrity check rehearsed on a copy of realistic data
- [ ] Authenticated browser smoke and PAT MCP smoke through the gateway
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

Do not add `--volumes` to `down` unless deleting all persisted Notarius data is
intentional.
