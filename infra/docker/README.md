# Deploy Notarius behind an existing Nginx

This Compose project runs four processes:

1. `migrate` applies Alembic migrations and exits.
2. `prefect` runs the pinned self-hosted Prefect server and background services.
3. `api` runs one FastAPI process with every bundled plugin installed.
4. `web` runs the minimal Next.js standalone server.

The web, API, and Prefect ports bind to `127.0.0.1` by default. They are not
reachable from another machine unless the host's Nginx proxies them.

## Prepare the VPS

Install Docker Engine with the Compose plugin, clone the repository, and create
the production environment file:

```bash
cp infra/docker/.env.production.example .env.production
openssl rand -base64 32
```

Put the generated encryption key in `NOTARIUS_CREDENTIAL_ENCRYPTION_KEY`, then
set the real public origin in both `NEXT_PUBLIC_NOTARIUS_API_URL` and
`NOTARIUS_CORS_ORIGINS`.

`NOTARIUS_DOCKER_DATABASE_URL` is deliberately separate from the local
`NOTARIUS_DATABASE_URL`. This prevents a development `.env` file with a relative
SQLite path from sending migrations to a container-local, non-persistent file.

The browser API URL is a build argument. Rebuild the web image whenever that
URL changes. Keep `.env.production` outside version control and restrict it to
the deployment user:

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
  logs --follow api web
```

Compose waits for the migration and Prefect health check before starting the
API, and waits for the API health check before starting the web server.

## Prefect

The Prefect image is pinned to the same `3.6.21` version used by the Notarius
API. Notarius submits local flows to `http://prefect:4200/api`; execution still
happens inside the API container, so this topology does not require a Prefect
worker.

Prefect stores its SQLite database in the separate `prefect-data` volume.
Its dashboard port is bound to `127.0.0.1:4200`. Access it without adding a
public Nginx route:

```bash
ssh -L 4200:127.0.0.1:4200 deploy@your-vps
```

Then open `http://127.0.0.1:4200`.

## Nginx routing contract

With `NEXT_PUBLIC_NOTARIUS_API_URL=https://notarius.example.com/api`, configure
Nginx so:

- `/` proxies to `http://127.0.0.1:3000`.
- `/api/` proxies to `http://127.0.0.1:8000/`.
- The trailing slash on `proxy_pass` strips `/api` before FastAPI receives the
  request.
- Proxy buffering is disabled for execution event streams.
- Request body limits are high enough for the table, image, and GIS files you
  intend to upload.

Notarius does not yet provide application-level user authentication. Protect
both routes at Nginx with your existing VPN, SSO, or authentication policy.
Do not expose ports 3000 or 8000 publicly.

## Persistence and backups

The `notarius-data` volume contains the Notarius SQLite database, staged
uploads, and local artifact objects. The `prefect-data` volume contains Prefect
orchestration state. Back up both volumes while the stack is stopped:

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

Restore only while the Compose project is stopped. Keep one API replica when
using SQLite and local artifact storage. Use an external PostgreSQL database
and S3-compatible storage before introducing multiple API replicas.

## MCP on the VPS

The API image also contains the Notarius MCP package. Run its stdio server over
an SSH session when needed:

```bash
docker compose \
  --env-file .env.production \
  -f infra/docker/compose.yaml \
  exec -T api .venv/bin/notarius-mcp
```

Inside the API container, `NOTARIUS_MCP_API_URL` points to
`http://127.0.0.1:8000`; the MCP server does not need a public API route.

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
