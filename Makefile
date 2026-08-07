.PHONY: install install-all install-gis install-llm install-ocr install-sql api api-gis api-llm api-ocr api-sql mcp web test lint typecheck contract build check smoke db-upgrade db-downgrade db-current db-history db-revision docker-up docker-down keycloak-up keycloak-down bootstrap-oidc-owner

-include .env
export

install:
	uv sync
	npm --prefix apps/web ci

install-all:
	uv sync --extra gis --extra llm --extra ocr --extra sql
	npm --prefix apps/web ci

install-ocr:
	uv sync --extra ocr
	npm --prefix apps/web ci

install-gis:
	uv sync --extra gis
	npm --prefix apps/web ci

install-llm:
	uv sync --extra llm
	npm --prefix apps/web ci

install-sql:
	uv sync --extra sql
	npm --prefix apps/web ci

api: db-upgrade
	uv run --exact --no-dev --package notarius-api uvicorn notarius_api.main:app --reload --host 0.0.0.0 --port 8000

api-ocr: db-upgrade
	uv run --exact --no-dev --extra ocr uvicorn notarius_api.main:app --reload --host 0.0.0.0 --port 8000

api-gis: db-upgrade
	uv run --exact --no-dev --extra gis uvicorn notarius_api.main:app --reload --host 0.0.0.0 --port 8000

api-llm: db-upgrade
	uv run --exact --no-dev --extra llm uvicorn notarius_api.main:app --reload --host 0.0.0.0 --port 8000

api-sql: db-upgrade
	uv run --exact --no-dev --extra sql uvicorn notarius_api.main:app --reload --host 0.0.0.0 --port 8000

api-all: db-upgrade
	uv run --exact --no-dev --extra llm --extra gis --extra ocr --extra sql uvicorn notarius_api.main:app --reload --host 0.0.0.0 --port 8000

mcp:
	@echo "MCP is mounted on the API at /mcp (stateless Streamable HTTP)."
	@echo "Start the API (make api), create a workspace-bound PAT, then connect"
	@echo "an MCP client to http://127.0.0.1:8000/mcp with Authorization: Bearer <token>."
	@exit 1

prefect:
	.venv/bin/prefect server start

web:
	npm --prefix apps/web run dev

test:
	uv run --extra gis --extra llm --extra ocr --extra sql pytest
	npm --prefix apps/web test

lint:
	uv run ruff check apps/api/src apps/mcp/src libs/core/src libs/persistence/src libs/storage/src plugins/gis/src plugins/llm/src plugins/ocr/src plugins/sql/src infra/db/migrations scripts tests
	npm --prefix apps/web run lint

typecheck:
	uv run --extra gis --extra llm --extra ocr --extra sql basedpyright
	npm --prefix apps/web run typecheck

contract:
	npm --prefix apps/web run check:api

build:
	npm --prefix apps/web run build

check: test lint typecheck contract build

smoke:
	uv run --extra ocr python scripts/smoke_workbench.py

db-upgrade:
	uv run --no-dev alembic upgrade head

db-downgrade:
	uv run --no-dev alembic downgrade -1

db-current:
	uv run --no-dev alembic current

db-history:
	uv run --no-dev alembic history --verbose

db-revision:
	@test -n "$(message)" || (echo "message is required" && exit 1)
	uv run --no-dev alembic revision --autogenerate -m "$(message)"

docker-up:
	docker compose -f infra/docker/compose.yaml up --build

docker-down:
	docker compose -f infra/docker/compose.yaml down

keycloak-up:
	docker compose -f infra/docker/compose.keycloak.yaml up -d --wait

keycloak-down:
	docker compose -f infra/docker/compose.keycloak.yaml down

bootstrap-oidc-owner:
	uv run --no-dev notarius-admin bootstrap-oidc-owner \
		--issuer $${NOTARIUS_OIDC_ISSUER:?set NOTARIUS_OIDC_ISSUER} \
		--subject $${NOTARIUS_OIDC_BOOTSTRAP_SUBJECT:?set NOTARIUS_OIDC_BOOTSTRAP_SUBJECT}
