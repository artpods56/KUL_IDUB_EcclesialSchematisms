.PHONY: install install-ocr api api-ocr web test lint typecheck contract build check smoke db-upgrade db-downgrade db-current db-history db-revision docker-up docker-down

-include .env
export

install:
	uv sync
	npm --prefix apps/web ci

install-ocr:
	uv sync --extra ocr
	npm --prefix apps/web ci

api: db-upgrade
	uv run --exact --no-dev --package notarius-api uvicorn notarius_api.main:app --reload --host 0.0.0.0 --port 8000

api-ocr: db-upgrade
	uv run --exact --no-dev --extra ocr uvicorn notarius_api.main:app --reload --host 0.0.0.0 --port 8000

web:
	npm --prefix apps/web run dev

test:
	uv run --extra ocr pytest
	npm --prefix apps/web test

lint:
	uv run ruff check apps/api/src libs/core/src libs/persistence/src libs/storage/src plugins/ocr/src infra/db/migrations scripts tests
	npm --prefix apps/web run lint

typecheck:
	uv run --extra ocr basedpyright
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
