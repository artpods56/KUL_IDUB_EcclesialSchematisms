.PHONY: install api web test lint typecheck contract build check smoke docker-up docker-down

-include .env
export

install:
	uv sync
	npm --prefix apps/web ci

api:
	uv run --package notarius-api uvicorn notarius_api.main:app --reload --host 0.0.0.0 --port 8000

web:
	npm --prefix apps/web run dev

test:
	uv run pytest

lint:
	uv run ruff check apps/api/src libs/core/src libs/storage/src scripts tests
	npm --prefix apps/web run lint

typecheck:
	uv run basedpyright
	npm --prefix apps/web run typecheck

contract:
	npm --prefix apps/web run check:prototype-api

build:
	npm --prefix apps/web run build

check: test lint typecheck contract build

smoke:
	uv run --package notarius-api python scripts/smoke_prototype.py

docker-up:
	docker compose -f infra/docker/compose.yaml up --build

docker-down:
	docker compose -f infra/docker/compose.yaml down
