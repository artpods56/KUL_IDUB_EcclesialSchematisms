.PHONY: install install-ocr api api-ocr web test lint typecheck contract build check smoke docker-up docker-down

-include .env
export

install:
	uv sync
	npm --prefix apps/web ci

install-ocr:
	uv sync --extra ocr
	npm --prefix apps/web ci

api:
	uv run --exact --no-dev --package notarius-api uvicorn notarius_api.main:app --reload --host 0.0.0.0 --port 8000

api-ocr:
	uv run --exact --no-dev --extra ocr uvicorn notarius_api.main:app --reload --host 0.0.0.0 --port 8000

web:
	npm --prefix apps/web run dev

test:
	uv run --extra ocr pytest

lint:
	uv run ruff check apps/api/src libs/core/src libs/storage/src plugins/ocr/src scripts tests
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

docker-up:
	docker compose -f infra/docker/compose.yaml up --build

docker-down:
	docker compose -f infra/docker/compose.yaml down
