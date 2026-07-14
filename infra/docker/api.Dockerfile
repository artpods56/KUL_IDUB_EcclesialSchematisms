FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS source

WORKDIR /app

COPY pyproject.toml uv.lock alembic.ini ./
COPY libs/core ./libs/core
COPY libs/persistence ./libs/persistence
COPY libs/storage ./libs/storage
COPY plugins/ocr ./plugins/ocr
COPY apps/api ./apps/api
COPY infra/db ./infra/db

EXPOSE 8000

CMD [".venv/bin/uvicorn", "notarius_api.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM source AS api-ocr

RUN uv sync --locked --no-dev --extra ocr

FROM source AS api

RUN uv sync --locked --no-dev --package notarius-api
