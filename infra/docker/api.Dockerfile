FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY libs/core ./libs/core
COPY libs/storage ./libs/storage
COPY apps/api ./apps/api

RUN uv sync --frozen --no-dev --package notarius-api

EXPOSE 8000

CMD [".venv/bin/uvicorn", "notarius_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
