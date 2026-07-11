FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY libs ./libs
COPY apps/api ./apps/api
COPY src ./src

RUN uv sync --frozen --package notarius-api

EXPOSE 8000

CMD [".venv/bin/uvicorn", "notarius_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
