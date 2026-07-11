FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY libs ./libs
COPY apps/worker ./apps/worker

RUN uv sync --frozen --package notarius-worker

CMD [".venv/bin/python", "-m", "notarius_worker.main"]
