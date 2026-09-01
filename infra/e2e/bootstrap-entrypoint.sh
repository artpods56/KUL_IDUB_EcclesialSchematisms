#!/bin/sh
set -eu

: "${GRAFY_E2E_HOST_UID:?GRAFY_E2E_HOST_UID must be set}"
: "${GRAFY_E2E_HOST_GID:?GRAFY_E2E_HOST_GID must be set}"
: "${GRAFY_E2E_DOCKER_GID:?GRAFY_E2E_DOCKER_GID must be set}"

cp -a /root/.cache/uv /tmp/.uv_cache
chown -R "${GRAFY_E2E_HOST_UID}:${GRAFY_E2E_HOST_GID}" /tmp/.uv_cache

exec setpriv \
    --reuid="${GRAFY_E2E_HOST_UID}" \
    --regid="${GRAFY_E2E_HOST_GID}" \
    --groups="${GRAFY_E2E_DOCKER_GID}" \
    /app/.venv/bin/python /opt/grafy-e2e/bootstrap.py
