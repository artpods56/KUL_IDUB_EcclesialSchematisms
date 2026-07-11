#!/usr/bin/env bash
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "Run as root." >&2
  exit 1
fi

install -m 0755 -d /opt/llm-stack
install -m 0750 -d /etc/llm-stack
install -m 0755 -d /var/lib/llm-stack
install -m 0775 -d /var/lib/llm-stack/huggingface
install -m 0775 -d /var/lib/llm-stack/litellm
install -m 0770 -d /var/lib/llm-stack/postgres
install -m 0775 -d /var/lib/llm-stack/vllm

env_file=/etc/llm-stack/llm-stack.env

if [ ! -f /etc/llm-stack/llm-stack.env ]; then
  master_key="$(openssl rand -hex 32)"
  salt_key="$(openssl rand -hex 32)"
  postgres_password="$(openssl rand -hex 32)"
  ui_password="$(openssl rand -hex 24)"
  cat >"${env_file}" <<EOF
LITELLM_MASTER_KEY=sk-${master_key}
LITELLM_SALT_KEY=sk-${salt_key}
HF_TOKEN=
VLLM_IMAGE=vllm/vllm-openai:latest
LITELLM_IMAGE=docker.litellm.ai/berriai/litellm-database:latest
POSTGRES_IMAGE=postgres:16-alpine
LITELLM_POSTGRES_USER=litellm
LITELLM_POSTGRES_PASSWORD=${postgres_password}
LITELLM_POSTGRES_DB=litellm
DATABASE_URL=postgresql://litellm:${postgres_password}@postgres:5432/litellm
UI_USERNAME=admin
UI_PASSWORD=${ui_password}
EOF
  chmod 600 "${env_file}"
fi

chmod 600 "${env_file}"

if grep -q '^LITELLM_IMAGE=ghcr.io/berriai/litellm:main-latest$' "${env_file}"; then
  sed -i 's#^LITELLM_IMAGE=.*#LITELLM_IMAGE=docker.litellm.ai/berriai/litellm-database:latest#' "${env_file}"
fi

if grep -q '^VLLM_IMAGE=vllm/vllm-openai:v0.11.0$' "${env_file}"; then
  sed -i 's#^VLLM_IMAGE=.*#VLLM_IMAGE=vllm/vllm-openai:latest#' "${env_file}"
fi

append_secret() {
  local name="$1"
  local value="$2"
  if ! grep -q "^${name}=" "${env_file}"; then
    printf '%s=%s\n' "${name}" "${value}" >>"${env_file}"
  fi
}

append_secret "POSTGRES_IMAGE" "postgres:16-alpine"
append_secret "LITELLM_POSTGRES_USER" "litellm"
append_secret "LITELLM_POSTGRES_PASSWORD" "$(openssl rand -hex 32)"
append_secret "LITELLM_POSTGRES_DB" "litellm"
append_secret "UI_USERNAME" "admin"
append_secret "UI_PASSWORD" "$(openssl rand -hex 24)"

set -a
. "${env_file}"
set +a

append_secret "DATABASE_URL" "postgresql://${LITELLM_POSTGRES_USER}:${LITELLM_POSTGRES_PASSWORD}@postgres:5432/${LITELLM_POSTGRES_DB}"
