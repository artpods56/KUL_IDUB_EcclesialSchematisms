#!/usr/bin/env bash
set -euo pipefail

failures=0

check() {
  local label="$1"
  shift
  if "$@" >/tmp/llm-stack-doctor.out 2>/tmp/llm-stack-doctor.err; then
    echo "ok  ${label}"
  else
    echo "bad ${label}"
    sed -n '1,4p' /tmp/llm-stack-doctor.err >&2
    failures=$((failures + 1))
  fi
}

check "nvidia-smi sees GPU" nvidia-smi
check "docker is installed" docker --version
check "docker compose is installed" docker compose version
check "docker daemon is reachable" docker info
check "llm-stack env exists" test -f /etc/llm-stack/llm-stack.env
check "LiteLLM database URL exists" grep -q '^DATABASE_URL=' /etc/llm-stack/llm-stack.env
check "LiteLLM UI username exists" grep -q '^UI_USERNAME=' /etc/llm-stack/llm-stack.env
check "LiteLLM UI password exists" grep -q '^UI_PASSWORD=' /etc/llm-stack/llm-stack.env
check "Hugging Face cache exists" test -d /var/lib/llm-stack/huggingface
check "LiteLLM state dir exists" test -d /var/lib/llm-stack/litellm
check "Postgres state dir exists" test -d /var/lib/llm-stack/postgres
check "vLLM cache dir exists" test -d /var/lib/llm-stack/vllm
check "nginx config parses" nginx -t

if docker info >/dev/null 2>&1; then
  check "Docker NVIDIA runtime smoke test" docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi
  if docker ps --format '{{.Names}}' | grep -qx 'llm-stack-postgres'; then
    set -a
    . /etc/llm-stack/llm-stack.env
    set +a
    check "Postgres container is ready" docker exec llm-stack-postgres pg_isready -U "${LITELLM_POSTGRES_USER}" -d "${LITELLM_POSTGRES_DB}"
  fi
fi

rm -f /tmp/llm-stack-doctor.out /tmp/llm-stack-doctor.err

if [ "${failures}" -ne 0 ]; then
  echo "${failures} check(s) failed." >&2
  exit 1
fi
