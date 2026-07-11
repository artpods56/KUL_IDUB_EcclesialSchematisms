#!/usr/bin/env bash
set -euo pipefail

if [ ! -f /etc/llm-stack/llm-stack.env ]; then
  echo "/etc/llm-stack/llm-stack.env is missing." >&2
  exit 1
fi

set -a
. /etc/llm-stack/llm-stack.env
set +a

if [ -z "${LITELLM_MASTER_KEY:-}" ]; then
  echo "LITELLM_MASTER_KEY is missing." >&2
  exit 1
fi

curl --fail-with-body -sS http://127.0.0.1:4000/v1/chat/completions \
  -H "Authorization: Bearer ${LITELLM_MASTER_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-4-31b-it","messages":[{"role":"user","content":"Say OK"}],"max_tokens":16}'
