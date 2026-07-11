# LLM Stack Phase 1

This directory contains the first deployable slice for `ai-test.ihpan.edu.pl`:

- vLLM serves `google/gemma-4-31B-it` on the private Compose network in text-only eager mode.
- LiteLLM exposes the OpenAI-compatible proxy on `127.0.0.1:4000`.
- Postgres stores LiteLLM Admin UI state, virtual keys, and usage data.
- nginx remains the only public listener and proxies HTTPS traffic to LiteLLM.

llama-swap and declarative multi-model generation are intentionally left for a later phase.

## Host Layout

```text
/opt/llm-stack
  compose.yaml
  justfile
  litellm.yaml
  nginx/
  scripts/

/etc/llm-stack
  llm-stack.env

/var/lib/llm-stack
  huggingface/
  litellm/
  postgres/
  vllm/
```

`/opt/llm-stack` contains deployment files, `/etc/llm-stack` contains secrets and environment, and `/var/lib/llm-stack` contains persistent service-owned state.

## First Install

Run these commands on the VM as `root`:

```bash
mkdir -p /opt/llm-stack
rsync -a infra/llm-stack/ root@10.89.99.10:/opt/llm-stack/
cd /opt/llm-stack
./scripts/bootstrap-host.sh
./scripts/init-host.sh
just doctor
just up
just logs
```

If the model is gated on Hugging Face, add `HF_TOKEN` to `/etc/llm-stack/llm-stack.env` before `just up`.

`./scripts/init-host.sh` creates the LiteLLM master key, salt key, Postgres password, and Admin UI password in `/etc/llm-stack/llm-stack.env`. Treat that file as secret material.

## Operations

```bash
just doctor
just up
just down
just restart
just logs
just logs-service vllm
just logs-service litellm
just test
just test-ui
```

nginx is not reloaded by `just up`. Apply the proxy config explicitly after reviewing it:

```bash
just install-nginx
just nginx-test
just nginx-reload
```

## Local API Test

From the VM:

```bash
source /etc/llm-stack/llm-stack.env
curl --fail-with-body -sS http://127.0.0.1:4000/v1/chat/completions \
  -H "Authorization: Bearer ${LITELLM_MASTER_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-4-31b-it","messages":[{"role":"user","content":"Say OK"}],"max_tokens":16}'
```

## LiteLLM Admin UI

Access the dashboard through an SSH tunnel instead of exposing it publicly:

```bash
ssh -i ~/.ssh/id_ed25519_uni_vm -L 4000:127.0.0.1:4000 root@10.89.99.10
```

Then open:

```text
http://localhost:4000/ui
```

Read `UI_USERNAME` and `UI_PASSWORD` from `/etc/llm-stack/llm-stack.env` on the VM.
