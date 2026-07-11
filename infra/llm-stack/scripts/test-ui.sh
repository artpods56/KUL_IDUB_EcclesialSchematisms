#!/usr/bin/env bash
set -euo pipefail

curl --fail-with-body -LsS -o /dev/null -w "LiteLLM UI HTTP %{http_code}\n" http://127.0.0.1:4000/ui
