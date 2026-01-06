#!/bin/bash
set -euo pipefail

# Ensure required environment variables are set
: "${MINIO_SOURCE_DIR:?MINIO_SOURCE_DIR is required}"
: "${AWS_S3_ENDPOINT_URL:?AWS_S3_ENDPOINT_URL is required}"
: "${AWS_ACCESS_KEY_ID:?AWS_ACCESS_KEY_ID is required}"
: "${AWS_SECRET_ACCESS_KEY:?AWS_SECRET_ACCESS_KEY is required}"

echo "[sync-minio] Loaded environment variables."

echo "[sync-minio] Setting up MinIO client."
mc alias set local "${AWS_S3_ENDPOINT_URL}" \
  "${AWS_ACCESS_KEY_ID}" "${AWS_SECRET_ACCESS_KEY}" >/dev/null 2>&1 || true

echo "Testing connection to $AWS_S3_ENDPOINT_URL"
mc ready local
if [ $? -ne 0 ]; then
  echo "[sync-minio] Connection to MinIO failed. Exiting."
  exit 1
fi
echo "[sync-minio] Connection to MinIO successful."
echo "[sync-minio] Starting synchronization..."

SOURCE_DIR="${MINIO_SOURCE_DIR}"

# Mirror all subdirectories in SOURCE_DIR to MinIO buckets
for dir in "${SOURCE_DIR}"/*; do
  [ -d "${dir}" ] || continue

  # Bucket description: remove underscores for compatibility
  bucket="$(basename "${dir}" | tr -d '_')"

  echo "[sync-minio] Ensuring bucket '${bucket}' exists..."
  mc mb --ignore-existing "local/${bucket}"

  echo "[sync-minio] Mirroring: ${dir} → ${bucket}"
  mc mirror --overwrite --remove --watch=false "${dir}" "local/${bucket}"
done

echo "[sync-minio] Synchronization complete."
