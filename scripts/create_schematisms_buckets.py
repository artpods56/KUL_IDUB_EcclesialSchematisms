import minio

import os
from minio import Minio
from dotenv import load_dotenv
from pathlib import Path

ROOT_DIR = Path(__file__).resolve(strict=True).parent.parent

env_path = os.path.join(ROOT_DIR,"label-studio/.env.dev")
print(env_path)

env_loaded = load_dotenv(env_path)
if env_loaded:
    print("Environment variables loaded successfully.")
else:
    print("Failed to load environment variables.")



MINIO_ROOT_USER = os.getenv("MINIO_ROOT_USER")
MINIO_ROOT_PASSWORD = os.getenv("MINIO_ROOT_PASSWORD")
MINIO_URL = os.getenv("AWS_S3_ENDPOINT_URL")
# client  = Minio(
#     "localhost:9000",
#     access_key="minioadmin",