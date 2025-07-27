# Makefile for AI_Osrodek – helper targets for containerised workflows

IMAGE_NAME ?= ai-osrodek-core
DOCKERFILE ?= src/services/core/Dockerfile
# If you keep secrets / configuration in a .env file, point ENV_FILE to it.
# The file will be passed to docker run with --env-file.
ENV_FILE ?= .env

.PHONY: docker-build docker-run-eval docker-shell clean

# Build the Docker image

docker-build:
	docker build -t $(IMAGE_NAME) -f $(DOCKERFILE) .

# Run the evaluation workflow (default command defined in the Dockerfile)
# Mount local data & configs read-only so the container can access them.

# If $(ENV_FILE) exists, pass it; otherwise docker will ignore the flag.
docker-run-eval: docker-build
	docker run  \
      --memory=16G \
	  --volume ./tmp3:/home/appuser/app/tmp \
	  --env-file $(ENV_FILE) \
	  $(IMAGE_NAME)

docker-shell: docker-build
	docker run --rm -it \
	  --memory=16G \
	  --volume ./tmp3:/home/appuser/app/tmp \
	  --env-file $(ENV_FILE) \
	  $(IMAGE_NAME) /bin/bash

# Clean up dangling images and build cache
clean:
	docker system prune -f