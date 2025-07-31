# Makefile for AI_Osrodek – helper targets for containerised workflows

# Loads the environment variables
include .env
export

CORE_IMAGE_NAME ?= ai-osrodek-core
DOCKERFILE ?= src/services/core/Dockerfile

VLLM_REPO_DIR ?= vendor/vllm
LLAMA_REPO_DIR ?= vendor/llama
VLLM_MODELS_DIR ?= tmp/models/ggufs
VLLM_SILICON_IMAGE_NAME ?= vllm-silicon:latest
LLAMA_ARM64_IMAGE_NAME ?= llama-arm64:latest
MODEL_PATH = $(VLLM_MODELS_DIR)/$(VLLM_MODEL_FILE)

VLLM_IMAGE_NAME ?= vllm/vllm-openai:latest
PORT ?= 8000
HF_CACHE ?= ~/.cache/huggingface
VLLM_CPU_KVCACHE_SPACE ?= 4  # GiB for KV cache
VLLM_CPU_OMP_THREADS_BIND ?= auto

# Defaults for scheduler config (adjust to avoid validation errors)
VLLM_MAX_BATCHED_TOKENS ?= 16384  # Increase to at least match max-model-len; e.g., 40960 for large models
VLLM_MAX_MODEL_LEN ?= 8192        # Decrease if full model len causes OOM; set to model's max if needed

ENV_FILE ?= .env

.PHONY: docker-build docker-run-eval docker-shell clone-vllm build-vllm-apple docker-vllm docker-vllm-arm64 docker-vllm-cuda clean

# Build the Docker image

docker-build:
	docker build -t $(CORE_IMAGE_NAME) -f $(DOCKERFILE) .

# Run the evaluation workflow (default command defined in the Dockerfile)
# Mount local data & configs read-only so the container can access them.
# Note: If bind mount doesn't work properly, try restarting Docker Desktop
# Note: For Colima with external drives, ensure /Volumes/T7 is mounted in colima.yaml

# If $(ENV_FILE) exists, pass it; otherwise docker will ignore the flag.
docker-run-eval: docker-build
	docker run  \
      --memory=16G \
	  --volume $(shell pwd)/tmp:/home/appuser/app/tmp:rw \
	  --env-file $(ENV_FILE) \
	  $(CORE_IMAGE_NAME)

docker-shell: docker-build
	docker run --rm -it \
	  --memory=16G \
	  --volume $(shell pwd)/tmp:/home/appuser/app/tmp:rw \
	  --env-file $(ENV_FILE) \
	  $(CORE_IMAGE_NAME) /bin/bash


# Clone vLLM repo if it doesn't exist
clone-vllm:
	@mkdir -p $(dir $(VLLM_REPO_DIR))
	@if [ ! -d $(VLLM_REPO_DIR) ]; then \
		echo "Cloning vLLM repo into $(VLLM_REPO_DIR)..."; \
		git clone https://github.com/vllm-project/vllm.git $(VLLM_REPO_DIR); \
	else \
		echo "vLLM repo already exists in $(VLLM_REPO_DIR); skipping clone."; \
		cd $(VLLM_REPO_DIR) && git pull; \
	fi

# Build the ARM64 image for Apple Silicon (depends on clone)
build-vllm-silicon: clone-vllm
	@cd $(VLLM_REPO_DIR) && \
	docker build -f docker/Dockerfile.cpu \
		--tag $(VLLM_SILICON_IMAGE_NAME) \
		--platform linux/arm64 .

hf_model_download:
	@mkdir -p $(VLLM_MODELS_DIR)
	@if [ ! -f "$(MODEL_PATH)" ]; then \
		echo "Downloading model..."; \
		wget -O "$(MODEL_PATH)" "$(VLLM_MODEL_URL)"; \
	else \
		echo "Model already exists at $(MODEL_PATH), skipping download."; \
	fi

docker-vllm-silicon: build-vllm-silicon
	docker run --rm \
		--privileged=true \
		--shm-size=4g \
		-p $(PORT):8000 \
		-v $(HF_CACHE):/root/.cache/huggingface \
		--env HUGGING_FACE_HUB_TOKEN=$(HF_TOKEN) \
		--env VLLM_CPU_KVCACHE_SPACE=$(VLLM_CPU_KVCACHE_SPACE) \
		--env VLLM_CPU_OMP_THREADS_BIND=$(VLLM_CPU_OMP_THREADS_BIND) \
		$(VLLM_SILICON_IMAGE_NAME) \
		--model $(VLLM_MODEL_NAME) \
		--dtype float16 \
		--max-num-batched-tokens $(VLLM_MAX_BATCHED_TOKENS) \
		--max-model-len $(VLLM_MAX_MODEL_LEN)

docker-vllm-gguf: hf_model_download build-vllm-silicon
	docker run --rm \
		--privileged=true \
		--shm-size=4g \
		-p $(PORT):8000 \
		-v $(HF_CACHE):/root/.cache/huggingface \
		-v ./$(MODEL_PATH):/models/$(VLLM_MODEL_FILE):ro \
		--env HUGGING_FACE_HUB_TOKEN=$(HF_TOKEN) \
		--env VLLM_CPU_KVCACHE_SPACE=$(VLLM_CPU_KVCACHE_SPACE) \
		--env VLLM_CPU_OMP_THREADS_BIND=$(VLLM_CPU_OMP_THREADS_BIND) \
		$(VLLM_SILICON_IMAGE_NAME) \
		--model /models/$(VLLM_MODEL_FILE) \
		--max-num-batched-tokens $(VLLM_MAX_BATCHED_TOKENS) \
		--max-model-len $(VLLM_MAX_MODEL_LEN) \
		--dtype bfloat16 \
		--trust-remote-code

define DOCKER_RUN
	docker run \
		--rm \
		-v $(HF_CACHE):/root/.cache/huggingface \
		--env HUGGING_FACE_HUB_TOKEN=$(HF_TOKEN) \
		-p $(PORT):8000 \
		--ipc=host \
		$(EXTRA_FLAGS) \
		$(VLLM_IMAGE_NAME) \
		--model $(VLLM_MODEL_NAME)
endef
# Clone vLLM repo if it doesn't exist
clone-llama:
	@mkdir -p $(dir $(LLAMA_REPO_DIR))
	@if [ ! -d $(LLAMA_REPO_DIR) ]; then \
		echo "Cloning llama.cpp repo into $(LLAMA_REPO_DIR)..."; \
		git clone https://github.com/ggerganov/llama.cpp $(LLAMA_REPO_DIR); \
	else \
		echo "llama.cpp repo already exists in $(LLAMA_REPO_DIR); skipping clone."; \
		cd $(LLAMA_REPO_DIR) && git pull; \
	fi

# Build the ARM64 image for Apple Silicon (depends on clone)
build-llama-arm64: clone-llama
	@cd $(LLAMA_REPO_DIR) && \
	docker build -f .devops/cpu.Dockerfile \
		--tag $(LLAMA_ARM64_IMAGE_NAME) \
		--platform linux/arm64 .  # Explicit platform for safety

# Run llama.cpp server with GGUF model (depends on download and build; removed --pull always)
docker-llama: hf_model_download build-llama-arm64
	docker run --rm \
		--platform linux/arm64 \
		-v ./$(MODEL_PATH):/models/$(VLLM_MODEL_FILE):ro \
		-p 8000:8000 \
		$(LLAMA_ARM64_IMAGE_NAME) \
		-m /models/$(VLLM_MODEL_FILE) \
		--port 8000 \
		-n 8064 \

docker-vllm:
	$(DOCKER_RUN)

docker-vllm-arm64:
	$(MAKE) EXTRA_FLAGS='--platform "linux/arm64"' docker-vllm

docker-vllm-cuda:
	$(MAKE) EXTRA_FLAGS="--runtime=nvidia --gpus=all" docker-vllm

# Clean up dangling images and build cache
clean:
	docker system prune -f