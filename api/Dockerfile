FROM ubuntu:22.04 AS base

# Common dependencies for both CPU and GPU builds
RUN apt-get update -y && \
    apt-get install -y curl git wget vim python3 python3-pip ffmpeg libsm6 libxext6 libgl1

WORKDIR /workspace

# Clone vLLM repository
RUN git clone https://github.com/vllm-project/vLLM.git && \
    cd vLLM && \
    git checkout main

# Base stage for ARM64 CPU build
FROM base AS arm64-cpu

RUN apt-get install -y ccache numactl gcc-12 g++-12 libtcmalloc-minimal4 libnuma-dev && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12

ENV CCACHE_DIR=/root/.cache/ccache
ENV CMAKE_CXX_COMPILER_LAUNCHER=ccache
ENV LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libtcmalloc_minimal.so.4"
ENV VLLM_TARGET_DEVICE=cpu
ENV VLLM_CPU_DISABLE_AVX512="true"

# Base stage for NVIDIA GPU build
FROM base AS nvidia-gpu

RUN apt-get install -y nvidia-cuda-toolkit

ENV VLLM_TARGET_DEVICE=cuda

# Common build stage
FROM ${TARGET_STAGE:-arm64-cpu} AS build

WORKDIR /workspace/vLLM

RUN pip install --upgrade pip && \
    pip install -r requirements/build.txt && \
    pip install -r requirements/common.txt

# Install docling_core dependency
RUN pip install docling_core

# Copy the custom script into the workspace
COPY run_docling.py /workspace/run_docling.py

# Build and install vLLM
RUN python3 setup.py bdist_wheel && \
    pip install dist/*.whl && \
    rm -rf dist

WORKDIR /workspace/

# Set entrypoint to the custom script
ENTRYPOINT ["python3", "/workspace/run_docling.py"]

