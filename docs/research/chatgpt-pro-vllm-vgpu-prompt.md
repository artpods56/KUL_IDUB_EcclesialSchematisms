# Deep Research prompt: two vLLM servers on one H200 vGPU

Copy the text below into ChatGPT Deep Research.

---

I need an evidence-based diagnosis and a practical path to run two independent vLLM servers at the same time inside one KVM guest that has a full-framebuffer NVIDIA H200-141C time-sliced vGPU. Do not give me a generic vLLM deployment guide. Investigate the exact failure described below.

Use current information as of 2026-08-28. Prefer primary sources: NVIDIA CUDA, NCCL, vGPU, and Container Toolkit documentation; the vLLM repository, documentation, pull requests, issues, and release notes; PyTorch documentation and issues; official Qwen and Gemma model repositories. You may use secondary sources only to find primary evidence. Link every material claim to the exact source, version, commit, pull request, or issue that supports it. State when a source describes a different GPU, topology, model, or software version.

## Desired production setup

The machine should run these services concurrently:

1. `Qwen/Qwen3.8-27B-FP8`, revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, for coding agents. It runs in text-only mode. A long context is useful in production, but a 4,096-token context is acceptable for proving that both servers can start.
2. `google/gemma-4-31B-it`, revision `842da3794eaa0b77d5f08bae87a17459d91ff475`, in BF16 with vision enabled. It extracts structured data from historical document scans. A 65,536-token context is desirable, but 4,096 tokens is acceptable for the dual-server startup test.
3. Existing embedding services use about 4.2 GiB of VRAM in total.

We prefer vLLM 0.28.0 for both large models. Gemma structured output works in our 0.28.0 canary. Gemma has a separate structured-output failure in 0.25.1, described below. That bug is not the main research question.

The final setup should expose the two vLLM OpenAI-compatible servers on different ports and route requests through LiteLLM. The workloads are intermittent. Qwen serves coding agents. A colleague periodically runs document-extraction batches against Gemma.

## Hardware and guest environment

- Hostname: `ai-test.ihpan.edu.pl`, SSH alias `ai-test-ihpan`.
- Guest virtualization: KVM.
- GPU: NVIDIA H200-141C, Hopper SM90.
- vGPU product: NVIDIA Virtual Compute Server, full 141 GB time-sliced profile.
- Reported framebuffer: 144,384 MiB.
- BAR1: 8,192 MiB.
- Guest kernel: `7.0.0-28-generic`.
- Guest driver: NVIDIA open kernel module `580.173.02`.
- CUDA reported by the driver: 13.0.
- `nvidia_uvm` is loaded.
- Addressing mode: HMM.
- CUDA attributes measured in the guest: `unifiedAddressing=1`, `managedMemory=0`, `concurrentManagedAccess=0`, `pageableMemoryAccess=0`, `pageableMemoryAccessUsesHostPageTables=0`, and `memoryPoolsSupported=0`.
- The host vGPU Manager version is not yet known. Compatibility between that version and guest driver `580.173.02` needs investigation.
- An earlier, separate DS4 investigation proved that this vGPU setup cannot use the expected host-memory path. Device-side reads from host pointers caused `FAULT_PDE` and `ROBUST_CHANNEL`. Copying all DS4 model data into VRAM fixed DS4. `cudaMallocAsync` is unavailable. This history may be relevant to UVM or vGPU behavior, but do not assume that it explains the vLLM failure.

NVIDIA documentation says that a VM can run multiple CUDA containers or processes inside one vGPU. The two embedding processes already running on this VM also show that the guest can host more than one CUDA process. The failure appears only when a second large vLLM EngineCore starts.

## Model memory evidence

The failure is not a normal capacity shortage in the minimal configuration.

- Qwen FP8 model weights measured by vLLM: about 27.67 GiB.
- Qwen with `--gpu-memory-utilization 0.22` reserves about 31.02 GiB.
- Gemma BF16 model weights measured by vLLM: 58.99 GiB.
- Gemma running alone with `--gpu-memory-utilization 0.47`, `max_model_len=4096`, and `max_num_seqs=1` reported 6.17 GiB of KV cache and capacity for 7,328 tokens.
- After minimal Gemma startup, total GPU use was 72,705 MiB and 68,401 MiB remained free. Qwen needed about 31 GiB.
- Qwen FP8 checkpoint files total about 30.9 GB. Google estimates Gemma 4 31B BF16 at about 69.9 GB with its stated loading allowance. Adding the embedding services gives an approximate 104.9 GB before KV-cache and runtime differences.
- In no dual-start test did vLLM or CUDA report OOM.

The production Gemma configuration is larger and currently uses about 94,390 MiB, but all dual-start experiments use deliberately minimal context and concurrency unless stated otherwise.

## Minimal dual-server configuration

The initial minimal tests used:

| Parameter | Qwen | Gemma |
|---|---:|---:|
| vLLM | 0.27.1, later 0.28.0 | 0.25.1, later 0.28.0 |
| `max_model_len` | 4,096 | 4,096 |
| `max_num_seqs` | 1 or 2 | 1 |
| `max_num_batched_tokens` | 4,096 | 4,096 |
| `gpu_memory_utilization` | 0.22 | 0.47 |
| CUDA graphs | disabled with `--enforce-eager` | disabled with `--enforce-eager` |
| tensor parallelism | 1 | 1 |

Qwen uses `--language-model-only`. Gemma keeps vision enabled. Both containers see the same vGPU through the NVIDIA Container Toolkit and use `--ipc=host`.

Gemma 0.28.0 at `gpu_memory_utilization=0.47` cannot support a 65,536-token context. vLLM reported that 64K required about 12.04 GiB of KV cache while only about 5.08 GiB was available, with an estimated maximum length near 6,032 tokens. We therefore used 4,096 tokens for the dual test. This explicit validation failure is understood and is not the unexplained hang.

## Reproduced dual-start failures

### Both containers started together

Qwen began loading its 66 checkpoint shards. Gemma reached creation of `Gemma4VisionRotaryEmbedding` and failed in `Tensor.reciprocal()` with:

```text
torch.AcceleratorError: CUDA error: an illegal instruction was encountered
```

There was no OOM.

### Gemma first, then Qwen

Gemma reached `healthy`. We then started Qwen with about 68.4 GiB of free VRAM. Qwen created its V1 EngineCore and initialized the world-size-one NCCL parallel state. Its last useful log line was:

```text
rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, PCP rank 0,
TP rank 0, EP rank N/A, EPLB rank N/A
```

Qwen never reached `Loading safetensors checkpoint shards`. Its EngineCore remained alive, consumed about 14 MiB of VRAM, and did not increase its allocation. The host process state was `R`, with no reported wait channel. The API health request returned connection reset and HTTP code `000`. Waiting longer than a normal standalone Qwen startup did not change the state.

### MPS tests

We enabled NVIDIA MPS, mounted a shared `CUDA_MPS_PIPE_DIRECTORY` into both containers, and confirmed that both processes registered as clients of one MPS server.

When Gemma started first, Gemma stayed healthy and Qwen stopped at the same post-NCCL, pre-weight-loading point. With a larger earlier configuration and the reverse start order, Qwen started and Gemma stopped before weight loading. The first large model wins. MPS changes which process starts first but does not let both start.

Research must account for NVIDIA's documented MPS limitations and NVIDIA NCCL issue #204. Do not recommend MPS as a fix without proving that the exact NCCL and vGPU combination supports it.

### Both models on vLLM 0.28.0

Gemma 0.28.0 started successfully with the minimal 4K configuration. Starting Qwen 0.28.0 beside it reproduced the same hang after world-size-one NCCL initialization. A Qwen 0.27.1 control beside Gemma behaved the same way. This result argues against a vLLM 0.28.0-only regression.

## Tests performed after upstream research

### Fixed KV-cache bytes

We started Qwen 0.28.0 beside the healthy production Gemma with:

```text
--max-model-len 4096
--max-num-seqs 2
--gpu-memory-utilization 0.22
--kv-cache-memory-bytes 2147483648
--enforce-eager
```

vLLM documents that an explicit `kv_cache_memory_bytes` value overrides automatic KV sizing from `gpu_memory_utilization`. Qwen still stopped at the same world-size-one NCCL point before loading weights. It used about 14 MiB. No OOM occurred.

This test makes the known V1 cross-process KV-memory accounting bug less likely as the immediate cause of this exact hang. It does not prove that vLLM has no other cross-process memory-accounting bug.

### `NCCL_P2P_DISABLE=1`

We repeated the same fixed-KV Qwen canary with `NCCL_P2P_DISABLE=1`. It stopped at the same line, used about 14 MiB, and never loaded weights. This makes NCCL peer-to-peer transport selection less likely. It does not rule out another NCCL initialization path or a CUDA call immediately after vLLM logs the rank assignment.

We also set `NCCL_DEBUG=INFO` and `NCCL_DEBUG_SUBSYS=INIT,ENV`. In one run, `NCCL_DEBUG_FILE` did not produce a file even though vLLM logged `backend=nccl` and the rank assignment. Determine whether this absence is meaningful for NCCL 2.x with `world_size=1`, or whether vLLM completes a trivial NCCL setup without emitting those logs.

Every canary container was removed after its observation window. The healthy production Gemma container remained running.

## Standalone behavior and nearby findings

- Qwen FP8 on vLLM 0.27.1 runs by itself. Its former production configuration used a 262,144-token context, `gpu_memory_utilization=0.35`, `max_num_seqs=4`, and reached healthy in about two minutes.
- Gemma BF16 on vLLM 0.25.1 runs by itself. Its production configuration uses a 65,536-token context, `gpu_memory_utilization=0.65`, `max_num_seqs=1`, `--enforce-eager`, and disabled asynchronous scheduling. It reaches healthy in about three minutes.
- Gemma 0.28.0 also starts by itself in the tested minimal and structured-output canaries.
- vLLM 0.27.1 failed to start Gemma correctly because Gemma 4 has heterogeneous attention head dimensions of 256 and 512. vLLM 0.28.0 starts this model.
- Existing embedding servers use the same vGPU concurrently without this failure.

## Separate Gemma findings that should not be conflated with the dual-start hang

Gemma previously produced `CUBLAS_STATUS_EXECUTION_FAILED` and `cudaErrorIllegalAddress` with at least two concurrent sequences. Later canaries could not reproduce that failure. Both forced `TRITON_ATTN` and the default FlashAttention 4 completed large concurrent text series with HTTP 200. A tiny two-request image canary also completed. Do not assume that FA4 is the current dual-start cause.

Gemma structured output on vLLM 0.25.1 failed 10 out of 10 times with:

```text
grammar rejected tokens [1]
```

The production workload had also seen `grammar rejected tokens [236777]`, where token 236777 decodes to `I`, across guidance, xgrammar, and outlines. The EOS token 1 failure matches the known Gemma stop-token bug fixed by vLLM pull request #49227.

A Gemma vLLM 0.28.0 canary with XGrammar passed the same small JSON Schema test 10 out of 10 times. It also passed 40 out of 40 parallel text requests. This confirms the v0.28.0 fix for our EOS reproducer, but it does not prove that token 236777 or the outlines backend is fixed. This structured-output result is why vLLM 0.28.0 is desirable for Gemma. It is probably separate from the second-EngineCore startup hang.

## Relevant upstream material already found

Start with these sources and verify their current status, linked fixes, affected releases, and applicability:

- vLLM 0.28.0 `serve` reference. It explicitly describes two vLLM instances on one GPU using per-instance `--gpu-memory-utilization` values and documents `--kv-cache-memory-bytes`: https://docs.vllm.ai/en/v0.28.0/cli/serve/
- vLLM issue #17979, V1 available-VRAM calculation includes memory from another vLLM instance: https://github.com/vllm-project/vllm/issues/17979
- vLLM issue #10643, second Docker instance receives an incorrect negative KV-cache budget: https://github.com/vllm-project/vllm/issues/10643
- vLLM issue #10451, changed multiple-instance memory accounting after vLLM 0.6.3: https://github.com/vllm-project/vllm/issues/10451
- vLLM issue #26619, `gpu-memory-utilization` not honored as expected for multiple instances: https://github.com/vllm-project/vllm/issues/26619
- vLLM issue #12401, two asynchronous engines on one A100 fail: https://github.com/vllm-project/vllm/issues/12401
- vLLM troubleshooting documentation: https://docs.vllm.ai/en/latest/usage/troubleshooting/
- vLLM issue #25127, an H100 EngineCore stuck in running state with UVM symptoms. This is not a dual-instance reproducer, but the process-state symptom may be relevant: https://github.com/vllm-project/vllm/issues/25127
- vLLM issue #33041, NCCL initialization succeeds and EngineCore never reaches model loading on a different Blackwell multi-GPU topology. Treat it only as a symptom match: https://github.com/vllm-project/vllm/issues/33041
- NVIDIA vGPU introduction, including multiple containers inside one vGPU: https://docs.nvidia.com/vgpu/latest/grid-vgpu-user-guide/grid-vgpu-introduction.html
- NVIDIA vGPU time-sliced architecture: https://docs.nvidia.com/vgpu/latest/grid-vgpu-user-guide/installing-configuring-grid-vgpu.html#time-sliced-nvidia-vgpu-internal-architecture-for-vgpus-on-a-single-instance-gpu
- NVIDIA MPS documentation: https://docs.nvidia.com/deploy/mps/latest/index.html
- NVIDIA MPS usage guidance: https://docs.nvidia.com/deploy/mps/when-to-use-mps.html
- NCCL environment variable reference: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
- NCCL issue #204 about MPS support and hangs: https://github.com/NVIDIA/nccl/issues/204
- vLLM pull request #49227 for Gemma stop-token handling in structured output: https://github.com/vllm-project/vllm/pull/49227

Do not treat an issue as resolved merely because GitHub marks it closed. Find the closing commit, pull request, maintainer conclusion, or stale bot action. Map each fix to the first vLLM release that contains it.

## Questions the research must answer

1. Does NVIDIA officially support two independent, memory-heavy CUDA or vLLM containers inside one H200-141C time-sliced vGPU? Separate support for generic CUDA containers, MPS clients, NCCL, and this exact vGPU profile.
2. Is there a known vLLM, PyTorch, CUDA, NCCL, or NVIDIA vGPU issue that matches this signature: the second world-size-one V1 EngineCore remains in process state `R`, uses about 14 MiB, and stops after rank assignment before loading weights, while the first large process remains healthy and tens of GiB remain free?
3. What exact operation happens immediately after vLLM logs the rank assignment in vLLM 0.25.1, 0.27.1, and 0.28.0? Trace the source code. Identify the next CUDA, NCCL, PyTorch, allocator, model-registry, or device-capability calls that could spin or block before the weight-loader log appears.
4. Does vLLM need to initialize NCCL for `tensor_parallel_size=1`, `pipeline_parallel_size=1`, and `data_parallel_size=1`? Is there a supported flag or small patch that bypasses NCCL for world size one? Find tests, code paths, and maintainer statements. Do not propose an unsupported patch without marking it as diagnostic only.
5. Does vLLM 0.28.0 still contain the cross-process V1 memory-accounting bugs described in #17979, #10643, and #10451? Explain why explicit `--kv-cache-memory-bytes` did not change our stop point.
6. Could the hang occur in CUDA context creation, CUDA primary-context retention, UVM initialization, peer-access probing, cuMem APIs, pinned-memory setup, FlashInfer warmup, CUTLASS or cuDNN probing, or another step after parallel-state initialization? Rank these possibilities and tie every prediction to a specific next test.
7. Which guest driver and vGPU Manager combinations officially support H200-141C, CUDA 13.0, KVM, the NVIDIA open kernel module, and the tested vLLM container CUDA runtime? Identify the exact compatibility matrix and what host information the administrator must collect.
8. Is `NCCL_SHM_DISABLE=1` a meaningful next A/B test for world size one? Assess `NCCL_CUMEM_ENABLE`, `NCCL_LAUNCH_MODE`, `CUDA_MODULE_LOADING`, `CUDA_DEVICE_MAX_CONNECTIONS`, `--disable-custom-all-reduce`, and vLLM distributed-executor options the same way. Reject flags that cannot affect this topology.
9. Would separate Docker IPC namespaces be safer than `--ipc=host` for two TP1 servers, or does vLLM require host IPC only for performance? Find evidence before recommending a change.
10. Can one process or one vLLM server host both models without two EngineCore processes? Check current first-class multi-model support, dynamic load and unload support, Sleep Mode, RLHF weight-reload features, and practical model-switching alternatives. Distinguish concurrent serving from sequential swapping.
11. If the root cause is below vLLM, what is the smallest standalone reproducer? Design a two-process PyTorch or CUDA test in which the first process allocates about 60 GiB and runs a simple kernel, then the second process allocates about 31 GiB and runs a BF16 GEMM or another operation that matches the first failing vLLM call. The test must distinguish allocation failure, context-init hang, NCCL-init hang, and kernel failure.
12. What should we include in a strong vLLM issue and an NVIDIA administrator escalation? List exact commands and artifacts. Cover process stacks, `strace`, `gdb` or `py-spy` where applicable, `/proc/<pid>/stack`, NCCL logs, CUDA samples, `nvidia-smi -q`, guest and host `nvidia-bug-report.sh`, Xid and MMU logs, vGPU Manager version, container image digests, and a minimal reproducer. Mark commands that may disrupt the healthy GPU context.

## Required research output

Produce these sections:

1. A short conclusion with the most likely cause, confidence level, and the safest current operating mode.
2. An evidence table with columns for finding, primary source, affected version, match quality, and implication for this machine.
3. Four or five ranked, falsifiable root-cause hypotheses. For each one, state the exact observation that would support or reject it.
4. A one-variable-at-a-time experiment plan. Give exact commands or pseudocode, expected logs, pass and fail criteria, runtime risk, and cleanup. Put non-disruptive tests first. Do not repeat the fixed-KV or `NCCL_P2P_DISABLE=1` tests unless a materially different configuration changes their diagnostic value.
5. A minimal two-process CUDA or PyTorch reproducer suitable for the VM and for an NVIDIA support case. Avoid host-memory and managed-memory APIs because this vGPU has already shown that those paths are unavailable. Use device allocations only.
6. A recommended production workaround if simultaneous serving cannot be made reliable. Compare sequential model switching, separate VMs or physical GPUs, passthrough, MIG-backed vGPU if supported, and alternate inference servers only when primary evidence supports them.
7. A draft vLLM GitHub issue and a separate administrator or NVIDIA escalation note. Keep facts, inferences, and open questions clearly separated.
8. A source list with direct links. Prefer exact versioned documentation and source lines over search-result pages.

Do not conclude that the models do not fit in VRAM. The measured minimal budgets leave about 37 GiB after both models and embedding services. Do not prescribe smaller context or lower concurrency as the main fix because 4,096 tokens and one sequence already reproduce the failure. Do not claim that MPS solves it. Do not conflate Gemma structured-output bugs or the earlier batch-two CUDA error with the second-EngineCore startup hang.

If no public issue exactly matches, say so. Then identify the nearest matches, explain each mismatch, and build the next diagnostic steps from the vLLM source path and NVIDIA support matrix rather than from guesswork.
