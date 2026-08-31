# Two vLLM containers on one vGPU

Research date: 2026-08-28

## Short answer

NVIDIA supports running multiple CUDA containers inside one vGPU, and vLLM documents running two instances on one GPU by assigning each instance a separate memory fraction. The main practical risk is vLLM memory profiling: upstream reports show that the second V1 instance can count memory owned by the first instance and fail during KV-cache sizing.

## Primary sources

- [vLLM 0.28.0 `serve` reference](https://docs.vllm.ai/en/v0.28.0/cli/serve/): `--gpu-memory-utilization` is documented as a per-instance limit. The reference explicitly gives two vLLM instances at `0.5` each as an example. It also documents `--kv-cache-memory-bytes`, which gives a fixed KV-cache budget and overrides the automatic calculation based on the utilization fraction.
- [vLLM cache configuration](https://docs.vllm.ai/en/latest/api/vllm/config/cache/): the API reference repeats the per-instance behavior and describes the current KV-cache configuration fields.
- [NVIDIA vGPU introduction](https://docs.nvidia.com/vgpu/latest/grid-vgpu-user-guide/grid-vgpu-introduction.html): NVIDIA states that a VM can run multiple containers or CUDA processes in parallel inside one vGPU.
- [NVIDIA Container Toolkit user guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.12.0/user-guide.html): explains how `--gpus` and `NVIDIA_VISIBLE_DEVICES` expose a GPU to a container. These options select devices; they do not partition framebuffer memory or compute capacity.
- [NVIDIA vGPU time-sliced architecture](https://docs.nvidia.com/vgpu/latest/grid-vgpu-user-guide/installing-configuring-grid-vgpu.html#time-sliced-nvidia-vgpu-internal-architecture-for-vgpus-on-a-single-instance-gpu): explains scheduling for time-sliced vGPUs. This is useful context for the H200-141C setup, but it does not describe vLLM-specific isolation.

## Directly relevant vLLM issues

- [#17979: available VRAM calculation bug in V1](https://github.com/vllm-project/vllm/issues/17979): the closest upstream report. The first vLLM starts, while the second counts memory used by the first as non-Torch allocation and fails KV-cache sizing. The reporter's V0 workaround applies to an older vLLM release and should not be assumed to work in 0.28.0.
- [#10643: GPU memory accounting with multiple instances](https://github.com/vllm-project/vllm/issues/10643): includes two Docker commands and logs showing the second server receiving a negative KV-cache budget because the first server's allocation enters its memory profile.
- [#10451: changed `gpu_memory_utilization` behavior](https://github.com/vllm-project/vllm/issues/10451): documents the history of global versus per-process memory accounting and why multiple instances became unreliable across versions.
- [#26619: utilization not honored with multiple instances](https://github.com/vllm-project/vllm/issues/26619): a newer report where two Docker services exceed the expected fractions. It is useful evidence that the utilization value is not a strict process-level VRAM quota.
- [#12401: two engines on one GPU](https://github.com/vllm-project/vllm/issues/12401): an older failure involving two asynchronous engines on one A100. The topology differs, but it confirms that same-GPU multi-engine use has been fragile.

## Diagnostics and MPS

- [vLLM troubleshooting](https://docs.vllm.ai/en/latest/usage/troubleshooting/): official checks for CUDA and NCCL failures. It treats environment switches such as `NCCL_P2P_DISABLE=1` as diagnostic experiments, not general fixes.
- [NCCL environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html): reference for `NCCL_DEBUG` and subsystem logging.
- [NVIDIA MPS overview](https://docs.nvidia.com/deploy/mps/latest/index.html) and [when to use MPS](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html): MPS can improve CUDA scheduling between processes and provide compute partitioning. It does not divide model memory automatically.
- [NCCL issue #204](https://github.com/NVIDIA/nccl/issues/204): NVIDIA states that NCCL with MPS is unsupported in the reported setup and may hang. MPS should therefore be a separate canary experiment, not the default remedy for vLLM.

## Practical reading of the sources

For vLLM 0.28.0, the cleanest documented setup is two independent `vllm serve` containers with the same vGPU visible, different host ports, conservative `--gpu-memory-utilization` values, small `--max-model-len` and `--max-num-seqs`, and preferably explicit `--kv-cache-memory-bytes` values. Start them sequentially and leave weight, CUDA-graph, multimodal-cache, and runtime headroom outside the two KV-cache budgets.

The sources support the architecture, but they do not prove that every V1, NCCL, driver, and time-sliced-vGPU combination works. A second engine hanging before weight loading despite ample free framebuffer is more consistent with an initialization or driver interaction than a normal out-of-memory condition.

## Local canary results

- 2026-08-28: Qwen 0.28.0 beside the healthy production Gemma, with `--kv-cache-memory-bytes 2147483648`, `max_model_len=4096`, and `max_num_seqs=2`, still stopped after `world_size=1 ... backend=nccl`, before weight loading. The Qwen process used about 14 MiB; no OOM was reported.
- 2026-08-28: repeating the same canary with `NCCL_P2P_DISABLE=1` produced the same result. The temporary containers were removed; production Gemma stayed healthy.

These tests make a KV-cache sizing error and NCCL peer-to-peer selection less likely as the immediate cause. They do not yet distinguish a vGPU driver/runtime issue from another V1 initialization path.

## Follow-up after Deep Research

The 2026-08-28 follow-up isolated the failure below vLLM. Direct markers showed
that the second EngineCore hangs in `torch.accelerator.get_memory_info`, after
`memory_stats` returns. A separate PyTorch-only process then hung in
`torch.cuda.set_device(0)` beside the healthy Gemma process. Both processes
used one hot CPU thread, issued the same periodic NVIDIA RM-control ioctl, and
spent their sampled CPU time inside `libcuda.so.580.173.02`.

The full evidence and remaining control test are in
[the second-context follow-up](h200-vgpu-second-context-follow-up.md).
