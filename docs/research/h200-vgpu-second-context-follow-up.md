# A new CUDA context hangs beside Gemma on the H200 vGPU

Test date: 2026-08-28

## Result

The failure reproduces without vLLM startup logic, NCCL, model loading, KV
cache, FP8 kernels, or attention code. A tiny PyTorch process hangs in
`torch.cuda.set_device(0)` when it starts beside the healthy Gemma process.

The maintenance-window A/B test is complete. With Gemma stopped, all three
probe runs completed. With the same Gemma container running and healthy, all
three probe runs hung after printing `set-device: start`. The Gemma-resident
state is therefore the condition that changed the result in this test.

This A/B result does not distinguish Gemma's CUDA context from its memory
reservation or another property of the resident workload. It does show that
the failure is below vLLM and occurs before the probe allocates its one-element
CUDA tensor.

## Guest inventory

The read-only inventory returned:

| Component | Measured value |
|---|---|
| Guest OS | Ubuntu 24.04.4 LTS |
| Guest kernel | `7.0.0-28-generic` |
| Guest driver | NVIDIA open kernel module `580.173.02` |
| GPU | `NVIDIA H200-141C` |
| Reported framebuffer | 144,384 MiB |
| Addressing mode | HMM |
| Container Toolkit | `1.19.1` |
| Docker | `29.7.2` client and server |
| vLLM image | `vllm/vllm-openai:v0.28.0` |
| Image digest | `sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14` |
| vLLM build commit | `2cf0a6915ce544dc493a0990f2ea38d81601128a` |
| PyTorch | `2.13.0+cu130` |
| Container CUDA runtime | `13.0` |
| Container NCCL | `2.29.7` |

The guest driver and Container Toolkit match the guest-side versions in the
AI Enterprise 7.7 matrix cited by the Deep Research report. The host vGPU
Manager version remains unknown. The report identifies `580.159.01` as the
tested host Manager paired with guest driver `580.173.02`.

`nvidia-smi vgpu -q` returned `No supported devices in vGPU mode` in the
guest. Do not use that command as proof that the guest lacks a vGPU. The
regular `nvidia-smi` output identifies the device as `H200-141C`.

## vLLM call boundary

The canary used Qwen 0.28.0 with the already tested minimal settings:

```text
max_model_len=4096
max_num_seqs=2
gpu_memory_utilization=0.22
kv_cache_memory_bytes=2147483648
tensor_parallel_size=1
enforce_eager=true
CUDA_MODULE_LOADING=LAZY
Docker IPC mode=host
```

Direct markers in `GPUWorker.init_device` produced this final sequence:

```text
[DEBUG-vgpu-stage] after-distributed-environment
[DEBUG-vgpu-stage] before-set-random-seed
[DEBUG-vgpu-stage] after-set-random-seed
[DEBUG-vgpu-stage] before-gc
[DEBUG-vgpu-stage] after-gc
[DEBUG-vgpu-stage] before-empty-cache
[DEBUG-vgpu-stage] after-empty-cache
[DEBUG-vgpu-stage] before-memory-snapshot
[DEBUG-vgpu-stage] before-memory-stats
[DEBUG-vgpu-stage] after-memory-stats
[DEBUG-vgpu-stage] before-get-memory-info
```

No marker followed `before-get-memory-info`. The non-returning call is
`torch.accelerator.get_memory_info(cuda:0)`, which reaches `cudaMemGetInfo`.

This result rules out these earlier candidates at the observed boundary:

- `torch.cuda.manual_seed_all`;
- Python garbage collection;
- `torch.accelerator.empty_cache`;
- PyTorch allocator `memory_stats`;
- KV-cache budget calculation;
- workspace-manager construction;
- model-runner construction and weight loading.

## Process and driver evidence

The EngineCore main thread stayed in state `R` and used about one CPU core.
The other threads slept in futex, epoll, or watchdog waits. The main thread had
no kernel wait channel and was not blocked in a syscall.

A short `strace` observed the same successful ioctl once per second:

```text
ioctl(/dev/nvidiactl, _IOC(_IOC_READ|_IOC_WRITE, 0x46, 0x2a, 0x20), ...)
    = 0 in about 0.7 ms
```

Between calls, the process used the CPU. A five-second `perf` sample captured
628 samples with none lost. The leading frames and call chain were inside
`libcuda.so.580.173.02`; the library has no public symbols for those offsets.

This is a user-space driver polling loop around NVIDIA RM-control calls. It is
not a socket wait, a shared-memory futex, or an OOM path.

## Minimal PyTorch reproducer

The test uses the exact vLLM 0.28.0 image but runs only
[tiny_cuda_memory_probe.py](diagnostics/tiny_cuda_memory_probe.py). The script
uses one device tensor and no NCCL, managed memory, pinned memory, model code,
or vLLM imports.

Beside Gemma, the only output was:

```json
{"stage":"runtime","status":"ok","torch":"2.13.0+cu130","cuda":"13.0"}
{"stage":"set-device","status":"start"}
```

The process never printed `set-device: ok`. Its main thread stayed in `R` at
about 100% CPU. It issued the same once-per-second ioctl to `/dev/nvidiactl`,
and `perf` again placed the hot frames in `libcuda.so.580.173.02`.

The minimal result changes the leading diagnosis. The trigger does not require
a second memory-heavy model because the probe hangs before allocating its
one-element tensor. Existing embedding contexts were running before Gemma
started, so their continued operation does not test creation of a new context
after Gemma is resident.

## Maintenance-window A/B result

The test kept both embedding servers running and changed only the state of the
existing `llm-stack-vllm-gemma` container.

| Gemma state | Probe runs | Result |
|---|---:|---|
| Stopped | 3 | 3 completed through `probe: complete` |
| Running and `healthy` | 3 | 3 hung after printing `set-device: start` |

In the three completed runs, `torch.cuda.set_device(0)` returned in 0.33 to 0.37
seconds. Each run then allocated a CUDA tensor, synchronized the device,
seeded CUDA, cleared the allocator cache, read allocator statistics, called
`cudaMemGetInfo`, and printed `probe: complete`.

The three hanging runs used the same script and the same
`vllm/vllm-openai:v0.28.0` image. Each process loaded PyTorch and printed
`set-device: start`, but none printed `set-device: ok`. Each probe container
was removed after observation. Gemma stayed `healthy` during all three runs.

After the test, `nvidia-smi` reported 42,485 MiB free. Gemma used 94,390 MiB,
and the two embedding processes used 1,805 MiB and 2,422 MiB. No probe emitted
an OOM error.

## Updated confidence ledger

### Confirmed

- The Qwen EngineCore hang occurs inside `cudaMemGetInfo` in the second
  process.
- A separate PyTorch-only process can hang earlier in
  `torch.cuda.set_device(0)` under the same resident Gemma workload.
- The PyTorch-only probe completes three times when Gemma is stopped and the
  two embedding servers remain running.
- The same probe hangs three times at `torch.cuda.set_device(0)` after the same
  Gemma container returns to `healthy`.
- Both hangs are active user-space loops inside guest `libcuda.so.580.173.02`.
- Both loops call the same NVIDIA RM-control ioctl about once per second, and
  the ioctl returns success.
- Production Gemma remained healthy throughout the disposable tests.

### Strong inference

The leading cause is the NVIDIA guest driver, vGPU Manager, or their context
and virtual-memory integration under the resident Gemma workload. The
repeatable A/B result makes vLLM-specific startup code an unlikely cause.

### Still unknown

- Whether Gemma's memory size, its CUDA context, or another property is the
  necessary trigger.
- The host vGPU Manager version and whether it matches the supported matrix.
- Whether an aligned host stack or direct passthrough removes the failure.

## Next action

Open a case with the platform administrator or NVIDIA. Attach this report,
`tiny_cuda_memory_probe.py`, the driver traces, the exact image digest, and
synchronized host logs. Ask the host administrator for the vGPU Manager
version and confirmation that the host and guest versions match the supported
matrix.

If another maintenance window is available, vary the resident GPU allocation
without changing the probe. That test can locate a memory threshold and show
whether the trigger follows the size of the resident workload. A small
persistent CUDA process can also test whether the trigger follows the number
of contexts instead of their memory use.

## Cleanup state

All Qwen and PyTorch canary containers were removed. The remote copy of the
probe and the temporary profiler files were deleted. The production
`llm-stack-vllm-gemma` container finished the test as `running/healthy` with
zero restarts. A final chat-completions request returned `OK` with finish reason
`stop`. Only Gemma and the two established embedding processes used the GPU
after cleanup.
