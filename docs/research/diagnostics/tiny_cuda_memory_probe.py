"""Tiny device-only CUDA startup probe for the H200 vGPU incident."""

import json
import os
import time

import torch


def stage(name: str, status: str, **details: object) -> None:
    print(
        json.dumps(
            {
                "timestamp": time.time(),
                "pid": os.getpid(),
                "stage": name,
                "status": status,
                **details,
            }
        ),
        flush=True,
    )


stage("runtime", "ok", torch=torch.__version__, cuda=torch.version.cuda)
stage("set-device", "start")
torch.cuda.set_device(0)
stage("set-device", "ok")

stage("context-kernel", "start")
value = torch.ones(1, device="cuda")
torch.cuda.synchronize()
stage("context-kernel", "ok", value=float(value.item()))

stage("seed", "start")
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.rand(1, device="cuda")
torch.cuda.synchronize()
stage("seed", "ok")

stage("empty-cache", "start")
torch.cuda.empty_cache()
stage("empty-cache", "ok")

stage("memory-stats", "start")
stats = torch.cuda.memory_stats(0)
stage("memory-stats", "ok", keys=len(stats))

stage("mem-get-info", "start")
free_memory, total_memory = torch.cuda.mem_get_info(0)
stage(
    "mem-get-info",
    "ok",
    free_memory=free_memory,
    total_memory=total_memory,
)

stage("memory-reserved", "start")
reserved = torch.cuda.memory_reserved(0)
stage("memory-reserved", "ok", reserved=reserved)
stage("probe", "complete")
