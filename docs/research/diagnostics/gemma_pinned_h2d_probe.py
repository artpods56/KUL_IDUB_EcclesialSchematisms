"""Compare pageable and pinned CPU-to-GPU bitmask transfers."""

import importlib.metadata
import json

import torch
from vllm.utils.torch_utils import PIN_MEMORY


TOKEN_ID = 2717
WORD_INDEX = TOKEN_ID // 32
BIT_INDEX = TOKEN_ID % 32


def read_bit(value: int) -> int:
    return (value >> BIT_INDEX) & 1


results = []
for pinned in (False, True):
    for non_blocking in (False, True):
        cpu_mask = torch.full(
            (1, 8192),
            -1,
            dtype=torch.int32,
            pin_memory=pinned,
        )
        numpy_mask = cpu_mask.numpy()
        numpy_mask[0, WORD_INDEX] = int(numpy_mask[0, WORD_INDEX]) & ~(
            1 << BIT_INDEX
        )
        cpu_bit = read_bit(int(cpu_mask[0, WORD_INDEX]))
        pointers_match = (
            cpu_mask.data_ptr() == numpy_mask.__array_interface__["data"][0]
        )

        gpu_mask = cpu_mask.to("cuda", non_blocking=non_blocking)
        torch.cuda.synchronize()
        gpu_bit = read_bit(int(gpu_mask[0, WORD_INDEX].item()))
        results.append(
            {
                "pinned": pinned,
                "non_blocking": non_blocking,
                "numpy_and_tensor_share_pointer": pointers_match,
                "cpu_bit": cpu_bit,
                "gpu_bit": gpu_bit,
                "passed": cpu_bit == 0 and gpu_bit == 0,
            }
        )

print(
    json.dumps(
        {
            "versions": {
                "vllm": importlib.metadata.version("vllm"),
                "torch": importlib.metadata.version("torch"),
            },
            "vllm_pin_memory_default": PIN_MEMORY,
            "results": results,
        },
        sort_keys=True,
    )
)

if not all(result["passed"] for result in results):
    raise SystemExit("at least one CPU-to-GPU transfer variant failed")
