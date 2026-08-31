"""Verify sparse XGrammar CUDA masking used by vLLM's indexed path."""

import importlib.metadata
import json

import torch
import xgrammar
from transformers import AutoTokenizer
from vllm.utils.torch_utils import PIN_MEMORY, async_tensor_h2d


MODEL = "google/gemma-4-31B-it"
REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
MARKDOWN_FENCE_TOKEN = 2717
OPENING_BRACE_TOKEN = 236782
SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "integer"}},
    "required": ["value"],
    "additionalProperties": False,
}


def run_variant(
    mask_cpu: torch.Tensor,
    vocab_size: int,
    dtype: torch.dtype,
    indices_kind: str,
) -> dict[str, object]:
    sparse_mask_cpu = torch.full(
        (3, mask_cpu.shape[1]),
        -1,
        dtype=mask_cpu.dtype,
        pin_memory=PIN_MEMORY,
    )
    sparse_mask_cpu[1].copy_(mask_cpu[0])
    sparse_mask = sparse_mask_cpu.to("cuda", non_blocking=True)

    logits = torch.zeros((3, vocab_size), dtype=dtype, device="cuda")
    logits[:, MARKDOWN_FENCE_TOKEN] = 100.0
    logits[:, OPENING_BRACE_TOKEN] = 1.0

    if indices_kind == "python_list":
        indices: list[int] | torch.Tensor = [1]
    elif indices_kind == "cuda_tensor":
        indices = torch.tensor([1], dtype=torch.int32, device="cuda")
    elif indices_kind == "async_h2d":
        indices = async_tensor_h2d([1], dtype=torch.int32, device="cuda")
    else:
        raise ValueError(f"unknown indices kind: {indices_kind}")

    xgrammar.apply_token_bitmask_inplace(logits, sparse_mask, indices=indices)
    torch.cuda.synchronize()

    finite_markdown = torch.isfinite(
        logits[:, MARKDOWN_FENCE_TOKEN]
    ).tolist()
    finite_opening_brace = torch.isfinite(
        logits[:, OPENING_BRACE_TOKEN]
    ).tolist()
    argmax = logits.argmax(dim=-1).tolist()
    passed = (
        finite_markdown == [True, False, True]
        and finite_opening_brace == [True, True, True]
        and argmax
        == [MARKDOWN_FENCE_TOKEN, OPENING_BRACE_TOKEN, MARKDOWN_FENCE_TOKEN]
    )
    return {
        "dtype": str(dtype),
        "indices": indices_kind,
        "finite_markdown": finite_markdown,
        "finite_opening_brace": finite_opening_brace,
        "argmax": argmax,
        "passed": passed,
    }


tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    revision=REVISION,
    local_files_only=True,
)
tokenizer_info = xgrammar.TokenizerInfo.from_huggingface(
    tokenizer,
    vocab_size=tokenizer.vocab_size,
)
compiled = xgrammar.GrammarCompiler(tokenizer_info).compile_json_schema(
    json.dumps(SCHEMA)
)
matcher = xgrammar.GrammarMatcher(
    compiled,
    override_stop_tokens=[1, 106, 50],
)
mask_cpu = xgrammar.allocate_token_bitmask(1, tokenizer.vocab_size)
matcher.fill_next_token_bitmask(mask_cpu, 0)

results = [
    run_variant(mask_cpu, tokenizer.vocab_size, dtype, indices_kind)
    for dtype in (torch.float32, torch.bfloat16)
    for indices_kind in ("python_list", "cuda_tensor", "async_h2d")
]
print(
    json.dumps(
        {
            "versions": {
                "vllm": importlib.metadata.version("vllm"),
                "xgrammar": importlib.metadata.version("xgrammar"),
                "torch": importlib.metadata.version("torch"),
            },
            "results": results,
        },
        sort_keys=True,
    )
)

if not all(result["passed"] for result in results):
    raise SystemExit("at least one sparse masking variant failed")
