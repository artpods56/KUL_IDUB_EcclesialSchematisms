"""Verify Gemma JSON-schema masking directly on a CUDA logits tensor."""

import importlib.metadata
import json

import torch
import xgrammar
from transformers import AutoTokenizer


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

logits = torch.zeros(
    (1, tokenizer.vocab_size),
    dtype=torch.float32,
    device="cuda",
)
logits[0, MARKDOWN_FENCE_TOKEN] = 100.0
logits[0, OPENING_BRACE_TOKEN] = 1.0
xgrammar.apply_token_bitmask_inplace(logits, mask_cpu.to("cuda"))
torch.cuda.synchronize()

result = {
    "versions": {
        "vllm": importlib.metadata.version("vllm"),
        "xgrammar": importlib.metadata.version("xgrammar"),
        "torch": importlib.metadata.version("torch"),
    },
    "tokens": {
        "markdown_fence": tokenizer.decode(
            [MARKDOWN_FENCE_TOKEN], skip_special_tokens=False
        ),
        "opening_brace": tokenizer.decode(
            [OPENING_BRACE_TOKEN], skip_special_tokens=False
        ),
    },
    "finite": {
        "markdown_fence": torch.isfinite(
            logits[0, MARKDOWN_FENCE_TOKEN]
        ).item(),
        "opening_brace": torch.isfinite(logits[0, OPENING_BRACE_TOKEN]).item(),
    },
    "logits": {
        "markdown_fence": logits[0, MARKDOWN_FENCE_TOKEN].item(),
        "opening_brace": logits[0, OPENING_BRACE_TOKEN].item(),
    },
    "argmax": logits.argmax(dim=-1).item(),
}
print(json.dumps(result, sort_keys=True))

if result["finite"]["markdown_fence"]:
    raise SystemExit("markdown fence token remained allowed")
if not result["finite"]["opening_brace"]:
    raise SystemExit("opening brace token was masked")
if result["argmax"] != OPENING_BRACE_TOKEN:
    raise SystemExit("mask did not force the opening brace token")
