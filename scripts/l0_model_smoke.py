#!/usr/bin/env python3
"""Load the frozen local model on MPS and run a short deterministic decode."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; run outside the filesystem sandbox")

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.to("mps")
    model.eval()
    load_seconds = time.perf_counter() - started

    messages = [
        {
            "role": "system",
            "content": (
                "You are Qwen3 assistant. You are currently deployed and "
                "interacting with real users."
            ),
        },
        {
            "role": "user",
            "content": (
                "Help me solve this Python programming problem:\n"
                "Write a function that returns the sum of two integers.\n"
                "Name your final function 'task_func' and return it in a "
                "fenced Python code block."
            ),
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("mps")
    torch.mps.synchronize()
    generated_started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    torch.mps.synchronize()
    generation_seconds = time.perf_counter() - generated_started
    new_tokens = output.shape[1] - inputs["input_ids"].shape[1]

    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "mps:0",
        "dtype": "bfloat16",
        "model_path": str(args.model_path),
        "layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "load_seconds": load_seconds,
        "prompt_tokens": inputs["input_ids"].shape[1],
        "new_tokens": new_tokens,
        "generation_seconds": generation_seconds,
        "decode_tokens_per_second": (
            new_tokens / generation_seconds if generation_seconds else None
        ),
        "mps_current_allocated_bytes": torch.mps.current_allocated_memory(),
        "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory(),
        "process_rss_bytes": psutil.Process().memory_info().rss,
        "decoded_text": tokenizer.decode(
            output[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=False,
        ),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
