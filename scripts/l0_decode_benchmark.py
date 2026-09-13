#!/usr/bin/env python3
"""Measure warmed batched Qwen generation on a fixed coding-prompt panel."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


TASKS = [
    "Return the indices of a longest increasing subsequence of a list.",
    "Check whether two strings are anagrams under Unicode case folding.",
    "Merge overlapping closed integer intervals.",
    "Find the first non-repeating character in a string.",
    "Compute shortest paths in a nonnegative weighted directed graph.",
    "Rotate a rectangular matrix ninety degrees clockwise.",
    "Parse a CSV row supporting quoted commas and doubled quotes.",
    "Find all bridge edges in an undirected graph.",
    "Evaluate a postfix arithmetic expression with integer operands.",
    "Group words that are anagrams while preserving group order.",
    "Compute the edit distance between two strings.",
    "Return the top k most frequent integers with deterministic ties.",
    "Validate balanced brackets while ignoring brackets inside quotes.",
    "Find the maximum sum of a contiguous non-empty subarray.",
    "Perform stable topological sorting or report a cycle.",
    "Run length encode a string and decode the resulting representation.",
]


def vm_stat() -> dict[str, int]:
    result = subprocess.run(
        ["vm_stat"],
        check=True,
        capture_output=True,
        text=True,
    )
    parsed: dict[str, int] = {}
    for line in result.stdout.splitlines():
        if ":" not in line:
            continue
        name, raw_value = line.split(":", 1)
        value = raw_value.strip().rstrip(".")
        if value.isdigit():
            parsed[name.strip()] = int(value)
    return parsed


def prompts(tokenizer, *, enable_thinking: bool) -> list[str]:
    system = (
        "You are Qwen3 assistant. Solve the programming task carefully. "
        "Reason first, then return a complete Python implementation in a "
        "python code fence."
    )
    return [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": task},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        for task in TASKS
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--attn-implementation",
        choices=["sdpa", "eager"],
        default="sdpa",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--warmup-new-tokens", type=int, default=16)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to("mps")
    model.eval()

    rendered = prompts(tokenizer, enable_thinking=not args.disable_thinking)
    warmup = tokenizer(
        rendered[:1],
        return_tensors="pt",
        padding=True,
    ).to("mps")
    with torch.inference_mode():
        model.generate(
            **warmup,
            do_sample=False,
            max_new_tokens=args.warmup_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
    torch.mps.synchronize()

    process = psutil.Process()
    resident_before = process.memory_info().rss
    vm_before = vm_stat()
    started = time.perf_counter()
    decoded: list[str] = []
    lengths: list[int] = []
    prompt_widths: list[int] = []
    peak_driver_bytes = torch.mps.driver_allocated_memory()
    for start in range(0, len(rendered), args.batch_size):
        batch = tokenizer(
            rendered[start : start + args.batch_size],
            return_tensors="pt",
            padding=True,
        ).to("mps")
        prompt_width = batch["input_ids"].shape[1]
        prompt_widths.append(prompt_width)
        with torch.inference_mode():
            output = model.generate(
                **batch,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        torch.mps.synchronize()
        peak_driver_bytes = max(
            peak_driver_bytes,
            torch.mps.driver_allocated_memory(),
        )
        continuation_ids = output[:, prompt_width:]
        decoded.extend(
            tokenizer.batch_decode(continuation_ids, skip_special_tokens=False)
        )
        for row in continuation_ids:
            eos_positions = row.eq(tokenizer.eos_token_id).nonzero(
                as_tuple=False
            )
            if len(eos_positions):
                lengths.append(int(eos_positions[0].item()) + 1)
            else:
                lengths.append(int(row.shape[0]))
    torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    vm_after = vm_stat()
    resident_after = process.memory_info().rss

    total_new_tokens = sum(lengths)
    closed_think = ["</think>" in text for text in decoded]
    nonempty_fence = [
        "```" in text and bool(text.split("```", 1)[1].strip())
        for text in decoded
    ]
    report = {
        "panel_size": len(rendered),
        "batch_size": args.batch_size,
        "prompt_width_tokens_by_batch": prompt_widths,
        "thinking_enabled": not args.disable_thinking,
        "attention_implementation": args.attn_implementation,
        "max_new_tokens": args.max_new_tokens,
        "elapsed_seconds": elapsed,
        "aggregate_new_tokens": total_new_tokens,
        "aggregate_tokens_per_second": total_new_tokens / elapsed,
        "completion_lengths": lengths,
        "closed_think_fraction": sum(closed_think) / len(closed_think),
        "nonempty_fence_fraction": sum(nonempty_fence) / len(nonempty_fence),
        "mps_current_allocated_bytes": torch.mps.current_allocated_memory(),
        "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory(),
        "peak_mps_driver_allocated_bytes": peak_driver_bytes,
        "resident_bytes_before": resident_before,
        "resident_bytes_after": resident_after,
        "swapins_delta": vm_after.get("Swapins", 0) - vm_before.get("Swapins", 0),
        "swapouts_delta": vm_after.get("Swapouts", 0) - vm_before.get("Swapouts", 0),
        "outputs": decoded,
    }
    serialized = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
