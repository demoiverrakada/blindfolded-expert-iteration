#!/usr/bin/env python3
"""Benchmark real Qwen LoRA training with memory-bounded vocabulary loss."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import psutil
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from bei.chunked_loss import chunked_causal_lm_loss


BENCHMARK_TEXT = """
You are solving a Python programming task. Explain the algorithm carefully,
consider edge cases and complexity, then return a complete implementation in a
python code fence. The implementation should be readable, deterministic, and
avoid unnecessary dependencies. Given a sequence of integers, return the
length and indices of a longest increasing subsequence. If several answers
exist, return the lexicographically smallest index sequence.
"""


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


def build_batch(tokenizer, sequence_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    token_ids = tokenizer.encode(BENCHMARK_TEXT, add_special_tokens=False)
    if not token_ids:
        raise RuntimeError("benchmark text tokenized to an empty sequence")
    repeats = (sequence_length + len(token_ids) - 1) // len(token_ids)
    input_ids = torch.tensor(
        [(token_ids * repeats)[:sequence_length]],
        dtype=torch.long,
    )
    labels = input_ids.clone()
    prefix = sequence_length // 3
    labels[:, :prefix] = -100
    return input_ids, labels


def load_model(model_path: Path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    model.to("mps")
    model.train()
    return model


def run_length(
    model_path: Path,
    tokenizer,
    sequence_length: int,
    *,
    warmup_steps: int,
    measured_steps: int,
    accumulation_steps: int,
    chunk_size: int,
) -> dict[str, object]:
    torch.mps.empty_cache()
    model = load_model(model_path)
    base_model = model.get_base_model()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=2e-4,
    )
    input_ids, labels = build_batch(tokenizer, sequence_length)
    input_ids = input_ids.to("mps")
    labels = labels.to("mps")
    supervised_per_microbatch = int(labels[:, 1:].ne(-100).sum().item())

    def micro_step() -> float:
        outputs = base_model.model(
            input_ids=input_ids,
            use_cache=False,
            return_dict=True,
        )
        loss = chunked_causal_lm_loss(
            outputs.last_hidden_state,
            labels,
            base_model.lm_head,
            chunk_size=chunk_size,
        )
        (loss / accumulation_steps).backward()
        return float(loss.detach().cpu())

    for _ in range(warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accumulation_steps):
            micro_step()
        optimizer.step()
    torch.mps.synchronize()

    vm_before = vm_stat()
    process = psutil.Process()
    resident_before = process.memory_info().rss
    losses: list[float] = []
    step_seconds: list[float] = []

    for _ in range(measured_steps):
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        for _ in range(accumulation_steps):
            losses.append(micro_step())
        optimizer.step()
        torch.mps.synchronize()
        step_seconds.append(time.perf_counter() - started)

    vm_after = vm_stat()
    resident_after = process.memory_info().rss
    elapsed = sum(step_seconds)
    supervised_tokens = (
        measured_steps * accumulation_steps * supervised_per_microbatch
    )
    total_tokens = measured_steps * accumulation_steps * sequence_length
    result = {
        "sequence_length": sequence_length,
        "warmup_steps": warmup_steps,
        "measured_optimizer_steps": measured_steps,
        "gradient_accumulation_steps": accumulation_steps,
        "chunk_size": chunk_size,
        "supervised_tokens_per_microbatch": supervised_per_microbatch,
        "mean_loss": sum(losses) / len(losses),
        "elapsed_seconds": elapsed,
        "mean_seconds_per_optimizer_step": elapsed / measured_steps,
        "total_tokens_per_second": total_tokens / elapsed,
        "supervised_tokens_per_second": supervised_tokens / elapsed,
        "mps_current_allocated_bytes": torch.mps.current_allocated_memory(),
        "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory(),
        "resident_bytes_before": resident_before,
        "resident_bytes_after": resident_after,
        "pageins_delta": vm_after.get("Pageins", 0) - vm_before.get("Pageins", 0),
        "pageouts_delta": vm_after.get("Pageouts", 0) - vm_before.get("Pageouts", 0),
        "swapins_delta": vm_after.get("Swapins", 0) - vm_before.get("Swapins", 0),
        "swapouts_delta": vm_after.get("Swapouts", 0) - vm_before.get("Swapouts", 0),
        "step_seconds": step_seconds,
    }
    del optimizer, model, base_model, input_ids, labels
    torch.mps.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--lengths", type=int, nargs="+", default=[1536, 2048])
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measured-steps", type=int, default=50)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    if platform.system() != "Darwin":
        raise RuntimeError("this benchmark expects Apple MPS")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
    )
    report = {
        "model_path": str(args.model_path),
        "torch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "results": [
            run_length(
                args.model_path,
                tokenizer,
                sequence_length,
                warmup_steps=args.warmup_steps,
                measured_steps=args.measured_steps,
                accumulation_steps=args.accumulation_steps,
                chunk_size=args.chunk_size,
            )
            for sequence_length in args.lengths
        ],
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
