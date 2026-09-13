#!/usr/bin/env python3
"""Exercise PEFT layer resolution and projection hooks on the real Qwen model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from bei.projection import ProjectionHookSet, orthonormalize, resolve_transformer_layers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    base_layers = len(resolve_transformer_layers(model))
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
    peft_layers = len(resolve_transformer_layers(model))
    model.to("mps")
    model.train()

    encoded = tokenizer("Return the integer 7.", return_tensors="pt").to("mps")
    labels = encoded["input_ids"].clone()

    with torch.no_grad():
        baseline = model(**encoded).logits

    raw = torch.randn(2048, 1, device="mps", dtype=torch.bfloat16)
    basis = orthonormalize(raw)
    with ProjectionHookSet(model, {10: basis}, scale=0.0):
        with torch.no_grad():
            identity = model(**encoded).logits
    with ProjectionHookSet(model, {10: basis}, scale=1.0):
        with torch.no_grad():
            projected = model(**encoded).logits

    model.zero_grad(set_to_none=True)
    loss = model(**encoded, labels=labels).loss
    loss.backward()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    parameters_with_grad = sum(parameter.grad is not None for parameter in trainable)

    print(
        json.dumps(
            {
                "base_layers": base_layers,
                "peft_layers": peft_layers,
                "hidden_size": model.config.hidden_size,
                "trainable_parameters": sum(p.numel() for p in trainable),
                "trainable_tensors": len(trainable),
                "trainable_tensors_with_grad": parameters_with_grad,
                "loss": float(loss.detach().cpu()),
                "scale_zero_max_logit_delta": float(
                    (baseline - identity).abs().max().cpu()
                ),
                "scale_one_max_logit_delta": float(
                    (baseline - projected).abs().max().cpu()
                ),
                "basis_device": str(basis.device),
                "basis_dtype": str(basis.dtype),
                "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
