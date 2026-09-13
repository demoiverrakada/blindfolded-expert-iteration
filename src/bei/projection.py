"""Residual-stream projection hooks used during training and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch


def orthonormalize(directions: torch.Tensor) -> torch.Tensor:
    """Return a basis in the input dtype/device, using CPU fp32 factorization."""

    if directions.ndim == 1:
        directions = directions[:, None]
    if directions.ndim != 2:
        raise ValueError("directions must have shape [hidden] or [hidden, rank]")
    if directions.shape[1] == 0:
        return directions
    if not torch.isfinite(directions).all():
        raise ValueError("directions contain non-finite values")
    cpu_directions = directions.detach().to(device="cpu", dtype=torch.float32)
    if torch.linalg.matrix_rank(cpu_directions) < directions.shape[1]:
        raise ValueError("directions are rank deficient")
    q, _ = torch.linalg.qr(cpu_directions, mode="reduced")
    return q.to(device=directions.device, dtype=directions.dtype)


def seeded_random_directions(
    hidden_size: int,
    rank: int,
    seed: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if hidden_size <= 0 or rank <= 0 or rank > hidden_size:
        raise ValueError("Require 0 < rank <= hidden_size")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    raw = torch.randn(hidden_size, rank, generator=generator, dtype=torch.float32)
    return orthonormalize(raw).to(device=device, dtype=dtype)


def project_output(
    output,
    basis: torch.Tensor,
    *,
    scale: float = 1.0,
):
    """Subtract a basis projection from a tensor or block-output tuple."""

    activation = output[0] if isinstance(output, tuple) else output
    if not torch.is_tensor(activation):
        raise TypeError("Block output must be a tensor or tuple beginning with one")
    if activation.shape[-1] != basis.shape[0]:
        raise ValueError(
            f"Hidden dimension {activation.shape[-1]} does not match "
            f"basis dimension {basis.shape[0]}"
        )
    q = basis.detach().to(device=activation.device, dtype=torch.float32)
    activation_float = activation.to(torch.float32)
    projected = (
        activation_float - scale * ((activation_float @ q) @ q.transpose(0, 1))
    ).to(activation.dtype)
    if isinstance(output, tuple):
        return (projected,) + output[1:]
    return projected


def resolve_transformer_layers(model):
    """Resolve common Hugging Face/PEFT paths to the decoder layer list."""

    candidates = (
        ("module", "model", "model", "layers"),
        ("module", "model", "layers"),
        ("module", "base_model", "model", "model", "layers"),
        ("model", "model", "layers"),
        ("model", "layers"),
        ("base_model", "model", "model", "layers"),
        ("base_model", "model", "layers"),
    )
    for path in candidates:
        current = model
        try:
            for component in path:
                current = getattr(current, component)
        except AttributeError:
            continue
        if hasattr(current, "__len__") and hasattr(current, "__getitem__"):
            return current
    raise AttributeError("Could not resolve transformer decoder layers")


@dataclass
class ProjectionHookSet:
    """Manage post-block hooks and guarantee cleanup."""

    model: object
    bases_by_layer: Mapping[int, torch.Tensor]
    scale: float = 1.0
    handles: list = field(default_factory=list, init=False)

    def install(self) -> "ProjectionHookSet":
        if self.handles:
            raise RuntimeError("Projection hooks are already installed")
        layers = resolve_transformer_layers(self.model)
        invalid = [
            index
            for index in self.bases_by_layer
            if index < 0 or index >= len(layers)
        ]
        if invalid:
            raise IndexError(
                f"Layers {invalid} outside model range [0, {len(layers)})"
            )
        for layer_index, basis in sorted(self.bases_by_layer.items()):
            def hook(_module, _inputs, output, *, q=basis, scale=self.scale):
                return project_output(output, q, scale=scale)

            self.handles.append(layers[layer_index].register_forward_hook(hook))
        return self

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self) -> "ProjectionHookSet":
        return self.install()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.remove()
