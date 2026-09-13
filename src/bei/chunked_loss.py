"""Memory-bounded causal-LM cross entropy over output-vocabulary chunks."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def chunked_causal_lm_loss(
    hidden_states: Tensor,
    labels: Tensor,
    lm_head: nn.Module,
    *,
    chunk_size: int,
) -> Tensor:
    """Compute next-token mean CE without materializing all sequence logits.

    ``hidden_states[:, t]`` predicts ``labels[:, t + 1]``. Chunks containing
    only ignored labels are skipped entirely. The returned scalar matches
    PyTorch's mean cross entropy over non-ignored shifted labels.
    """

    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
    if labels.ndim != 2:
        raise ValueError("labels must have shape [batch, sequence]")
    if hidden_states.shape[:2] != labels.shape:
        raise ValueError("hidden-state and label batch/sequence dimensions differ")
    if hidden_states.shape[1] < 2:
        raise ValueError("causal loss requires at least two sequence positions")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    shifted_hidden = hidden_states[:, :-1, :]
    shifted_labels = labels[:, 1:]
    total_loss = hidden_states.new_zeros((), dtype=torch.float32)
    total_tokens = 0

    for start in range(0, shifted_hidden.shape[1], chunk_size):
        stop = min(start + chunk_size, shifted_hidden.shape[1])
        target = shifted_labels[:, start:stop]
        valid_tokens = int(target.ne(-100).sum().item())
        if valid_tokens == 0:
            continue
        logits = lm_head(shifted_hidden[:, start:stop, :]).float()
        total_loss = total_loss + F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        total_tokens += valid_tokens

    if total_tokens == 0:
        raise ValueError("labels contain no supervised next-token positions")
    return total_loss / total_tokens
