import pytest
import torch
import torch.nn.functional as F

from bei.chunked_loss import chunked_causal_lm_loss


@pytest.mark.parametrize("chunk_size", [1, 2, 4, 20])
def test_chunked_loss_matches_full_logits_and_gradients(chunk_size: int) -> None:
    generator = torch.Generator().manual_seed(17)
    hidden_full = torch.randn(2, 7, 5, generator=generator, requires_grad=True)
    hidden_chunked = hidden_full.detach().clone().requires_grad_(True)
    head_full = torch.nn.Linear(5, 11, bias=False)
    head_chunked = torch.nn.Linear(5, 11, bias=False)
    head_chunked.load_state_dict(head_full.state_dict())
    labels = torch.randint(0, 11, (2, 7), generator=generator)
    labels[0, :3] = -100
    labels[1, 4] = -100

    logits = head_full(hidden_full[:, :-1, :]).float()
    full_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    chunked_loss = chunked_causal_lm_loss(
        hidden_chunked,
        labels,
        head_chunked,
        chunk_size=chunk_size,
    )

    torch.testing.assert_close(chunked_loss, full_loss)
    full_loss.backward()
    chunked_loss.backward()
    torch.testing.assert_close(hidden_chunked.grad, hidden_full.grad)
    torch.testing.assert_close(head_chunked.weight.grad, head_full.weight.grad)


def test_chunked_loss_rejects_no_supervision() -> None:
    hidden = torch.randn(1, 4, 3)
    labels = torch.full((1, 4), -100)
    with pytest.raises(ValueError, match="no supervised"):
        chunked_causal_lm_loss(
            hidden,
            labels,
            torch.nn.Linear(3, 5, bias=False),
            chunk_size=2,
        )
