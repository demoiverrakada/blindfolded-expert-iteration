import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bei.projection import (
    ProjectionHookSet,
    orthonormalize,
    project_output,
    seeded_random_directions,
)


class ProjectionMathTests(unittest.TestCase):
    def test_orthonormalize(self):
        q = orthonormalize(torch.tensor([[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]))
        self.assertTrue(torch.allclose(q.T @ q, torch.eye(2), atol=1e-6))

    def test_projection_removes_basis_component(self):
        q = torch.tensor([[1.0], [0.0], [0.0]])
        x = torch.tensor([[[3.0, 4.0, 5.0]]])
        y = project_output(x, q)
        self.assertTrue(torch.allclose(y, torch.tensor([[[0.0, 4.0, 5.0]]])))
        self.assertTrue(torch.allclose(y @ q, torch.zeros(1, 1, 1)))

    def test_scale_zero_is_identity_with_same_code_path(self):
        q = seeded_random_directions(4, 1, 7)
        x = torch.randn(2, 3, 4)
        self.assertTrue(torch.equal(project_output(x, q, scale=0.0), x))

    def test_tuple_output_preserved(self):
        q = torch.tensor([[1.0], [0.0]])
        cache = object()
        output = (torch.tensor([[[2.0, 3.0]]]), cache)
        projected = project_output(output, q)
        self.assertIs(projected[1], cache)
        self.assertTrue(torch.equal(projected[0], torch.tensor([[[0.0, 3.0]]])))

    def test_dtype_is_preserved(self):
        q = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
        x = torch.tensor([[[2.0, 3.0]]], dtype=torch.float32)
        self.assertEqual(project_output(x, q).dtype, torch.float32)

    def test_dimension_mismatch_fails_closed(self):
        with self.assertRaises(ValueError):
            project_output(torch.zeros(1, 1, 3), torch.zeros(2, 1))

    def test_rank_deficient_basis_is_rejected(self):
        with self.assertRaises(ValueError):
            orthonormalize(torch.tensor([[1.0, 1.0], [0.0, 0.0]]))

    def test_gradients_are_projected(self):
        q = torch.tensor([[1.0], [0.0]])
        x = torch.tensor([[[2.0, 3.0]]], requires_grad=True)
        project_output(x, q).sum().backward()
        self.assertTrue(torch.allclose(x.grad, torch.tensor([[[0.0, 1.0]]])))

    def test_random_directions_are_deterministic(self):
        self.assertTrue(
            torch.equal(
                seeded_random_directions(8, 2, 42),
                seeded_random_directions(8, 2, 42),
            )
        )


class ToyBlock(torch.nn.Module):
    def forward(self, hidden):
        return hidden + 1


class ToyInner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([ToyBlock(), ToyBlock()])


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ToyInner()

    def forward(self, hidden):
        for layer in self.model.layers:
            hidden = layer(hidden)
        return hidden


class HookTests(unittest.TestCase):
    def test_context_manager_installs_and_removes_hook(self):
        model = ToyModel()
        q = torch.tensor([[1.0], [0.0]])
        x = torch.zeros(1, 1, 2)
        baseline = model(x)
        with ProjectionHookSet(model, {0: q}):
            intervened = model(x)
        restored = model(x)
        self.assertFalse(torch.equal(intervened, baseline))
        self.assertTrue(torch.equal(restored, baseline))

    def test_invalid_layer_does_not_partially_install(self):
        model = ToyModel()
        q = torch.tensor([[1.0], [0.0]])
        hooks = ProjectionHookSet(model, {0: q, 99: q})
        with self.assertRaises(IndexError):
            hooks.install()
        self.assertEqual(hooks.handles, [])


if __name__ == "__main__":
    unittest.main()
