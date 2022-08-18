import pytest
import torch
from utils.eval import calculate_error

@pytest.mark.parametrize("logits, labels, expected", [
    (torch.Tensor([[0.2, 0.5, 0.1, -0.2]]), torch.Tensor([1]), 0),
    (torch.Tensor([[0.2, 0.5, 0.1, -0.2], [0.2, 0.1, 0.1, -0.2]]), 
        torch.Tensor([1, 1]), 0.5),
    (torch.Tensor([[0.2, 0.5, 0.1, -0.2], [0.2, 0.1, 0.1, -0.2]]), 
        torch.Tensor([0, 1]), 1.0)
])
def test_calculate_error(logits, labels, expected):
    assert calculate_error(logits, labels) == expected
