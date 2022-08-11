"""
PyTorch utilities for evaluating models.

@author: Ye Danqi
"""
import torch

def calculate_error(logits, labels):
    """Calculate the error of a given set of logits and labels.
    Labels are not one-hot-encoded.

    Args:
        logits (torch.Tensor): Logits from a neural network predicting the labels
        labels (torch.Tensor): Ground truth labels

    Returns:
        float: An error rate of the prediction.
    """
    if logits.size(0) != labels.size(0):
        raise RuntimeError(f"Logits of size {logits.size()} does not match \
            labels of size {labels.size()}")
            
    predictions = torch.argmax(torch.softmax(logits, dim=1), dim=1)
    error_rate = (labels != predictions).sum() / len(labels)
    return error_rate.item()