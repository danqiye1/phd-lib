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

def validate(
        model, dataloader, 
        criterion=torch.nn.CrossEntropyLoss(),
        device=torch.device("cpu")
    ):
    """Function for validating the model.

    Args:
        model (torch.nn.Module): PyTorch model to validate.
        dataloader (torch.utils.data.DataLoader): DataLoader of validation set.
        criterion (torch.nn.Module): Loss function for validation.
        device (torch.device): CPU or GPU flag for training.

    Returns:
        avg_vloss (float): Average validation loss.
        avg_verror (float): Average validation error.
    """
    model = model.to(device)
    running_vloss = 0.0
    running_error = 0.0
    for i, vdata in enumerate(dataloader):
        v_inputs, v_labels = vdata
        v_outputs = model(v_inputs.to(device))
        error = calculate_error(v_outputs, v_labels.to(device))
        running_error += error
        v_loss = criterion(v_outputs, v_labels.to(device))
        running_vloss += v_loss
        
    avg_vloss = running_vloss / (i + 1)
    avg_verror = running_error / (i + 1)
    
    return (avg_vloss.item(), avg_verror)