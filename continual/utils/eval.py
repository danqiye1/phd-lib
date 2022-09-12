import torch
from torch.utils.data import DataLoader
from patterns.utils import calculate_error

def validate(
        model, dataset,
        batch_size=32,
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
    model.eval()
    dataloader = DataLoader(dataset, batch_size)
    running_vloss = []
    running_error = []
    for _, vdata in enumerate(dataloader):
        v_inputs, v_labels = vdata
        v_outputs = model(v_inputs.to(device))
        error = calculate_error(v_outputs, v_labels.to(device))
        running_error.append(error)
        v_loss = criterion(v_outputs, v_labels.to(device))
        running_vloss.append(v_loss.item())
    
    return running_vloss, running_error