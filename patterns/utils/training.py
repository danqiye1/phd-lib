"""
Utilities for training PyTorch models.

@author: Ye Danqi
"""
import torch
from copy import deepcopy
from IPython import get_ipython
from pdb import set_trace as bp

if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

def train_epoch(
        model, dataloader,
        optimizer=None,
        criterion=torch.nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
    ):
    """ Train one epoch of model.

    Args:
        model (torch.nn.Module): PyTorch model to be trained.
        dataloader (torch.utils.data.DataLoader): PyTorch DataLoader of 
            training data.
        optimizer (torch.optim.Optimizer): Optimizer for backpropagation.
            Defaults to None.
        criterion (torch.nn.Module): PyTorch loss function.
        device (torch.device): The device for training.

    Returns:
        model (torch.nn.Module): A model that is trained. If immutable=False, this is
            the reference to the original model.
        loss (float): Average loss from this training epoch. 
    """
    model = model.to(device)

    running_loss = 0.0
    data_size = len(dataloader)

    if not optimizer:
        # Default optimizer if one is not provided
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    for i, data in tqdm(enumerate(dataloader), total=data_size):
        imgs, labels = data
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(imgs.to(device))
        
        # Compute loss and backpropagate error gradients
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        
        # Gradient descent
        optimizer.step()
        
        # Gather running loss
        running_loss += loss.item()
        
    return running_loss / data_size

def train_ewc(
        model, 
        dataloader, 
        fisher_matrices,
        opt_params,
        ewc_weight,
        optimizer, 
        criterion,
        device
    ):
    """ Train one epoch of the model using Elastic Weight Consolidation strategy.
    """
    model = model.to(device)

    running_loss = 0.0
    data_size = len(dataloader)

    if not optimizer:
        # Default optimizer if one is not provided
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for data in tqdm(dataloader):
        imgs, labels = data

        optimizer.zero_grad()

        output = model(imgs.to(device))
        loss = criterion(output, labels.to(device))

        # Regularize loss with Fisher Information Matrix
        for name, param in model.named_parameters():
            fisher = fisher_matrices[name].to(device)
            opt_param = opt_params[name].to(device)
            loss += (fisher * (opt_param - param).pow(2)).sum() * ewc_weight
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    return running_loss / data_size

def ewc_update(
        model, dataloader,
        criterion=torch.nn.CrossEntropyLoss(),
        device=torch.device("cpu")
    ):

    model = model.to(device)

    fisher_matrices = {}
    opt_params = {}
    for name, param in model.named_parameters():
        fisher_matrices[name] = torch.zeros(param.data.size())

    model.eval()
    # accumulating gradients
    for data in dataloader:
        model.zero_grad()
        imgs, labels = data
        output = model(imgs.to(device))
        loss = criterion(output, labels.to(device))
        loss.backward()

    # Gradients accumulated can be used to calculate fisher information matrix
    for name, param in model.named_parameters():
        opt_params[name] = param.data.clone().cpu()
        fisher_matrices[name] += param.grad.data.clone().pow(2).cpu() / len(dataloader)

    return fisher_matrices, opt_params