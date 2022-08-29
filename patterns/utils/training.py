"""
Utilities for training PyTorch models.

@author: Ye Danqi
"""
import torch
from IPython import get_ipython

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
        task_id,
        fisher_dict,
        opt_param_dict,
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

        output = model(imgs)
        loss = criterion(output, labels)

        # Regularize loss with Fisher Information Matrix
        for task in range(task_id):
            for name, param in model.named_parameters():
                fisher = fisher_dict[task][name]
                opt_param = opt_param_dict[task][name]
                loss += (fisher * (opt_param - param).pow(2)).sum() * ewc_weight
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    return running_loss / data_size