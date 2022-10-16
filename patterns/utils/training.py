"""
Utilities for training PyTorch models.

@author: Ye Danqi
"""
import torch
from torch.utils.data import DataLoader
from IPython import get_ipython

if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

def train_epoch(
        model, dataset,
        batch_size=32,
        optimizer=None,
        criterion=torch.nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        validate_fn=None,
        valset=None
    ):
    """ Train one epoch of model.

    Args:
        model (torch.nn.Module): PyTorch model to be trained.
        dataset (torch.utils.data.DataSet): PyTorch DataSet of 
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
    model.train()

    running_loss = 0.0

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    train_loss = {task:[] for task in range(dataset.num_tasks())}
    val_loss = {task:[] for task in range(dataset.num_tasks())}
    val_error = {task:[] for task in range(dataset.num_tasks())}

    if not optimizer:
        # Default optimizer if one is not provided
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    for data in tqdm(dataloader):
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
        
        train_loss[dataset.get_current_task()].append(loss.item())

        # In-training validation
        if validate_fn:
            for task in range(dataset.get_current_task() + 1):
                valset = valset.go_to_task(task)
                vloss, verror = validate_fn(model, valset, criterion=criterion, device=device)
                val_loss[task] += [vloss]
                val_error[task] += [verror]

        model.train()
    
    return train_loss, val_loss, val_error