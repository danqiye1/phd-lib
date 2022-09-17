"""
Utilities for training Continual Learning models.

@author: Ye Danqi
"""
import random
import torch
from torch.utils.data import DataLoader
from copy import deepcopy
from IPython import get_ipython

if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

def train_ewc(
        model, trainset,
        fisher_matrices,
        opt_params,
        batch_size=32,
        ewc_weight=0.4,
        optimizer=None, 
        criterion=torch.nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        validate_fn=None,
        valset=None
    ):
    """ Train one epoch of the model using Elastic Weight Consolidation strategy.

    Args:
        model (torch.nn.Module): PyTorch model to be trained.
        trainset (continual.SplitMNIST): SplitMNIST dataset used for 
            benchmarking continual learning.
        fisher_matrices (dict{dict{torch.Tensor}}): Fisher information matrices.
        opt_params (dict{dict{torch.Tensor}}): Optimal parameters from previous tasks.
        batch_size (int): Desired batch size for minibatch. Note that we will split
            this equally among the current tasks and previous tasks.
        ewc_weight (float): Weight of consolidation regularizer.
        optimizer (torch.optim.Optimizer): Optimizer for backpropagation.
            Defaults to None.
        criterion (torch.nn.Module): PyTorch loss function.
        device (torch.device): The device for training.
        validate_fn (): Validation function for on-the-fly validation.
        valset (continual.SplitMNIST): Evaluation dataset for on-the-fly benchmarking.

    Returns:
        train_loss (list[float]): List of training losses for each batch.
        val_loss (list[float]): List of validation losses for each batch.
        val_error (list[float]): List of validation errors for each batch.
    """
    model = model.to(device)
    model.train()

    train_loss = {task:[] for task in range(trainset.num_tasks())}
    val_loss = {task:[] for task in range(trainset.num_tasks())}
    val_error = {task:[] for task in range(trainset.num_tasks())}

    dataloader = DataLoader(trainset, batch_size, shuffle=True)

    if not optimizer:
        # Default optimizer if one is not provided
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for data in tqdm(dataloader):
        imgs, labels = data

        optimizer.zero_grad()

        output = model(imgs.to(device))
        loss = criterion(output, labels.to(device))

        # Regularize loss with Fisher Information Matrix
        for task in range(trainset.get_current_task()):
            for name, param in model.named_parameters():
                fisher = fisher_matrices[task][name]
                opt_param = opt_params[task][name]
                penalty = (fisher * (opt_param - param).pow(2)).sum() * ewc_weight
                loss += penalty.to(device)
        
        loss.backward()
        optimizer.step()
        train_loss[trainset.get_current_task()].append(loss.item())

        # In-training validation
        if validate_fn:
            for task in range(trainset.get_current_task() + 1):
                valset = valset.go_to_task(task)
                vloss, verror = validate_fn(model, valset, criterion=criterion, device=device)
                val_loss[task] += [vloss]
                val_error[task] += [verror]
    
    return train_loss, val_loss, val_error

def ewc_update(
        model, dataset,
        batch_size=32,
        optimizer=None,
        criterion=torch.nn.CrossEntropyLoss(),
        device=torch.device("cpu")
    ):

    model = model.to(device)
    optimizer.zero_grad()

    dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True, 
        )

    opt_params = {}
    fisher_matrices = {}

    model.train()
    # accumulating gradients
    for data in dataloader:
        imgs, labels = data
        output = model(imgs.to(device))
        loss = criterion(output, labels.to(device))
        loss.backward()
        optimizer.zero_grad()

    # Gradients accumulated can be used to calculate Fisher Information Matrix (FIM)
    # We only want the diagonals of the FIM which is just the square of our gradients.
    for name, param in model.named_parameters():
        opt_params[name] = param.data.clone()
        fisher_matrices[name] = param.grad.data.clone().pow(2) / len(dataset)

    return fisher_matrices, opt_params

def rehearsal(
        model, trainset,
        batch_size=32,
        optimizer=None,
        criterion=torch.nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        validate_fn=None,
        valset=None
    ):
    """ Training one epoch using the rehearsal strategy to mitigate catastrophic forgetting.

    Args:
        model (torch.nn.Module): PyTorch model to be trained.
        dataset (continual.SplitMNIST): SplitMNIST dataset used for 
            benchmarking continual learning.
        batch_size (int): Desired batch size for minibatch. Note that we will split
            this equally among the current tasks and previous tasks.
        optimizer (torch.optim.Optimizer): Optimizer for backpropagation.
            Defaults to None.
        criterion (torch.nn.Module): PyTorch loss function.
        device (torch.device): The device for training.

    Returns:
        loss (float): Average loss from this training epoch. 
    """
    model = model.to(device)

    train_loss = {task:[] for task in range(trainset.num_tasks())}
    val_loss = {task:[] for task in range(trainset.num_tasks())}
    val_error = {task:[] for task in range(trainset.num_tasks())}

    current_task = trainset.get_current_task()
    task_batch_size = batch_size // (current_task + 1)

    # Get dataset of all prev task
    prevsets = []
    for task in range(current_task):
        prevsets.append(trainset.go_to_task(task))

    # Initialize a dataloader
    trainloader = DataLoader(
                    dataset=trainset, 
                    batch_size=task_batch_size, 
                    shuffle=True, 
                    num_workers=4)

    if not optimizer:
        # Default optimizer if one is not provided
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for data in tqdm(trainloader):
        imgs, labels = data

        # Sample datasets of prev tasks and mix
        for prev_data in prevsets:
            indices = random.sample(range(0, len(prev_data)), task_batch_size)
            for idx in indices:
                prev_img, prev_label = prev_data[idx]
                imgs = torch.cat((imgs, prev_img.unsqueeze(0)))
                labels = torch.cat((labels, prev_label.unsqueeze(0)))
        
        # Permute the minibatch
        indices = torch.randperm(len(labels))
        imgs = imgs[indices]
        labels = labels[indices]

        optimizer.zero_grad()

        output = model(imgs.to(device))
        loss = criterion(output, labels.to(device))
        
        loss.backward()
        optimizer.step()

        train_loss[trainset.get_current_task()].append(loss.item())

        # In-training validation
        if validate_fn:
            for task in range(trainset.get_current_task() + 1):
                valset = valset.go_to_task(task)
                vloss, verror = validate_fn(model, valset, criterion=criterion, device=device)
                val_loss[task] += [vloss]
                val_error[task] += [verror]
    
    return train_loss, val_loss, val_error

def pseudo_rehearsal(
        model, trainset,
        batch_size=32,
        optimizer=None,
        criterion=torch.nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        mode="uniform",
        dist_params=(0.1307, 0.3081),
        validate_fn=None,
        valset=None
    ):
    """ Training one epoch using the pseudo rehearsal strategy to mitigate catastrophic forgetting.

    In this strategy, the model will be used to generate labels given a random input. 
    This (input,label) pair will then be batched with the new data to train the new data without
    forgetting.

    Args:
        model (torch.nn.Module): PyTorch model to be trained.
        dataset (continual.SplitMNIST): SplitMNIST dataset used for 
            benchmarking continual learning.
        batch_size (int): Desired batch size for minibatch. Note that we will split
            this equally among the current tasks and previous tasks.
        optimizer (torch.optim.Optimizer): Optimizer for backpropagation.
            Defaults to None.
        criterion (torch.nn.Module): PyTorch loss function.
        device (torch.device): The device for training.
        mode (str): PDF to use to generate pseudo-items. Can be "uniform" or "normal".
            If "normal" is used, parameters of mean and std can be supplied.
        dist_params (tuple): Tuple of (mean, std) to be used if mode="normal".

    Returns:
        loss (float): Average loss from this training epoch. 
    """
    model = model.to(device)

    # Make a copy of old model for pseudo item synthesis
    old_model = deepcopy(model)

    current_task = trainset.get_current_task()

    train_loss = {task:[] for task in range(trainset.num_tasks())}
    val_loss = {task:[] for task in range(trainset.num_tasks())}
    val_error = {task:[] for task in range(trainset.num_tasks())}

    # Get current task's batch size and the number of
    # pseudo items to generate.
    task_batch_size = batch_size // (current_task + 1)
    num_items = batch_size - task_batch_size

    # Initialize a dataloader
    trainloader = DataLoader(
                    dataset=trainset, 
                    batch_size=task_batch_size, 
                    shuffle=True, 
                    num_workers=4)

    if not optimizer:
        # Default optimizer if one is not provided
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for data in tqdm(trainloader):
        imgs, labels = data
        imgs, labels = imgs.to(device), \
            torch.nn.functional.one_hot(labels, 10)
        labels = labels.type(torch.FloatTensor).to(device)
        img_dims = imgs[0].unsqueeze(0).size()

        # Generate pseudoitems and mix
        for _ in range(num_items):
            if mode == "uniform":
                item = torch.rand(img_dims, device=device)
            elif mode == "normal":
                item = torch.normal(
                            mean=torch.ones(img_dims) * dist_params[0], 
                            std=torch.ones(img_dims) * dist_params[1],
                        ).to(device)
            pseudolabel = old_model(item)
            # Need to softmax the pseudolabels into valid probs
            pseudolabel = torch.softmax(pseudolabel, dim=1)
            imgs = torch.cat((imgs, item))
            labels = torch.cat((labels, pseudolabel))
        
        # Permute the minibatch
        indices = torch.randperm(len(labels))
        imgs = imgs[indices]
        labels = labels[indices]

        optimizer.zero_grad()

        output = model(imgs)
        loss = criterion(output, labels)
        
        loss.backward()
        optimizer.step()

        train_loss[trainset.get_current_task()].append(loss.item())

        # In-training validation
        if validate_fn:
            for task in range(trainset.get_current_task() + 1):
                valset = valset.go_to_task(task)
                vloss, verror = validate_fn(model, valset, criterion=criterion, device=device)
                val_loss[task] += [vloss]
                val_error[task] += [verror]
    
    return train_loss, val_loss, val_error

def train_multihead(
        model, trainset,
        batch_size=32,
        optimizer=None,
        criterion=torch.nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        validate_fn=None,
        valset=None
    ):
    """
    Multihead model training. 
    Everything is the same as epoch training except that labels 
    for subsequent heads need to be preprocessed to start from 0.
    """
    model = model.to(device)
    model.train()

    train_loss = {task:[] for task in range(trainset.num_tasks())}
    val_loss = {task:[] for task in range(trainset.num_tasks())}
    val_error = {task:[] for task in range(trainset.num_tasks())}


    dataloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    # Get the smallest label
    for data in DataLoader(trainset, len(trainset)):
        _, label = data
        min_label = label.min().item()

    if not optimizer:
        # Default optimizer if one is not provided
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    for data in tqdm(dataloader):
        imgs, labels = data
        
        optimizer.zero_grad()

        # Preprocess labels
        labels = labels - min_label
        
        # Forward pass
        outputs = model(imgs.to(device))
    
        # Compute loss and backpropagate error gradients
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        
        # Gradient descent
        optimizer.step()
        
        # Gather running loss
        train_loss[trainset.get_current_task()].append(loss.item())

        # In-training validation
        if validate_fn:
            for task in range(trainset.get_current_task() + 1):
                valset = valset.go_to_task(task)
                vloss, verror = validate_fn(model, valset, criterion=criterion, device=device)
                val_loss[task] += [vloss]
                val_error[task] += [verror]
    
    return train_loss, val_loss, val_error