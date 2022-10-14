import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from patterns.models import LeNetBase, LeNetHead
from patterns.utils import calculate_error
from .mlp import MLP
from tqdm import tqdm

class MultiHead(nn.Module):
    """ MultiHeadLeNet for Continual Learning """
    def __init__(
        self, device=torch.device("cpu"), 
        benchmark='SplitMNIST', 
        architecture='lenet',
        num_tasks=5
    ):
        super(MultiHead, self).__init__()
        assert architecture in ['lenet', 'mlp'], "Only 'lenet', 'mlp' available as architecture choice."
        self.base = LeNetBase() if architecture == 'lenet' else MLP(output_size=84)
        self.heads = nn.ModuleList([])

        # Initialize all heads
        for _ in range(num_tasks):
            self.heads.append(LeNetHead(2))
        self.task_pred = LeNetHead(num_tasks)

        assert benchmark in ['SplitMNIST', 'PermutedMNIST'], \
            "Benchmark must be SplitMNIST or PermutedMNIST!"
        self.benchmark = benchmark
        self.architecture = architecture

        # Keep track of which device model is on
        self.device = device
        self = self.to(device)

    def forward(self, img, task_id=0):
        """ Forward pass
        Note that training and inference takes different execution paths.
        For training, a task_id is required to also train the neural network
        to recognize which task is being fed in.
        Training returns class prediction and task prediction.
        Inference returns only class prediction.
        """
        X = self.base(img)
        task_id_pred = self.task_pred(X)
        if self.training:
            X = self.heads[task_id](X)
        else:
            task_id = torch.mode(
                        torch.argmax(
                            torch.softmax(task_id_pred, dim=1), 
                            dim=1
                        )).values.item()

            X = self.heads[task_id](X)
        return X, task_id_pred


    def fit(
        self, trainset,
        batch_size=32,
        optimizer=None,
        criterion=torch.nn.CrossEntropyLoss(),
        validate=True,
        valset=None
    ):
        """
        Multihead model training.
        """
        self.train()

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
            optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        
        for data in tqdm(dataloader):
            imgs, labels = data
            
            optimizer.zero_grad()

            # Preprocess labels
            labels = labels - min_label
            task_labels = torch.full((imgs.size(0),), trainset.get_current_task(), device=self.device)
            
            # Forward pass
            outputs, task_pred = self(imgs.to(self.device), trainset.get_current_task())
        
            # Compute loss and backpropagate error gradients
            class_loss = criterion(outputs, labels.to(self.device))
            task_loss = criterion(task_pred, task_labels)
            class_loss.backward(retain_graph=True)
            task_loss.backward()
            
            # Gradient descent
            optimizer.step()
            
            # Gather running loss
            train_loss[trainset.get_current_task()].append(class_loss.item())

            # In-training validation
            if validate:
                for task in range(trainset.get_current_task() + 1):
                    valset = valset.go_to_task(task)
                    vloss, verror = self.validate(valset, criterion=criterion)
                    val_loss[task] += [vloss]
                    val_error[task] += [verror]

            self.train()
        
        return train_loss, val_loss, val_error

    def validate(
        self, dataset,
        batch_size=32,
        criterion=torch.nn.CrossEntropyLoss()
    ):
        """Function for validating the model.

        Args:
            model (torch.nn.Module): PyTorch model to validate.
            dataloader (torch.utils.data.DataLoader): DataLoader of validation set.
            criterion (torch.nn.Module): Loss function for validation.

        Returns:
            avg_vloss (float): Average validation loss.
            avg_verror (float): Average validation error.
        """
        self.eval()
        dataloader = DataLoader(dataset, batch_size)
        running_vloss = 0.0
        running_error = 0.0

        # Get the smallest label
        for data in DataLoader(dataset, len(dataset)):
            _, label = data
            min_label = label.min().item()

        for i, vdata in enumerate(dataloader):
            v_inputs, v_labels = vdata
            v_labels = v_labels - min_label
            v_outputs, _ = self(v_inputs.to(self.device))
            error = calculate_error(v_outputs, v_labels.to(self.device))
            running_error += error

            v_loss = criterion(v_outputs, v_labels.to(self.device))
            running_vloss += v_loss.item()
            
        avg_vloss = running_vloss / (i + 1)
        avg_verror = running_error / (i + 1)
        
        return (avg_vloss, avg_verror)