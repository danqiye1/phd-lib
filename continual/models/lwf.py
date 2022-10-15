import copy
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from patterns.models import LeNetBase
from continual.models import MLP
from patterns.utils import calculate_error

class LwF(nn.Module):
    
    def __init__(self, temp=2, device=torch.device("cpu"), benchmark='SplitMNIST', architecture='lenet'):
        super(LwF, self).__init__()

        assert architecture in ['lenet', 'mlp'], "Only 'lenet', 'mlp' available as architecture choice."
        self.base = LeNetBase() if architecture == 'lenet' else MLP(output_size=84)
        self.old_head = None
        self.new_head = None
        assert benchmark in ['SplitMNIST', 'PermutedMNIST'], \
            "Benchmark must be SplitMNIST or PermutedMNIST!"
        self.benchmark = benchmark
        self.architecture = architecture

        # Keep track of which device model is on
        self.device = device

        # Temperature for knowledge distillation loss
        self.temp = temp

    def forward(self, img):
        """ Forward pass. For the first task, there is only new_out """
        X = self.base(img)
        old_out = None
        if self.old_head:
            old_out = self.old_head(X)
        new_out = self.new_head(X)
        return old_out, new_out


    def add_head(self, num_classes):
        self._consolidate()
        self.new_head = nn.Linear(84, num_classes)

    def _consolidate(self):
        """ Consolidate old head and new head into a single head """
        if self.old_head:
            with torch.no_grad():
                old_head_weights = copy.deepcopy(self.old_head.weight)
                new_head_weights = copy.deepcopy(self.new_head.weight)
                self.old_head = nn.Linear(84, old_head_weights.size(0) + new_head_weights.size(0))
                self.old_head.weight = nn.Parameter(torch.cat((old_head_weights, new_head_weights)))
        else:
            self.old_head = self.new_head

    def fit(self, dataset, batch_size=32, validate=True, valset=None):
        
        """
        Multihead model training. 
        Everything is the same as epoch training except that labels 
        for subsequent heads need to be preprocessed to start from 0.
        """
        self = self.to(self.device)
        old_model = copy.deepcopy(self)
        self.train()

        exp_id = dataset.get_current_task()

        criterion = nn.CrossEntropyLoss()

        train_loss = {task:[] for task in range(dataset.num_tasks())}
        val_loss = {task:[] for task in range(dataset.num_tasks())}
        val_error = {task:[] for task in range(dataset.num_tasks())}


        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )

        # Get the smallest label
        for data in DataLoader(dataset, len(dataset)):
            imgs, label = data
            min_label = label.min().item()

        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, weight_decay=0.0005)
        
        for data in tqdm(dataloader):
            imgs, labels = data
            
            optimizer.zero_grad()

            # Preprocess labels
            labels = labels - min_label

            # Get Y_o = CNN(Xn, theta_s, theta_o)
            # These are the distillation targets
            with torch.no_grad():
                Y_o, _ = old_model(imgs.to(self.device))
            
            # Forward pass
            Y_hat_o, Y_hat_n = self(imgs.to(self.device))
        
            # Compute loss and backpropagate error gradients
            class_loss = criterion(Y_hat_n, labels.to(self.device))
            if dataset.get_current_task():
                kd_loss = self.distil_loss(Y_hat_o, Y_o)
                loss = class_loss + kd_loss
            else:
                loss = class_loss
            loss.backward()
            
            # Gradient descent
            optimizer.step()
            
            # Gather running loss
            train_loss[dataset.get_current_task()].append(loss.item())

            # In-training validation
            if validate:
                for task in range(dataset.get_current_task() + 1):
                    valset = valset.go_to_task(task)
                    vloss, verror = self.validate(valset)
                    val_loss[task] += [vloss]
                    val_error[task] += [verror]

            self.train()
        
        return train_loss, val_loss, val_error

    def distil_loss(self, logits, target):
        pred = torch.log_softmax(logits/self.temp, dim=1)
        true_pred = torch.softmax(target/self.temp, dim=1)
        output = torch.sum(pred * true_pred, dim=1)
        return -torch.mean(output, dim=0)

    def validate(
            self, dataset,
            batch_size=32,
            criterion=torch.nn.CrossEntropyLoss()
        ):

        self = self.to(self.device)
        self.eval()
        dataloader = DataLoader(dataset, batch_size)
        running_vloss = 0.0
        running_error = 0.0
        for i, vdata in enumerate(dataloader):
            v_inputs, v_labels = vdata
            v_outputs = self(v_inputs.to(self.device))
            if self.old_head:
                v_outputs = torch.cat(v_outputs, dim=1)
            else:
                v_outputs = v_outputs[1]
            error = calculate_error(v_outputs, v_labels.to(self.device))
            running_error += error
            v_loss = criterion(v_outputs, v_labels.to(self.device))
            running_vloss += v_loss.item()
            
        avg_vloss = running_vloss / (i + 1)
        avg_verror = running_error / (i + 1)
        
        return (avg_vloss, avg_verror)

if __name__ == "__main__":
    model = LwF()
    model.add_head(2)


