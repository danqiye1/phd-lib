import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from patterns.utils import train_gan

class Scholar(object):

    def __init__(
            self, generator, discriminator, 
            solver, task_id, feature_size=100,
            device=torch.device("cpu")
        ):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.solver = solver.to(device)
        self.task_id = task_id
        self.device = device
        self.feature_size = feature_size

    def sample(self, batch_size):
        """ Sample a batch of replay data from Scholar """
        z = torch.randn((batch_size, self.feature_size, 1, 1), device=self.device)
        with torch.no_grad():
            x = self.generator(z)
            y = self.solver(x)
        return (x, torch.argmax(torch.softmax(y, dim=1), dim=1))

    def train_generator(self, dataset):
        """ Train the generator model """
        gloss, dloss = train_gan(self.generator, self.discriminator, dataset, device=self.device)
        return gloss, dloss

    def train_solver(
            self, dataset,
            old_scholar,
            batch_size=32,
            mix_ratio=0.5,
            lr=0.001,
            criterion=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
            validate_fn=None,
            valset=None
        ):
        """ Train the solver with a dataset """
        optimizer = torch.optim.SGD(self.solver.parameters(), lr=lr)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        replay_size = int(batch_size // mix_ratio) - batch_size

        train_loss = {task:[] for task in range(dataset.num_tasks())}
        val_loss = {task:[] for task in range(dataset.num_tasks())}
        val_error = {task:[] for task in range(dataset.num_tasks())}

        for imgs, labels in tqdm(dataloader):
            optimizer.zero_grad()
            imgs = imgs.to(device)
            labels = labels.to(device)

            # Generate replay batch
            if old_scholar:
                replay_img, replay_label = old_scholar.sample(replay_size)
                imgs = torch.cat((replay_img, imgs))
                labels = torch.cat((replay_label, labels))

            # Shuffle the replay batch
            indices = torch.randperm(len(labels))
            imgs = imgs[indices]
            labels = labels[indices]

            output = self.solver(imgs)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            # Gather running loss
            train_loss[dataset.get_current_task()].append(loss.item())

            # In-training validation
            if validate_fn:
                for task in range(dataset.get_current_task() + 1):
                    valset = valset.go_to_task(task)
                    vloss, verror = validate_fn(self.solver, valset, criterion=criterion, device=device)
                    val_loss[task] += [vloss]
                    val_error[task] += [verror]

            self.solver.train()

        return train_loss, val_loss, val_error