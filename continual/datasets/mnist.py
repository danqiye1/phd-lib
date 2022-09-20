import os
import torch
import torchvision
from tqdm import tqdm
from PIL import Image
from copy import deepcopy

class SplitMNIST(torchvision.datasets.MNIST):
    """ SplitMNIST dataset for continual learning """

    def __init__(
        self, root, 
        train=True, 
        download=False, 
        transform=None, 
        target_transform=None,
        tasks=[[0,1],[2,3],[4,5],[6,7],[8,9]],
    ):
        """ Constructor
        
        Creates a pytorch dataset that splits the MNIST dataset into experiences.
        Each experience should only contain a subset of the classes, ensuring
        that new experience will introduce new classes of data for simulating
        class incremental learning.

        Args:
            root (str): Root directory for data.
            train (bool): Flag for downloading MNIST trainset or MNIST test set.
                Defaults to True.
            download (bool): Flag for whether to download the dataset for source.
                Defaults to False.
            transform (torch.nn.module): Transformation functions of the dataset.
            target_transform (torch.nn.module): Transformation functions of the 
                labels.
            tasks (List[List[int]]): Define how to separate the dataset into tasks.
                Each List contains the labels to be included in that task. 
                If an empty list is provided, dataset will be equivalent to
                a normal MNIST. Defaults to [[0,1,2,3,4,5],[6],[7],[8],[9]].
        """
        super().__init__(
            root, 
            train=train, 
            download=download, 
            transform=transform, 
            target_transform=target_transform
        )
        
        imgs, labels = self._load_data()
        self.tasks = tasks

        # Split dataset
        self.data = []
        self.targets = []
        for task in self.tasks:
            task_mask = torch.zeros(len(labels), dtype=bool)
            for cls in task:
                mask = (labels == cls)
                task_mask = torch.logical_or(mask, task_mask)
            self.data.append(imgs[task_mask])
            self.targets.append(labels[task_mask])

        # Start current task from 0
        self.current_task = 0

    @property
    def raw_folder(self) -> str:
        """ Overwrite this method from parent class to correctly name data folder. """
        return os.path.join(self.root, "MNIST", "raw")

    def num_tasks(self):
        """ Get the number of tasks """
        return len(self.targets)

    def num_classes(self):
        """ Get the number of classes for current task """
        return len(self.targets[self.current_task].unique())

    def get_current_task(self):
        """ Get the current task """
        return self.current_task

    def __getitem__(self, idx: int):
        """ Get item override """
        img = self.data[self.current_task][idx]
        target = self.targets[self.current_task][idx]

        # Consistent with other torchvision datasets
        # to return PIL Image if no transform
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def go_to_task(self, task_id):
        """ Proceed to given task. 
        We do deepcopy to make the dataset immutable.

        Args:
            task_id (int): Task to go to.
        """
        newset = deepcopy(self)
        newset.current_task = task_id % newset.num_tasks()
        return newset

    def next_task(self):
        """ Proceed to next task. 
        We do deepcopy to make the dataset immutable.
        
        Returns:
            new_set (SplitMNIST): A deepcopy of this dataset with incremented task.
        """
        return self.go_to_task(self.current_task + 1)

    def restart(self):
        """ Restart from task 1 
        We do deepcopy to make the dataset immutable.
        
        Returns:
            new_set (SplitMNIST): A deepcopy of this dataset with restarted task.
        """
        return self.go_to_task(0)

    def __len__(self):
        return len(self.targets[self.current_task])

    def __repr__(self):
        head = f"Dataset {self.__class__.__name__}"
        body = [
            f"Number of tasks: {len(self.data)}",
            f"Current task: {self.current_task}",
            f"Number of datapoints: {self.__len__()}",
            f"Split: {'Train' if self.train else 'Test'}"
        ]
        return head + "\n\t" + "\n\t".join(body)

class PermutedMNIST(torchvision.datasets.MNIST):
    
    def __init__(
            self, root, 
            train=True, 
            download=False, 
            transform=None, 
            target_transform=None,
            num_experiences=5,
            seed=1
        ):
        super().__init__(root, train, transform, target_transform, download)

        # Start from experience 0
        self.current_experience = 0

        # Initial load data
        imgs, labels = self._load_data()
        img_width = imgs.size(2)
        img_height = imgs.size(1)

        # Permute data for each task
        # The first experience is the original mnist
        self.experiences = [(imgs, labels)]
        torch.manual_seed(seed)

        tqdm.write("Permuting dataset")
        for _ in tqdm(range(1, num_experiences)):
            indices = torch.randperm(img_width * img_height)
            imgs = deepcopy(imgs)
            labels = deepcopy(labels)

            for i in range(len(imgs)):
                imgs[i] = imgs[i].view(-1)[indices].view((img_height, img_width))
            
            self.experiences.append((imgs, labels))


    @property
    def raw_folder(self) -> str:
        """ Overwrite this method from parent class to correctly name data folder. """
        return os.path.join(self.root, "MNIST", "raw")

    def go_to_task(self, experience):
        newset = deepcopy(self)
        newset.current_experience = experience % len(newset.experiences)
        return newset

    def num_classes(self):
        """ Get the number of classes for current task """
        return len(self.targets.unique())

    def next_task(self):
        return self.go_to_task(self.current_experience + 1)

    def restart(self):
        return self.go_to_task(0)

    def num_tasks(self):
        return len(self.experiences)

    def get_current_task(self):
        return self.current_experience

    def __len__(self):
        return len(self.experiences[self.current_experience][1])

    def __getitem__(self, idx):
        """ Get item override """
        imgs, labels = self.experiences[self.current_experience]
        img, target = imgs[idx], labels[idx]

        # Consistent with other torchvision datasets
        # to return PIL Image if no transform
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
