import os
import torch
import torchvision

from pdb import set_trace as bp

class SplitMNIST(torchvision.datasets.MNIST):
    """ SplitMNIST dataset for continual learning """

    def __init__(
        self, root, 
        train=True, 
        download=False, 
        transform=None, 
        target_transform=None,
        tasks=[[0,1,2,3,4,5],[6],[7],[8],[9]],
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

    def __getitem__(self, idx: int):
        """ Get item override """
        img = self.data[self.current_task][idx]
        label = self.targets[self.current_task][idx]
        return self.transform(img), label

    def next_task(self):
        """ Proceed to next task """
        self.current_task += 1

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