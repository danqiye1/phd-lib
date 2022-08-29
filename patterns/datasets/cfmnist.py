import torchvision

class CFMNIST(torchvision.datasets.MNIST):
    
    def __init__(
        self, root, 
        train=True, 
        download=False, 
        transform=None, 
        target_transform=None,
        holdout_label=9,
        holdout=True,
    ):
        super().__init__(
            root, 
            train=train, 
            download=download, 
            transform=transform, 
            target_transform=target_transform
        )
        
        imgs, labels = self._load_data()
        
        if holdout:
            mask = labels != holdout_label
        else:
            mask = labels == holdout_label
            
        self.data = imgs[mask]
        self.targets = labels[mask]