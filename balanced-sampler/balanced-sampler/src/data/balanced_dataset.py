from abc import ABC, abstractmethod

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torchvision.datasets import MNIST


class BalancedDataset(Dataset, ABC):
    """
    Abstract base class to enforce returning classes.
    This is needed to allow returning n of each class per batch.
    """

    @abstractmethod
    def get_classes(self):
        pass


class BalancedMNIST(BalancedDataset):
    """
    Decorator class to add enabling function for balanced dataloader.
    """

    def __init__(self, path, train, download, transform, target_transform):
        super().__init__()
        self.dataset = MNIST(root=path,
                                   train=train,
                                   download=download,
                                   transform=transform,
                                   target_transform=target_transform)

    def get_classes(self):
        return self.dataset.targets

    def __getitem__(self, index) -> T_co:
        return self.dataset[index]
