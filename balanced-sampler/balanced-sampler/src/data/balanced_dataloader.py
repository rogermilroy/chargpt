from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST

class BalancedMNISTDataloader(DataLoader):
# load MNIST dataset
    def __init__(self, path, download, transform, target_transform):
        super().__init__()
        self.train_dataset = MNIST(root=path,
                                   train=True,
                                   download=download,
                                   transform=transform,
                                   target_transform=target_transform)
        self.test_dataset = MNIST(root=path,
                                  train=False,
                                  download=download,
                                  transform=transform,
                                  target_transform=target_transform)


# okay so new plan is to implement a custom batch (balanced batch) this may or may not need a new dataset which i can do above..