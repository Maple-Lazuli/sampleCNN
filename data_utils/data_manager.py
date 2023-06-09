import json
from dataclasses import dataclass

import torch
from ml_infrastructure.data_manager import DataManager

from data_utils.dataset import Dataset


@dataclass
class MnistDataManager:
    batch_size: int = 5
    training_noise: bool = False
    stats: str = './data/stats.json'
    train: str = './data/train.csv'
    test: str = './data/test.csv'
    val: str = './data/val.csv'

    def __post_init__(self):
        with open(self.stats, 'r') as file_in:
            stats = json.load(file_in)

        self.train_set = Dataset(csv=self.train, mu=stats['mu'], sigma=stats['sigma'],
                                 noise=self.training_noise)
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size,
                                                        shuffle=True, num_workers=2)

        self.test_set = Dataset(csv=self.test, mu=stats['mu'], sigma=stats['sigma'])
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size,
                                                       shuffle=False, num_workers=2)

        self.val_set = Dataset(csv=self.val, mu=stats['mu'], sigma=stats['sigma'])
        self.val_loader = torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size,
                                                      shuffle=False, num_workers=2)

        self.classes = [str(x) for x in range(0, 10)]

        self.dm = DataManager(train_loader=self.train_loader, validation_loader=self.val_loader,
                              test_loader=self.test_loader, classes=self.classes)


if __name__ == "__main__":
    mdm = MnistDataManager()
    print(mdm.train_set.__getitem__(1))
