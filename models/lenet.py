import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils.dataset import Dataset
from ml_infrastructure import Model, DataManager, Manager


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 7)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 7)
        self.fc1 = nn.Linear(64, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    batch_size = 100
    train_set = Dataset(csv="./data/train.csv")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    test_set = Dataset(csv="./data/test.csv")
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False, num_workers=2)
    val_set = Dataset(csv="./data/val.csv")
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    classes = [str(x) for x in range(0, 10)]

    dm = DataManager(train_loader=train_loader, validation_loader=val_loader, test_loader=test_loader, classes=classes)

    model = Model(net=Net(), name='lenet')
    if len(classes) <= 2:
        model.criterion = torch.nn.BCEWithLogitsLoss()
    else:
        model.criterion = torch.nn.CrossEntropyLoss()

    manager = Manager(models=[model], data_manager=dm, epochs=1)
    manager.perform()
    manager.save_watcher_results(save_location='./results', save_name='lenet.json')

    manager.shutdown_watcher()
