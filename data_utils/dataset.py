from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch


def add_noise(image):
    row, col = image.shape
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    return image + gauss


class Dataset(Dataset):

    def __init__(self, csv, mu=0, sigma=1, noise=False):
        self.annotations = pd.read_csv(csv)
        self.mu = mu
        self.sigma = sigma
        self.noise = False

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        example = self.annotations.iloc[index]
        img = (np.fromfile(example['name'], dtype=np.uint8).reshape(28, 28) - self.mu) / self.sigma
        if self.noise:
            img = add_noise(img)
        img = (img - np.min(img))/(np.max(img) - np.min(img))
        img = torch.from_numpy(img)
        img = img.unsqueeze(0).repeat(1, 1, 1)
        img = img.to(torch.float32)

        target = torch.tensor(example['class'])

        return img, target


class TorchStandardScaler:
    def __init__(self):
        self.std = None
        self.mean = None

    def fit(self, x):
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)

    def transform(self, x):
        x -= self.mean
        x /= (self.std + 1e-7)
        return x


if __name__ == "__main__":
    dataset = Dataset(csv='./data/train.csv')
    print(dataset.__getitem__(1))
