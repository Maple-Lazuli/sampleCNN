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
        self.noise = noise

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        example = self.annotations.iloc[index]
        img = (np.fromfile(example['name'], dtype=np.uint8).reshape(28, 28) - self.mu) / self.sigma
        if self.noise:
            img = add_noise(img)
        img = torch.from_numpy(img)
        img = img.unsqueeze(0).repeat(1, 1, 1)
        img = img.to(torch.float32)

        target = torch.tensor(example['status']).unsqueeze(0).type(torch.float32)

        return img, target
