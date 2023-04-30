import json
import numpy as np
import pandas as pd


def read_image(location):
    return np.fromfile(location, dtype=np.uint8).reshape(28, 28)


if __name__ == "__main__":
    # Randomize Data Order

    df = pd.read_csv("./data/dataset.csv").sample(frac=1)

    # Create Train, Val, Test splits

    df_len = int(df.shape[0] * .8)
    train = df[:df_len]
    left = df[df_len:]
    left_len = int(left.shape[0] * .5)
    val = left[left_len:]
    test = left[:left_len]

    # Write CSVs to disk
    train.to_csv(f"./data/train.csv", index=False)
    train[train['class'] == 4].iloc[0:2].to_csv("./data/train.csv", index=False)
    val.to_csv(f"./data/val.csv", index=False)
    test.to_csv(f"./data/test.csv", index=False)

    # Calculate Mu and Sigma

    train['loaded_image'] = train['name'].apply(lambda x: read_image(x))
    train['image_sum'] = train['loaded_image'].apply(lambda x: np.sum(x))
    mu = sum(train['image_sum']) / (28 * 28 * train.shape[0])

    train['loaded_image'] = train['loaded_image'].apply(lambda x: (x - mu) ** 2)
    train['image_sum'] = train['image_sum'].apply(lambda x: np.sum(x))

    sigma = sum(train['image_sum']) / (28 * 28 * train.shape[0] - 1) ** .5

    # Save Mu and Sigma
    with open(f"./data/stats.json", 'w') as file_out:
        json.dump({
            'mu': mu,
            'sigma': sigma
        }, file_out)