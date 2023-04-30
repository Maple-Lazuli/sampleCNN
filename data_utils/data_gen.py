from keras.datasets import mnist
import pandas as pd
import numpy as np
import hashlib

if __name__ == "__main__":
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    examples = []
    counter = 0
    md5 = hashlib.md5()
    for x, y in zip(train_X, train_y):
        md5.update(bytes(counter))
        digest = md5.hexdigest()

        file_name = f"./data/images/{digest}.npy"

        x.tofile(f"{file_name}")

        example = {
            'name': file_name,
            'class': y
        }

        examples.append(example)

        counter += 1

    for x, y in zip(test_X, test_y):
        md5.update(bytes(counter))
        digest = md5.hexdigest()

        file_name = f"./data/images/{digest}.npy"

        x.tofile(f"{file_name}")

        example = {
            'name': file_name,
            'class': y
        }

        examples.append(example)

        counter += 1

    df = pd.DataFrame.from_dict(examples)

    df.to_csv("./data/dataset.csv")
