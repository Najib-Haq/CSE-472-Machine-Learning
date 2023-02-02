import numpy as np
import pandas as pd

def split_dataset(parent_dir="NumtaDB_with_aug", validation_percentage=0.2):
    df1 = pd.read_csv(f"{parent_dir}/training-a.csv")
    df2 = pd.read_csv(f"{parent_dir}/training-b.csv")
    df3 = pd.read_csv(f"{parent_dir}/training-c.csv")
    df4 = pd.read_csv(f"{parent_dir}/training-d.csv")

    df = pd.concat([df1, df2, df3], ignore_index=True)

    df['split_col'] = df['database name original'] + '_' + df['digit'].astype(str)
    df = df.sample(frac=1) # shuffle

    split_col = df['split_col'].unique().tolist()
    train_indexes = []
    for cat in split_col:
        indexes = df[df['split_col'] == cat].index.tolist()
        train_indexes.extend(indexes[:int(len(indexes) * (1 - validation_percentage))])
    train_df = df.loc[train_indexes]
    val_df = df.drop(train_indexes)
    print("Train: ", train_df.shape, "; Valid: ", val_df.shape)

    # save csv
    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("val.csv", index=False)
    return train_df, val_df


def one_hot_encoding(y, num_class):
    bs = y.shape[0]
    label = np.zeros((bs, num_class))
    label[np.arange(bs), y] = 1
    return label

def set_seed(seed):
    np.random.seed(seed)


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, name):
        self.name = name
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count