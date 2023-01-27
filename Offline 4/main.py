import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from dataset.dataset import Dataset, DataLoader, check_dataset
from model.Model import Model
from fit import *


if __name__ == "__main__":
    # read config
    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print(config)
    model = Model(config['model'])
    print(model)

    train_data = pd.read_csv("Toy Dataset/trainNN.txt", sep="\t+", header=None, engine='python')
    valid_data = pd.read_csv("Toy Dataset/testNN.txt", sep="\t+", header=None, engine='python')
    # std dev scaling
    train_data.iloc[:,:-1] = (train_data.iloc[:,:-1] - train_data.iloc[:,:-1].mean()) / train_data.iloc[:,:-1].std()
    valid_data.iloc[:,:-1] = (valid_data.iloc[:,:-1] - valid_data.iloc[:,:-1].mean()) / valid_data.iloc[:,:-1].std()
    
    print(np.unique(valid_data.iloc[:,-1:].values))
    train(model, train_data, config)

    exit(0)

    # np.random.seed(42)
    train_df = pd.read_csv('train.csv')
    valid_df = pd.read_csv('val.csv')
    print("Train: ", train_df.shape, "; Valid: ", valid_df.shape)

    
    debug = True
    if debug:
        train_df = train_df[:1000]

    train_dataset = Dataset('NumtaDB_with_aug', train_df, label_col='digit', aug=True, mode='train', use_bbox=False)
    valid_dataset = Dataset('NumtaDB_with_aug', valid_df, label_col='digit', aug=False, mode='valid', use_bbox=False)
    # check_dataset(train_dataset, valid_dataset)


    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)

    for epoch in range(2):
        for i, data in enumerate(train_loader):
            print(i, data[0].shape, data[1].shape)
        # break


