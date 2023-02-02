import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import os
import sys

from dataset.dataset import Dataset, DataLoader, check_dataset
from model.Model import Model
from utils import set_seed, split_dataset
from fit import *


if __name__ == "__main__":
    set_seed(42)

    # read config
    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)
    os.makedirs(config['output_dir'], exist_ok=True)

    # make model
    model = Model(config['model'])
    print(model)

    # make dataset
    train_df, valid_df = split_dataset('NumtaDB_with_aug', validation_percentage=0.2)
    print("Train: ", train_df.shape, "; Valid: ", valid_df.shape)
    
    if config['debug']:
        train_df = train_df[:100]
        valid_df = valid_df[:100]

    train_dataset = Dataset('NumtaDB_with_aug', train_df, label_col='digit', mode='train', config=config['augment'])
    valid_dataset = Dataset('NumtaDB_with_aug', valid_df, label_col='digit', mode='valid', config=config['augment'])
    check_dataset(train_dataset, valid_dataset, save_dir=config['output_dir'])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)

    # train
    fit_model(model, train_loader, valid_loader, config)


