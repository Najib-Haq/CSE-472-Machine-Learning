import numpy as np
from tqdm import tqdm

from utils import *

def one_hot_encoding(y, num_class):
    bs = y.shape[0]
    label = np.zeros((bs, num_class))
    label[np.arange(bs), y] = 1
    return label

def train(model, train_loader, config):
    num_class = config['num_class']
    for epoch in range(config['epochs']):
        # insert bs loader after
        labels = train_loader.iloc[:,-1:].values
        train_data = train_loader.iloc[:,:-1].values

        one_hot_labels = one_hot_encoding(labels, num_class)
        out = model(train_data)
        # print("OUT => ", out - one_hot_labels)

        model.backward(out - one_hot_labels, lr=0.001)
        loss = cross_entropy_loss(one_hot_labels, out)

        out = np.argmax(out, axis=1)
        acc = accuracy(labels, out)
        macf1 = macro_f1(labels, out, num_class)

        print(f"Epoch: {epoch} => Loss: {loss}; Acc: {acc}; MacroF1: {macf1}")
        
        model.save_model(f"model_{epoch}.pkl")
        model.load_model(f"model_{epoch}.pkl")

        break


        # if epoch == 2: break




        




