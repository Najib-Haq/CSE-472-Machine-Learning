import numpy as np
from tqdm import tqdm
import pandas as pd

from model.Loss import CrossEntropyLoss
from model.Metrics import accuracy, macro_f1
from utils import one_hot_encoding, AverageMeter, update_loggings, visualize_training

def train_one_epoch(model, train_loader, config):
    celoss = CrossEntropyLoss()

    loss_meter = AverageMeter('loss')
    acc_meter = AverageMeter('acc')
    f1_meter = AverageMeter('f1')

    model.train()
    for step, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        images, labels = data[0], data[1]
        one_hot_labels = one_hot_encoding(labels, config['num_class'])
        
        out = model(images)
        loss = celoss(out, one_hot_labels)
        model.backward(celoss.get_grad_wrt_softmax(out, one_hot_labels), lr=0.00005)

        out = np.argmax(out, axis=1)
        acc = accuracy(labels, out)
        macf1 = macro_f1(labels, out)

        loss_meter.update(loss)
        acc_meter.update(acc)
        f1_meter.update(macf1)

    return model, loss_meter.avg, acc_meter.avg, f1_meter.avg

def validate_one_epoch(model, val_loader, config):
    celoss = CrossEntropyLoss()

    loss_meter = AverageMeter('loss')
    acc_meter = AverageMeter('acc')
    f1_meter = AverageMeter('f1')

    model.eval()
    for step, data in tqdm(enumerate(val_loader), total=len(val_loader)):
        images, labels = data[0], data[1]
        one_hot_labels = one_hot_encoding(labels, config['num_class'])
        
        out = model(images)
        loss = celoss(out, one_hot_labels)

        out = np.argmax(out, axis=1)
        acc = accuracy(labels, out)
        macf1 = macro_f1(labels, out)

        loss_meter.update(loss)
        acc_meter.update(acc)
        f1_meter.update(macf1)

    return loss_meter.avg, acc_meter.avg, f1_meter.avg


def fit_model(model, train_loader, val_loader, config):
    num_class = config['num_class']

    # save based on macro f1
    best_macro_f1 = 0

    loggings = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }

    for epoch in range(config['epochs']):
        model, train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, config)
        val_loss, val_acc, val_f1 = validate_one_epoch(model, val_loader, config)

        print(f"Epoch {epoch+1}/{config['epochs']} => ")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train Macro F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f}")

        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            model.save_model(f"{config['output_dir']}/best_model_E{epoch}.npy")

        loggings = update_loggings(loggings, epoch, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1)

    loggings = pd.DataFrame(loggings)
    loggings.to_csv(f"{config['output_dir']}/logs.csv", index=False)

    visualize_training(loggings, config['output_dir'])

       



        




