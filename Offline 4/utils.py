import numpy as np

def cross_entropy_loss(y_true, y_pred):
    # TODO: check
    return np.sum(-np.sum(y_true * np.log(y_pred), axis=1))

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision_score(y_true, y_pred, class_label):
    tp = np.sum((y_true == class_label) & (y_pred == class_label))
    fp = np.sum((y_true != class_label) & (y_pred == class_label))
    return tp / (tp + fp)

def recall_score(y_true, y_pred, class_label):
    tp = np.sum((y_true == class_label) & (y_pred == class_label))
    fn = np.sum((y_true == class_label) & (y_pred != class_label))
    return tp / (tp + fn)

def f1_score(y_true, y_pred, class_label):
    precision = precision_score(y_true, y_pred, class_label)
    recall = recall_score(y_true, y_pred, class_label)
    return 2 * precision * recall / (precision + recall)

def macro_f1(y_true, y_pred, num_classes):
    # TODO: doing nanmean here
    return np.nanmean([f1_score(y_true, y_pred, cls) for cls in range(num_classes)])


