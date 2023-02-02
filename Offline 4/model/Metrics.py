from sklearn.metrics import f1_score, accuracy_score

def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def macro_f1(y_true, y_pred):
    # Macro -> Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
    return f1_score(y_true, y_pred, average='macro')