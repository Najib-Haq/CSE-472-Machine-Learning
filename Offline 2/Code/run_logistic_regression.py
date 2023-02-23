"""
main code that you will run
"""

from linear_model import LogisticRegression
from data_handler import load_dataset, split_dataset, set_seed
from metrics import precision_score, recall_score, f1_score, accuracy


if __name__ == '__main__':
    set_seed()
    # data load
    X, y = load_dataset("data_banknote_authentication.csv", "isoriginal")
    # X, y = load_dataset("HTRU_2.csv", "class")

    # split train and test
    X_train, y_train, X_test, y_test = split_dataset(X, y, test_size=0.2, shuffle=True)
    

    # training
    params = {
        'n_features': X_train.shape[1],
        'random_init': True,
        'learning_rate': 0.001,
        'iterations': 1500,
        'threshold': 0.5,
    }
    classifier = LogisticRegression(params)
    classifier.fit(X_train, y_train)

    # testing
    y_pred = classifier.predict(X_test)

    # performance on test set
    print('Accuracy ', accuracy(y_true=y_test, y_pred=y_pred))
    print('Recall score ', recall_score(y_true=y_test, y_pred=y_pred))
    print('Precision score ', precision_score(y_true=y_test, y_pred=y_pred))
    print('F1 score ', f1_score(y_true=y_test, y_pred=y_pred))
