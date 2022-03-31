from Perceptron import Perceptron
from AdalineSGD import AdalineSGD

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def get_classifier(clf_name, params):
    if clf_name == "Perceptron":
        clf = Perceptron(eta=params["eta"], n_iter=params["n_iter"])
    elif clf_name == "Adeline SGD":
        clf = AdalineSGD(eta=params["eta"], n_iter=params["n_iter"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    elif clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    else:
        clf = RandomForestClassifier(n_estimators=params["n_tree"], max_depth=params["max_depth"])
    return clf
