from sklearn import datasets
from sklearn.model_selection import train_test_split


def load_iris_dataset():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    return X, y


def load_wine_dataset():
    wine = datasets.load_wine()
    X = wine.data
    y = wine.target
    return X, y


def load_besast_cancer_wisconsin_dataset():
    beast_cancer = datasets.load_breast_cancer()
    X = beast_cancer.data
    y = beast_cancer.target
    return X, y


def load_housing_dataset():
    housing = datasets.load_boston()
    return housing


def load_split_iris(test_size=0.3, shuffle=True):
    X, y = load_iris_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
    return X_train, X_test, y_train, y_test


def load_split_wine(test_size=0.3, shuffle=True):
    X, y = load_wine_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
    return X_train, X_test, y_train, y_test


def load_split_besast_cancer_wisconsin(test_size=0.3, shuffle=True):
    X, y = load_besast_cancer_wisconsin_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
    return X_train, X_test, y_train, y_test


def load_split_housing(test_size=0.3, shuffle=True):
    X, y = load_housing_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
    return X_train, X_test, y_train, y_test
