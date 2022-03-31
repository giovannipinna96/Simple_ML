from Perceptron import Perceptron
from AdalineGD import AdalineGD
from AdalineSGD import AdalineSGD
import Data
import plotUtils

import matplotlib.pyplot as plt
import numpy as np

# create the linear models
ppn = Perceptron(eta=0.01, n_iter=10)
adaGD = AdalineGD(eta=0.01, n_iter=10)
adaSGD = AdalineSGD(eta=0.01, n_iter=10)

# load iris dataset
X, y = Data.load_iris_dataset()
X = X[0:100, [0, 2]]
y = np.where(y[0:100] == 0, -1, 1)

# standardize features
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

# fit the linear models on the train data
ppn.fit(X, y)
adaGD.fit(X_std, y)
adaSGD.fit(X_std, y)

# plot learn curves
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.title('Train curves')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()

fig, ax = plt.subplots(nrows=2, ncols=2)

ada = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
ax[0, 0].plot(range(1, len(ada.cost_) + 1), np.log10(ada.cost_), marker='o')
ax[0, 0].set_xlabel('Epochs')
ax[0, 0].set_ylabel('log(Sum-squared-error)')
ax[0, 0].set_title('AdalineGD - Learning rate 0.01')

ada1 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[0, 1].plot(range(1, len(ada1.cost_) + 1), ada1.cost_, marker='o')
ax[0, 1].set_xlabel('Epochs')
ax[0, 1].set_ylabel('Sum-squared-error')
ax[0, 1].set_title('AdalineGD - Learning rate 0.0001')

ada2 = AdalineSGD(n_iter=10, eta=0.01).fit(X, y)
ax[1, 0].plot(range(1, len(ada2.cost_) + 1), np.log10(ada2.cost_), marker='o')
ax[1, 0].set_xlabel('Epochs')
ax[1, 0].set_ylabel('log(Sum-squared-error)')
ax[1, 0].set_title('AdalineSGD - Learning rate 0.01')

ada3 = AdalineSGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1, 1].plot(range(1, len(ada3.cost_) + 1), ada3.cost_, marker='o')
ax[1, 1].set_xlabel('Epochs')
ax[1, 1].set_ylabel('Sum-squared-error')
ax[1, 1].set_title('AdalineSGD - Learning rate 0.0001')

fig.tight_layout()
plt.show()

# plot data
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

plt.show()

# plot linear classifiers
plotUtils.plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.title('Perceptron')

plt.show()

plotUtils.plot_decision_regions(X_std, y, classifier=adaGD)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

plotUtils.plot_decision_regions(X_std, y, classifier=adaSGD)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()
