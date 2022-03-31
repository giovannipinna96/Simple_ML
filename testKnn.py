from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
import Data
from plotUtils import plot_decision_regions
from sklearn.preprocessing import StandardScaler

# load iris data
X_train, X_test, y_train, y_test = Data.load_split_iris(test_size=0.3, shuffle=True)

# scale data
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# remerge the data (used for the plot)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# train and fit the KNN
knn = KNeighborsClassifier(n_neighbors=5,
                           p=2,
                           metric='minkowski')
knn.fit(X_train_std[:, [2, 3]], y_train)

plot_decision_regions(X_combined_std[:, [2, 3]], y_combined,
                      classifier=knn, test_idx=range(105, 150))

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
