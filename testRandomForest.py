from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import Data
from plotUtils import plot_decision_regions

# load iris data
X_train, X_test, y_train, y_test = Data.load_split_iris(test_size=0.3, shuffle=True)

# remerge the data (used for the plot)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

# load and fit random forest
forest = RandomForestClassifier(criterion='gini',
                                n_estimators=25,
                                random_state=1,
                                n_jobs=-1)
forest.fit(X_train[:, [2, 3]], y_train)

# plot random forset regions
plot_decision_regions(X_combined[:, [2, 3]], y_combined,
                      classifier=forest, test_idx=range(105, 150))

plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
