from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import plotUtils
import Data

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

# create and fit the LINEAR svm
svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_std[:, [2, 3]], y_train)

# create and fit the NON-LINEAR svm
svm_nl = SVC(kernel='rbf', C=1.0, random_state=1, gamma=0.2)
svm_nl.fit(X_train_std[:, [2, 3]], y_train)

# plot for LINEAR svm
plotUtils.plot_decision_regions(X_combined_std[:, [2, 3]],
                                y_combined,
                                classifier=svm,
                                test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.title('SVM linear')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# plot for NON-LINEAR svm
plotUtils.plot_decision_regions(X_combined_std[:, [2, 3]],
                                y_combined,
                                classifier=svm_nl,
                                test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.title('SVM NON linear')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# create and fit the NON-LINEAR svm with high GAMMA
svm_nl_big_gamma = SVC(kernel='rbf', C=1.0, random_state=1, gamma=100.0)
svm_nl_big_gamma.fit(X_train_std[:, [2, 3]], y_train)

# plot for NON-LINEAR svm with high GAMMA regularization
plotUtils.plot_decision_regions(X_combined_std[:, [2, 3]],
                                y_combined,
                                classifier=svm_nl_big_gamma,
                                test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.title('SVM NON linear')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
