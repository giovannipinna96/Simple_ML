from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
import Data
from plotUtils import plot_decision_regions
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz

# load iris data
X_train, X_test, y_train, y_test = Data.load_split_iris(test_size=0.3, shuffle=True)

# remerge the data (used for the plot)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

# create and fit the decision tree
tree_model = DecisionTreeClassifier(criterion='gini',
                                    max_depth=4,
                                    random_state=1)
tree_model.fit(X_train[:, [2, 3]], y_train)

plot_decision_regions(X_combined[:, [2, 3]], y_combined,
                      classifier=tree_model,
                      test_idx=range(105, 150))

plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# plot the tree
dot_data = export_graphviz(tree_model,
                           filled=True,
                           rounded=True,
                           class_names=['Setosa',
                                        'Versicolor',
                                        'Virginica'],
                           feature_names=['petal length',
                                          'petal width'],
                           out_file=None)
graph = graph_from_dot_data(dot_data)
graph.write_png('tree.png')
